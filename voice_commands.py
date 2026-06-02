# voice_commands.py — Voice command listener for Bong
#
# Two-stage voice pipeline:
#   1. Receives 48kHz stereo PCM from Discord via discord.ext.voice_recv
#   2. Feeds 16kHz mono to openWakeWord for lightweight wake word detection
#   3. After wake word fires, buffers audio and transcribes with Whisper
#   4. Strips wake word and forwards command through the LLM tool loop
#
# DAVE encryption is handled at multiple levels:
#   - venv patches to reader.py: dave_strip_success, parse_fail, passthrough
#   - venv patch to opus.py: FEC decode skips DAVE-encrypted next packets
#   - Sink-level: drops frames with _voice_recv_needs_dave_inner_decrypt
#   - Peak detection: drops frames with |sample| > 15000 (garbage detection)
#
# Models (Whisper, openWakeWord) are loaded when voice commands start
# and unloaded when voice commands stop, to minimize RAM usage.

import asyncio
import gc
import io
import time
import wave
from collections import defaultdict
from pathlib import Path
from threading import Lock

import numpy as np
import debug
import user_data

_LOG_FILE = Path(__file__).parent / "logs" / "voice_commands.log"
_DEBUG_WAV_DIR = Path(__file__).parent / "logs" / "voice_debug"
_MAX_DEBUG_WAVS = 10

_model_lock = Lock()

_whisper_model = None
_WHISPER_MODEL_SIZE = "small"
_WHISPER_DOWNLOAD_ROOT = str(Path(__file__).parent / "whisper_models")

_oww_model = None
_OWW_WAKE_WORD = "hey_bong"
_OWW_THRESHOLD = 0.5


def _load_models():
    """Load both Whisper and openWakeWord models into RAM."""
    global _whisper_model, _oww_model
    with _model_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _vlog(f"Loading whisper model ({_WHISPER_MODEL_SIZE})...")
            _whisper_model = WhisperModel(
                _WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                download_root=_WHISPER_DOWNLOAD_ROOT,
                cpu_threads=4,
                num_workers=1,
            )
            _vlog("Whisper model loaded")
        if _oww_model is None:
            import warnings
            warnings.filterwarnings("ignore", message="Specified provider.*not in available")
            from openwakeword.model import Model
            custom_model = Path(__file__).parent / "wakeword_models" / "hey_bong.onnx"
            if custom_model.exists():
                _vlog(f"Loading custom openWakeWord model ({custom_model})...")
                _oww_model = Model(wakeword_model_paths=[str(custom_model)], vad_threshold=0.5)
            else:
                import openwakeword
                oww_file = openwakeword.__file__
                if oww_file is None:
                    raise RuntimeError("Cannot locate openwakeword package directory")
                model_dir = Path(oww_file).parent / "resources" / "models"
                model_path = model_dir / "hey_jarvis_v0.1.onnx"
                _vlog("Loading fallback openWakeWord model...")
                _oww_model = Model(wakeword_model_paths=[str(model_path)], vad_threshold=0.5)
            _vlog("openWakeWord model loaded")


def _unload_models():
    """Free both Whisper and openWakeWord models from RAM."""
    global _whisper_model, _oww_model
    with _model_lock:
        _whisper_model = None
        _oww_model = None
    gc.collect()
    _vlog("Unloaded voice models")


def _get_whisper_model():
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            _load_models()
        return _whisper_model


def _get_oww_model():
    global _oww_model
    with _model_lock:
        if _oww_model is None:
            _load_models()
        return _oww_model


def _vlog(msg: str):
    debug.log("VoiceCmd", msg)
    if debug.is_debug():
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            ts = time.strftime("%H:%M:%S")
            f.write(f"[{ts}] {msg}\n")


def _write_debug_wav(pcm_48k_stereo: bytes, pcm_16k_mono: bytes, user_id: int) -> None:
    if not debug.is_debug():
        return
    try:
        _DEBUG_WAV_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%H%M%S")
        raw_path = _DEBUG_WAV_DIR / f"raw_{ts}_{user_id}.wav"
        rst_path = _DEBUG_WAV_DIR / f"resampled_{ts}_{user_id}.wav"

        with wave.open(str(raw_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_48k_stereo)

        with wave.open(str(rst_path), "wb") as wf:
            wf.setnchannels(WHISPER_CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(WHISPER_SAMPLE_RATE)
            wf.writeframes(pcm_16k_mono)

        _vlog(f"[debug_wav] wrote {raw_path.name} + {rst_path.name}")

        existing_raw = sorted(_DEBUG_WAV_DIR.glob("raw_*.wav"))
        existing_rst = sorted(_DEBUG_WAV_DIR.glob("resampled_*.wav"))
        while len(existing_raw) > _MAX_DEBUG_WAVS:
            oldest = existing_raw.pop(0)
            oldest.unlink(missing_ok=True)
        while len(existing_rst) > _MAX_DEBUG_WAVS:
            oldest = existing_rst.pop(0)
            oldest.unlink(missing_ok=True)
    except Exception as e:
        _vlog(f"[debug_wav] error: {e}")


from discord.ext.voice_recv import AudioSink, VoiceData, VoiceRecvClient
from discord.ext.voice_recv.rtp import FakePacket, SilencePacket

import logging as _logging
_logging.getLogger("discord.ext.voice_recv.reader").setLevel(_logging.WARNING)

# --- Audio constants ---
SAMPLE_RATE = 48000   # Discord Opus output rate
SAMPLE_WIDTH = 2      # 16-bit PCM
CHANNELS = 2          # Stereo
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz mono
WHISPER_CHANNELS = 1
FRAME_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS // 50  # Bytes per 20ms frame (3840)
SILENCE_DURATION = 1.5   # Seconds of silence to mark end of utterance
MIN_UTTERANCE_DURATION = 0.5  # Minimum seconds to consider an utterance
MAX_UTTERANCE_DURATION = 30.0  # Maximum seconds before forcing transcription

PEAK_THRESHOLD = 12000    # DAVE stale-key garbage peaks above this; real speech rarely exceeds it

WAKE_WORD = "hey bong"
# openWakeWord processes 80ms frames at 16kHz mono = 1280 samples
OWW_FRAME_SAMPLES = 1280
OWW_FRAME_BYTES = OWW_FRAME_SAMPLES * 2

# State: which guilds are listening, and the sink instances
_active_listeners: dict[int, "BongVoiceSink"] = {}
_is_listening: dict[int, bool] = {}
_idle_task: asyncio.Task | None = None


def is_listening(guild_id: int) -> bool:
    return _is_listening.get(guild_id, False)


def _resample_48k_stereo_to_16k_mono(pcm_data: bytes) -> bytes:
    """Downsample 48kHz stereo to 16kHz mono by taking every 3rd sample and averaging L+R."""
    if len(pcm_data) < 4:
        return b''
    samples = np.frombuffer(pcm_data, dtype=np.int16).reshape(-1, 2)
    mono = samples[::3, :].mean(axis=1).astype(np.int16)
    return mono.tobytes()


class BongVoiceSink(AudioSink):
    """Custom AudioSink using openWakeWord for wake word detection and Whisper for transcription."""

    def __init__(self, bot, guild, text_channel, loop):
        super().__init__()
        self.bot = bot
        self.guild = guild
        self.text_channel = text_channel
        self.loop = loop
        # Per-user audio buffers (48kHz stereo, buffered after wake word activation)
        self._buffers: dict[int, bytearray] = defaultdict(bytearray)
        # Per-user last-speech timestamp for silence detection
        self._last_speech: dict[int, float] = {}
        # Per-user "wake word detected" flag
        self._activated: dict[int, bool] = defaultdict(bool)
        # Per-user 16kHz mono resampling buffer for openWakeWord
        self._oww_buffers: dict[int, bytearray] = defaultdict(bytearray)
        self._silence_task: asyncio.Task | None = None
        self._stopped = False

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: VoiceData):
        """Receive PCM audio from Discord, detect wake word, buffer for transcription."""
        if user is None or user.bot:
            return

        user_id = user.id
        pcm_data = data.pcm

        # Empty PCM = DAVE decrypt failure or Opus error — skip entirely
        if not pcm_data:
            return

        # SilencePacket = synthetic silence from SilenceGenerator — skip entirely
        if isinstance(data.packet, SilencePacket):
            return

        # DAVE-corrupted frames (flagged by venv patches) — skip entirely
        ext = getattr(data.packet, 'extension_data', None)
        if isinstance(ext, dict) and ext.get('_voice_recv_needs_dave_inner_decrypt'):
            return

        # Garbage detection: DAVE epoch transitions can produce frames where
        # inner decrypt "succeeded" with stale keys. Opus decodes these into
        # noise with extreme peak values. Real speech peaks rarely exceed PEAK_THRESHOLD.
        try:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            if np.abs(audio_array).max() > PEAK_THRESHOLD:
                return
        except Exception:
            return

        # FakePacket PLC frames produce sustained tones that cause false
        # openWakeWord activations — skip them for wake word detection
        is_fake = isinstance(data.packet, FakePacket)

        if not is_fake:
            resampled = _resample_48k_stereo_to_16k_mono(pcm_data)
            self._oww_buffers[user_id].extend(resampled)

            # Feed 80ms chunks to openWakeWord
            while len(self._oww_buffers[user_id]) >= OWW_FRAME_BYTES:
                chunk = bytes(self._oww_buffers[user_id][:OWW_FRAME_BYTES])
                self._oww_buffers[user_id] = self._oww_buffers[user_id][OWW_FRAME_BYTES:]
                audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                try:
                    oww = _get_oww_model()
                    prediction = oww.predict(audio_chunk)
                    score = float(prediction.get(_OWW_WAKE_WORD, 0))
                    if score >= _OWW_THRESHOLD:
                        if not self._activated.get(user_id, False):
                            _vlog(f"[write] user={user_id} WAKE WORD DETECTED (score={score:.2f})")
                            self._activated[user_id] = True
                            if _oww_model is not None:
                                try:
                                    _oww_model.reset()
                                    if hasattr(_oww_model, 'vad') and hasattr(_oww_model.vad, 'reset_states'):
                                        _oww_model.vad.reset_states()
                                    _vlog("[write] Reset openWakeWord after detection")
                                except Exception as e:
                                    _vlog(f"[write] Failed to reset openWakeWord: {e}")
                except Exception as e:
                    _vlog(f"[write] openWakeWord error: {e}")

        # Only buffer audio for Whisper after wake word has been detected
        if not self._activated.get(user_id, False):
            return

        self._buffers[user_id].extend(pcm_data)
        self._last_speech[user_id] = time.time()

    def _flush_utterance(self, user_id: int) -> bytes | None:
        """Extract buffered audio for a user, or None if too short."""
        buffer = self._buffers.get(user_id)
        if not buffer:
            return None
        pcm_data = bytes(buffer)
        duration = len(buffer) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        _vlog(f"[flush] user={user_id} duration={duration:.2f}s")
        self._buffers[user_id] = bytearray()
        self._activated[user_id] = False
        self._last_speech.pop(user_id, None)
        if duration < MIN_UTTERANCE_DURATION:
            _vlog(f"[flush] Utterance too short ({duration:.2f}s), discarding")
            return None
        return pcm_data

    def cleanup(self):
        self._stopped = True
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        guild_id = self.guild.id
        _active_listeners.pop(guild_id, None)
        _is_listening[guild_id] = False

    async def start_silence_checker(self):
        """Background task that checks for completed utterances based on silence gaps."""
        _vlog("[silence_checker] Started")
        while not self._stopped:
            await asyncio.sleep(0.5)
            now = time.time()
            for user_id in list(self._buffers.keys()):
                if self._stopped:
                    break
                buffer = self._buffers.get(user_id)
                if not buffer:
                    continue
                last_speech = self._last_speech.get(user_id, 0)
                duration = len(buffer) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
                silence_gap = now - last_speech

                if (silence_gap >= SILENCE_DURATION and duration >= MIN_UTTERANCE_DURATION) or duration >= MAX_UTTERANCE_DURATION:
                    _vlog(f"[silence_checker] Flushing user={user_id} duration={duration:.2f}s silence_gap={silence_gap:.1f}s")
                    pcm_data = self._flush_utterance(user_id)
                    if pcm_data is None:
                        continue

                    if not user_data.is_authorized(user_id):
                        _vlog(f"[silence_checker] Ignoring unauthorized user {user_id}")
                        continue

                    _vlog(f"[silence_checker] Transcribing utterance from {user_id} ({duration:.1f}s)")
                    text = await asyncio.to_thread(self._transcribe, pcm_data, user_id)
                    if text:
                        _vlog(f"[silence_checker] Transcribed ({user_id}): '{text}'")
                        await self._handle_transcription(user_id, text)
                    else:
                        _vlog(f"[silence_checker] Empty transcription for {user_id}")

    def _resample_to_whisper_format(self, pcm_data: bytes) -> bytes:
        """Resample 48kHz stereo PCM to 16kHz mono for Whisper."""
        samples = np.frombuffer(pcm_data, dtype=np.int16).reshape(-1, CHANNELS)
        step = SAMPLE_RATE // WHISPER_SAMPLE_RATE
        mono = samples[::step, :].mean(axis=1).astype(np.int16)
        return mono.tobytes()

    def _transcribe(self, pcm_data: bytes, user_id: int = 0) -> str:
        """Transcribe PCM audio using faster-whisper."""
        try:
            resampled = self._resample_to_whisper_format(pcm_data)
            _write_debug_wav(pcm_data, resampled, user_id)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(WHISPER_CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(WHISPER_SAMPLE_RATE)
                wf.writeframes(resampled)
            wav_buffer.seek(0)

            model = _get_whisper_model()
            segments, info = model.transcribe(
                wav_buffer,
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            _vlog(f"[transcribe] language={info.language} prob={info.language_probability:.2f} duration={info.duration:.2f}s")

            filtered = [seg for seg in segments if seg.no_speech_prob < 0.6]

            if info.language_probability < 0.5:
                _vlog(f"[transcribe] low language confidence ({info.language_probability:.2f}), discarding")
                return ""

            text = " ".join(seg.text.strip() for seg in filtered).strip()
            _vlog(f"[transcribe] final text: '{text}'")
            return text
        except Exception as e:
            _vlog(f"[transcribe] error: {e}")
            import traceback
            _vlog(f"[transcribe] traceback: {traceback.format_exc()}")
            return ""

    async def _handle_transcription(self, user_id: int, text: str):
        """Strip wake word from transcription and forward command to LLM."""
        text_lower = text.lower()
        command_text = text
        if WAKE_WORD in text_lower:
            wake_word_idx = text_lower.find(WAKE_WORD)
            command_text = text[wake_word_idx + len(WAKE_WORD):].strip()
            command_text = command_text.lstrip(",.!?;: ")
        else:
            # openWakeWord detected "hey bong" but Whisper may transcribe differently
            for wake_prefix in ("hey bong, ", "hey bong ", "bong, ", "bong "):
                if text_lower.startswith(wake_prefix):
                    command_text = text[len(wake_prefix):]
                    break

        if not command_text:
            command_text = "Hey"

        _vlog(f"[handle] Voice command from {user_id}: '{command_text}'")

        member = self.guild.get_member(user_id)
        if not member:
            try:
                member = await self.guild.fetch_member(user_id)
            except Exception:
                member = None
        username = member.display_name if member else f"User {user_id}"

        channel = self.text_channel
        if channel is None:
            _vlog("[handle] No text channel available, aborting")
            return

        import bong

        try:
            await bong.process_voice_command(
                bot=self.bot,
                guild=self.guild,
                channel=channel,
                user_id=user_id,
                username=username,
                text=command_text,
            )
        except Exception as e:
            _vlog(f"[handle] process_voice_command error: {e}")
            import traceback
            _vlog(f"[handle] traceback: {traceback.format_exc()}")


async def start_listening(bot, guild, text_channel) -> str:
    """Start listening for voice commands in a guild's voice channel."""
    guild_id = guild.id
    if _is_listening.get(guild_id, False):
        return "Already listening for voice commands."

    vc = guild.voice_client
    if not vc or not vc.is_connected():
        return "Not in a voice channel. Join a voice channel first."

    _vlog("Loading voice models...")
    try:
        await asyncio.to_thread(_load_models)
    except Exception as e:
        _vlog(f"Failed to load voice models: {e}")
        return f"Failed to load voice models: {e}"

    if not isinstance(vc, VoiceRecvClient):
        _vlog("Current voice client is not VoiceRecvClient, reconnecting...")
        import bong_tools as _bong_tools
        was_playing = (vc.is_playing() or vc.is_paused()) and _bong_tools.current_track
        saved_track = _bong_tools.current_track if was_playing else None
        channel = vc.channel
        await vc.disconnect()

        try:
            vc = await channel.connect(cls=VoiceRecvClient)
        except Exception as e:
            _vlog(f"Failed to connect with VoiceRecvClient: {e}")
            return f"Failed to start voice receive: {e}"

        if saved_track:
            _vlog(f"Resuming playback: {saved_track}")
            try:
                import bong
                import discord
                after_play = bong._make_after_play_callback(guild, asyncio.get_running_loop())
                source = discord.FFmpegPCMAudio(saved_track, options="-filter:a volume=0.3")
                vc.play(source, after=after_play)
            except Exception as e:
                _vlog(f"Failed to resume playback: {e}")

    sink = BongVoiceSink(bot, guild, text_channel, asyncio.get_running_loop())
    vc.listen(sink)

    _active_listeners[guild_id] = sink
    _is_listening[guild_id] = True

    sink._silence_task = asyncio.create_task(sink.start_silence_checker())

    _vlog(f"Started listening for voice commands in guild {guild_id}")
    return "Listening for voice commands. Say 'hey bong' followed by your command."


async def stop_listening(guild) -> str:
    """Stop listening for voice commands in a guild's voice channel."""
    guild_id = guild.id
    if not _is_listening.get(guild_id, False):
        return "Not currently listening for voice commands."

    sink = _active_listeners.pop(guild_id, None)
    if sink:
        sink._stopped = True
        if sink._silence_task and not sink._silence_task.done():
            sink._silence_task.cancel()

    vc = guild.voice_client
    if vc and isinstance(vc, VoiceRecvClient) and vc.is_listening():
        try:
            vc.stop_listening()
        except Exception as e:
            _vlog(f"Error stopping listener: {e}")

    _is_listening[guild_id] = False

    if not any(_is_listening.values()):
        _vlog("No more active listeners, unloading voice models")
        await asyncio.to_thread(_unload_models)

    _vlog(f"Stopped listening for voice commands in guild {guild_id}")
    return "Stopped listening for voice commands."
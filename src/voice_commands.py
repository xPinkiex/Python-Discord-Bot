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
import traceback
import wave
from collections import defaultdict
from pathlib import Path
from threading import Lock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"


import numpy as np
import debug
import user_data

_LOG_FILE = BONG_DATA / "logs" / "voice_commands.log"
_DEBUG_WAV_DIR = BONG_DATA / "logs" / "voice_debug"
_MAX_DEBUG_WAVS = 10

_model_lock = Lock()

_whisper_model = None
_WHISPER_MODEL_SIZE = "small"
_WHISPER_DOWNLOAD_ROOT = str(BONG_DATA / "whisper_models")

_oww_shared_model = None
_oww_user_states: dict[int, "OwwUserState"] = {}
_OWW_WAKE_WORD = "hey_bong"
_OWW_THRESHOLD = 0.5
_STALE_USER_TIMEOUT = 300

_whisper_semaphore: asyncio.Semaphore | None = None


class _SharedAudioFeatures:
    """AudioFeatures subclass that reuses pre-loaded ONNX sessions instead of creating duplicates.

    Standard AudioFeatures loads ~30MB of ONNX models per instance. By injecting
    shared sessions, per-user OWW state only costs the buffer overhead (~1MB).
    """

    def __init__(self, shared_melspec_model, shared_embedding_model):
        from collections import deque
        from openwakeword.utils import AudioFeatures as _OrigAudioFeatures
        self.melspec_model = shared_melspec_model
        self.embedding_model = shared_embedding_model
        self.onnx_execution_provider = self.melspec_model.get_providers()[0]
        self._orig = _OrigAudioFeatures
        self.raw_data_buffer = deque(maxlen=16000 * 10)
        self.melspectrogram_buffer = np.ones((76, 32))
        self.melspectrogram_max_len = 10 * 97
        self.accumulated_samples = 0
        self.feature_buffer = self._get_embeddings_init(np.zeros(160000).astype(np.int16))
        self.feature_buffer_max_len = 120

    def _get_embeddings_init(self, x):
        spec = self._orig._get_melspectrogram(self, x)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i + 76]
            if window.shape[0] == 76:
                windows.append(window)
        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self.embedding_model.run(None, {'input_1': batch})[0].squeeze()
        return embedding

    def _buffer_raw_data(self, x):
        if len(x) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

    def _streaming_melspectrogram(self, n_samples):
        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer, self._orig._get_melspectrogram(self, list(self.raw_data_buffer)[-n_samples - 160 * 3:]))
        )
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def _streaming_features(self, x):
        self._buffer_raw_data(x)
        self.accumulated_samples += len(x)
        if self.accumulated_samples >= 1280:
            self._streaming_melspectrogram(self.accumulated_samples)
            n_windows = self.accumulated_samples // 1280
            for i in range(n_windows):
                end = len(self.melspectrogram_buffer) - (n_windows - 1 - i) * 76
                start = end - 76
                x_feat = self.melspectrogram_buffer[start:end].astype(np.float32)[None, :, :, None]
                if x_feat.shape[1] == 76:
                    self.feature_buffer = np.vstack((self.feature_buffer,
                                                     self.embedding_model.run(None, {'input_1': x_feat})[0].squeeze()))
            self.accumulated_samples = 0
        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames) \
                if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
            return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
        else:
            return self.feature_buffer[int(-1 * n_feature_frames):, :][None, ].astype(np.float32)

    def __call__(self, x):
        self._streaming_features(x)


class OwwUserState:
    """Per-user openWakeWord state: separate preprocessor + prediction buffers,
    borrowing ONNX inference sessions from the shared model."""

    def __init__(self, user_id: int, shared_oww_model):
        self.user_id = user_id
        self.shared_model = shared_oww_model
        from functools import partial
        from collections import deque
        self.prediction_buffer: dict[str, deque] = defaultdict(partial(deque, maxlen=30))
        self.vad_prediction_buffer: list = []
        self.last_audio_time = time.time()
        self.preprocessor = _SharedAudioFeatures(
            shared_oww_model.preprocessor.melspec_model,
            shared_oww_model.preprocessor.embedding_model,
        )
        self._model_name = list(shared_oww_model.models.keys())[0]
        self._model_inputs = shared_oww_model.model_inputs[self._model_name]
        self._model_input_name = shared_oww_model.model_input_names[self._model_name]
        self._class_mapping = shared_oww_model.class_mapping.get(self._model_name, {})

    def predict(self, audio_chunk: np.ndarray, vad_threshold: float = 0.5) -> dict:
        self.last_audio_time = time.time()
        self.preprocessor(audio_chunk)
        onnx_session = self.shared_model.models[self._model_name]
        if len(audio_chunk) > 1280:
            group_predictions = []
            for i in np.arange(len(audio_chunk) // 1280 - 1, -1, -1):
                group_predictions.extend(
                    onnx_session.run(
                        None,
                        {self._model_input_name: self.preprocessor.get_features(
                            self._model_inputs,
                            start_ndx=-self._model_inputs - i
                        )}
                    )
                )
            prediction = np.array(group_predictions).max(axis=0)[None, ]
        else:
            prediction = onnx_session.run(
                None,
                {self._model_input_name: self.preprocessor.get_features(self._model_inputs)}
            )
        predictions = {}
        if self.shared_model.model_outputs[self._model_name] == 1:
            predictions[self._model_name] = prediction[0][0][0]
            if self._class_mapping:
                for int_label, cls in self._class_mapping.items():
                    predictions[cls] = prediction[0][0][int(int_label)]
        else:
            for int_label, cls in self._class_mapping.items():
                predictions[cls] = prediction[0][0][int(int_label)]
        for cls in predictions.keys():
            if len(self.prediction_buffer[cls]) < 5:
                predictions[cls] = 0.0
            self.prediction_buffer[cls].append(predictions[cls])
        if vad_threshold > 0 and hasattr(self.shared_model, 'vad') and self.shared_model.vad is not None:
            self.shared_model.vad(audio_chunk)
            self.vad_prediction_buffer = list(self.shared_model.vad.prediction_buffer)
            vad_frames = list(self.vad_prediction_buffer)[-7:-4]
            vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
            for mdl in predictions.keys():
                if vad_max_score < vad_threshold:
                    predictions[mdl] = 0.0
        return predictions

    def reset(self):
        from functools import partial
        from collections import deque
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))
        self.vad_prediction_buffer = []
        self.preprocessor = _SharedAudioFeatures(
            self.shared_model.preprocessor.melspec_model,
            self.shared_model.preprocessor.embedding_model,
        )


def _predict_for_user(user_id: int, audio_chunk: np.ndarray) -> dict:
    global _oww_shared_model
    if _oww_shared_model is None:
        return {}
    if user_id not in _oww_user_states:
        _oww_user_states[user_id] = OwwUserState(user_id, _oww_shared_model)
    state = _oww_user_states[user_id]
    vad_threshold = getattr(_oww_shared_model, 'vad_threshold', 0.5)
    return state.predict(audio_chunk, vad_threshold=vad_threshold if vad_threshold > 0 else 0)


def _reset_oww_for_user(user_id: int):
    if user_id in _oww_user_states:
        _oww_user_states[user_id].reset()
        _vlog(f"[oww] Reset state for user {user_id}")


def _destroy_oww_for_user(user_id: int):
    if user_id in _oww_user_states:
        del _oww_user_states[user_id]
        _vlog(f"[oww] Destroyed state for user {user_id}")


def _cleanup_stale_oww_states():
    now = time.time()
    stale = [uid for uid, state in _oww_user_states.items()
             if now - state.last_audio_time > _STALE_USER_TIMEOUT]
    for uid in stale:
        _destroy_oww_for_user(uid)
    if stale:
        _vlog(f"[oww] Cleaned up {len(stale)} stale user state(s)")


def _load_models():
    """Load both Whisper and openWakeWord models into RAM."""
    global _whisper_model, _oww_shared_model
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
        if _oww_shared_model is None:
            import warnings
            warnings.filterwarnings("ignore", message="Specified provider.*not in available")
            from openwakeword.model import Model
            custom_model = BONG_DATA / "wakeword_models" / "hey_bong.onnx"
            if custom_model.exists():
                _vlog(f"Loading custom openWakeWord model ({custom_model})...")
                _oww_shared_model = Model(wakeword_model_paths=[str(custom_model)], vad_threshold=0.5)
            else:
                import openwakeword
                oww_file = openwakeword.__file__
                if oww_file is None:
                    raise RuntimeError("Cannot locate openwakeword package directory")
                model_dir = Path(oww_file).parent / "resources" / "models"
                model_path = model_dir / "hey_jarvis_v0.1.onnx"
                _vlog("Loading fallback openWakeWord model...")
                _oww_shared_model = Model(wakeword_model_paths=[str(model_path)], vad_threshold=0.5)
            _oww_user_states.clear()
            _vlog("openWakeWord shared model loaded")


def _unload_models():
    """Free both Whisper and openWakeWord models from RAM."""
    global _whisper_model, _oww_shared_model
    with _model_lock:
        _whisper_model = None
        _oww_shared_model = None
        _oww_user_states.clear()
    gc.collect()
    _vlog("Unloaded voice models")


def _get_whisper_model():
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            _load_models()
        return _whisper_model



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
SILENCE_DURATION = 0.8   # Seconds of silence to mark end of utterance
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


def is_listening(guild_id: int) -> bool:
    return _is_listening.get(guild_id, False)



class BongVoiceSink(AudioSink):
    """Custom AudioSink using openWakeWord for wake word detection and Whisper for transcription."""

    def __init__(self, bot, guild, text_channel, loop):
        super().__init__()
        self.bot = bot
        self.guild = guild
        self.text_channel = text_channel
        self.loop = loop
        # Per-user audio buffers (48kHz stereo, buffered after wake word activation)
        self._buffers: dict[int, bytearray] = {}
        # Per-user last-speech timestamp for silence detection
        self._last_speech: dict[int, float] = {}
        # Per-user "wake word detected" flag
        self._activated: dict[int, bool] = {}
        # Per-user 16kHz mono resampling buffer for openWakeWord
        self._oww_buffers: dict[int, bytearray] = {}
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
            if user_data.has_permission(user_id, "vc_commands"):
                resampled = self._resample_to_whisper_format(pcm_data)
                self._oww_buffers.setdefault(user_id, bytearray()).extend(resampled)

                while len(self._oww_buffers[user_id]) >= OWW_FRAME_BYTES:
                    chunk = bytes(self._oww_buffers[user_id][:OWW_FRAME_BYTES])
                    self._oww_buffers[user_id] = self._oww_buffers[user_id][OWW_FRAME_BYTES:]
                    audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                    try:
                        prediction = _predict_for_user(user_id, audio_chunk)
                        score = float(prediction.get(_OWW_WAKE_WORD, 0))
                        if score >= _OWW_THRESHOLD:
                            if not self._activated.get(user_id, False):
                                _vlog(f"[write] user={user_id} WAKE WORD DETECTED (score={score:.2f})")
                                self._activated[user_id] = True
                                _reset_oww_for_user(user_id)
                    except Exception as e:
                        _vlog(f"[write] openWakeWord error: {e}")
            else:
                self._oww_buffers.pop(user_id, None)

        # Only buffer audio for Whisper after wake word has been detected
        if not self._activated.get(user_id, False):
            return

        self._buffers.setdefault(user_id, bytearray()).extend(pcm_data)
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
        _reset_oww_for_user(user_id)
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
        for uid in list(self._oww_buffers.keys()):
            _destroy_oww_for_user(uid)
        self._oww_buffers.clear()

    async def start_silence_checker(self):
        """Background task that checks for completed utterances based on silence gaps."""
        _vlog("[silence_checker] Started")
        while not self._stopped:
            try:
                await asyncio.sleep(0.15)
                now = time.time()
                _cleanup_stale_oww_states()
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

                        if not user_data.has_permission(user_id, "vc_commands"):
                            _vlog(f"[silence_checker] Ignoring user {user_id} without vc_commands permission")
                            continue

                        _vlog(f"[silence_checker] Queuing transcription for {user_id} ({duration:.1f}s)")
                        if _whisper_semaphore:
                            async with _whisper_semaphore:
                                _vlog(f"[silence_checker] Transcribing utterance from {user_id} ({duration:.1f}s)")
                                text = await asyncio.to_thread(self._transcribe, pcm_data, user_id)
                        else:
                            text = await asyncio.to_thread(self._transcribe, pcm_data, user_id)
                        if text:
                            _vlog(f"[silence_checker] Transcribed ({user_id}): '{text}'")
                            await self._handle_transcription(user_id, text)
                        else:
                            _vlog(f"[silence_checker] Empty transcription for {user_id}")
            except Exception:
                debug.error("[silence_checker]", traceback.format_exc())

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
            # openWakeWord detected wake word but Whisper may transcribe differently
            ww_parts = WAKE_WORD.split()
            prefixes = []
            for i in range(len(ww_parts)):
                short = " ".join(ww_parts[i:])
                prefixes.append(f"{short}, ")
                prefixes.append(f"{short} ")
            for wake_prefix in prefixes:
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
            _vlog(f"[handle] traceback: {traceback.format_exc()}")


async def start_listening(bot, guild, text_channel) -> str:
    """Start listening for voice commands in a guild's voice channel."""
    global _whisper_semaphore
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

    if _whisper_semaphore is None:
        _whisper_semaphore = asyncio.Semaphore(1)

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
    global _whisper_semaphore
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
        _whisper_semaphore = None

    _vlog(f"Stopped listening for voice commands in guild {guild_id}")
    return "Stopped listening for voice commands."
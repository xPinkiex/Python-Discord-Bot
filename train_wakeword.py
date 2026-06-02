#!/usr/bin/env python3
"""Train a custom 'hey bong' wake word model for openWakeWord using synthetic data.

Usage:
    python train_wakeword.py [--setup] [--generate] [--augment] [--train] [--all]

Steps:
    1. --setup:      Install dependencies, clone openWakeWord repo, download TTS model + training data
    2. --generate:   Generate synthetic positive and adversarial negative clips (via Piper TTS)
    3. --augment:    Augment clips with noise, RIR, and compute features
    4. --train:      Train the model and export to ONNX
    5. --all:        Run all steps

The trained model is saved to wakeword_models/hey_bong.onnx and can be loaded
by voice_commands.py by updating _OWW_WAKE_WORD and related config.

Note: Training uses the openWakeWord GitHub repo (which includes train.py,
generate_adversarial_texts, augment_clips, etc.) rather than the v0.4.0 pip
package that lacks these training utilities.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAINING_DIR = SCRIPT_DIR / "wakeword_training"
OWW_REPO_DIR = TRAINING_DIR / "openwakeword"
PIPER_DIR = TRAINING_DIR / "piper-sample-generator"
PIPER_MODEL = PIPER_DIR / "en_US-libritts_r-medium.pt"
MODEL_NAME = "hey_bong"
OUTPUT_DIR = TRAINING_DIR / "wakeword_model"
WAKEWORD_MODELS_DIR = SCRIPT_DIR / "wakeword_models"
CONFIG_PATH = TRAINING_DIR / "hey_bong_config.yml"

RIR_DIR = TRAINING_DIR / "mit_rirs"
BACKGROUND_DIR = TRAINING_DIR / "background_clips"
AUDIOSET_DIR = TRAINING_DIR / "audioset_raw"
ACAV_FEATURES = TRAINING_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_FEATURES = TRAINING_DIR / "validation_set_features.npy"

TRAIN_SCRIPT = OWW_REPO_DIR / "openwakeword" / "train.py"

N_SAMPLES = 12000
N_SAMPLES_VAL = 1200
STEPS = 30000

RIR_SCRIPT = TRAINING_DIR / "_download_rirs.py"
AUDIOSET_SCRIPT = TRAINING_DIR / "_convert_audioset.py"


def run(cmd, **kwargs):
    if isinstance(cmd, str):
        print(f"\n>>> {cmd}")
        result = subprocess.run(cmd, shell=True, **kwargs)
    else:
        print(f"\n>>> {' '.join(str(c) for c in cmd)}")
        result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def setup():
    print("=" * 60)
    print("STEP 1: Setting up training environment")
    print("=" * 60)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    install_packages = [
        "speechbrain",
        "audiomentations",
        "torch-audiomentations",
        "acoustics",
        "torchinfo",
        "torchmetrics",
        "mutagen",
        "datasets",
        "deep-phonemizer",
        "scipy",
        "pyyaml",
        "torch",
    ]

    print("\nInstalling Python packages...")
    run([sys.executable, "-m", "pip", "install"] + install_packages)

    if not OWW_REPO_DIR.exists():
        print("\nCloning openWakeWord repo (for train.py)...")
        run(["git", "clone", "https://github.com/dscripka/openwakeword.git", str(OWW_REPO_DIR)])
    else:
        print("\nopenWakeWord repo already cloned, skipping.")

    if not PIPER_DIR.exists():
        print("\nCloning piper-sample-generator...")
        run(["git", "clone", "https://github.com/rhasspy/piper-sample-generator", str(PIPER_DIR)])
    else:
        print("\npiper-sample-generator already cloned, skipping.")

    if not PIPER_MODEL.exists():
        print("\nDownloading Piper TTS model (~130MB)...")
        run([
            "wget", "-O", str(PIPER_MODEL),
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
        ])
    else:
        print("\nPiper TTS model already downloaded, skipping.")

    openwakeword_pkg = Path(
        subprocess.check_output(
            [sys.executable, "-c", "import openwakeword; print(openwakeword.__file__)"],
            universal_newlines=True,
        ).strip()
    ).parent

    models_dir = openwakeword_pkg / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    required_models = {
        "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    }
    for fname, url in required_models.items():
        dest = models_dir / fname
        if not dest.exists():
            print(f"\nDownloading {fname}...")
            run(["wget", "-O", str(dest), url])
        else:
            print(f"\n{fname} already exists, skipping.")

    if not ACAV_FEATURES.exists():
        print("\nDownloading ACAV100M features (~4.5GB) - this will take a while...")
        run([
            "wget", "-O", str(ACAV_FEATURES),
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
            "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        ])
    else:
        print("ACAV features already exist, skipping.")

    if not VALIDATION_FEATURES.exists():
        print("\nDownloading validation features (~250MB)...")
        run([
            "wget", "-O", str(VALIDATION_FEATURES),
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
            "validation_set_features.npy",
        ])
    else:
        print("Validation features already exist, skipping.")

    print("\nDownloading room impulse responses...")
    if not RIR_DIR.exists() or not any(RIR_DIR.iterdir()):
        RIR_DIR.mkdir(parents=True, exist_ok=True)
        rir_script_content = '''import scipy.io, os
from datasets import load_dataset
from tqdm import tqdm
rir_dir = r"''' + str(RIR_DIR) + '''"
rir = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
for row in tqdm(rir):
    name = row["audio"]["path"].split("/")[-1]
    scipy.io.wavfile.write(os.path.join(rir_dir, name), 16000, (row["audio"]["array"]*32767).astype(np.int16))
'''
        RIR_SCRIPT.write_text(rir_script_content)
        run([sys.executable, str(RIR_SCRIPT)])
    else:
        print("RIR directory already exists and not empty, skipping.")

    print("\nDownloading background audio (AudioSet subset)...")
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    if not AUDIOSET_DIR.exists():
        AUDIOSET_DIR.mkdir(parents=True, exist_ok=True)
        fname = "bal_train09.tar"
        out_tar = AUDIOSET_DIR / fname
        run(["wget", "-O", str(out_tar), f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{fname}"])
        run(["tar", "-xvf", str(out_tar)], cwd=str(AUDIOSET_DIR))

        audioset_script_content = '''import scipy.io, os
from pathlib import Path
from datasets import Dataset, Audio
from tqdm import tqdm
audioset_dir = r"''' + str(AUDIOSET_DIR) + '''"
background_dir = r"''' + str(BACKGROUND_DIR) + '''"
files = [str(i) for i in Path(audioset_dir + "/audio").glob("**/*.flac")]
ds = Dataset.from_dict({"audio": files}).cast_column("audio", Audio(sampling_rate=16000))
for row in tqdm(ds):
    name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
    scipy.io.wavfile.write(os.path.join(background_dir, name), 16000, (row["audio"]["array"]*32767).astype(np.int16))
'''
        AUDIOSET_SCRIPT.write_text(audioset_script_content)
        run([sys.executable, str(AUDIOSET_SCRIPT)])
    else:
        print("AudioSet directory already exists, skipping.")

    write_config()
    print("\nSetup complete!")


def write_config():
    import yaml

    config = {
        "model_name": MODEL_NAME,
        "target_phrase": ["hey bong"],
        "custom_negative_phrases": [],
        "n_samples": N_SAMPLES,
        "n_samples_val": N_SAMPLES_VAL,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(PIPER_DIR),
        "output_dir": str(OUTPUT_DIR),
        "rir_paths": [str(RIR_DIR)],
        "background_paths": [str(BACKGROUND_DIR)],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(VALIDATION_FEATURES),
        "augmentation_rounds": 1,
        "feature_data_files": {"ACAV100M_sample": str(ACAV_FEATURES)},
        "batch_n_per_class": {
            "ACAV100M_sample": 1024,
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": STEPS,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nConfig written to {CONFIG_PATH}")


def generate_clips():
    print("=" * 60)
    print("STEP 2: Generating synthetic clips")
    print("=" * 60)

    if not CONFIG_PATH.exists():
        write_config()

    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: Training script not found at {TRAIN_SCRIPT}")
        print("Did you run --setup first?")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OWW_REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["MULTIPROCESSING_START_METHOD"] = "fork"
    run([sys.executable, str(TRAIN_SCRIPT), "--training_config", str(CONFIG_PATH), "--generate_clips"], env=env)


def augment_clips():
    print("=" * 60)
    print("STEP 3: Augmenting clips and computing features")
    print("=" * 60)

    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: Training script not found at {TRAIN_SCRIPT}")
        print("Did you run --setup first?")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OWW_REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["MULTIPROCESSING_START_METHOD"] = "fork"
    run([sys.executable, str(TRAIN_SCRIPT), "--training_config", str(CONFIG_PATH), "--augment_clips"], env=env)


def train_model():
    print("=" * 60)
    print("STEP 4: Training model")
    print("=" * 60)

    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: Training script not found at {TRAIN_SCRIPT}")
        print("Did you run --setup first?")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OWW_REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["MULTIPROCESSING_START_METHOD"] = "fork"
    run([sys.executable, str(TRAIN_SCRIPT), "--training_config", str(CONFIG_PATH), "--train_model"], env=env)


def copy_model():
    onnx_src = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
    if not onnx_src.exists():
        print(f"\nERROR: Trained model not found at {onnx_src}")
        print("Did you run the training step?")
        return

    WAKEWORD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_dest = WAKEWORD_MODELS_DIR / f"{MODEL_NAME}.onnx"

    shutil.copy2(onnx_src, onnx_dest)
    print(f"\nModel copied to {onnx_dest}")
    print()
    print("To use this model, update voice_commands.py:")
    print(f"  _OWW_WAKE_WORD = '{MODEL_NAME}'")
    print(f"  And change the model loading to use the custom model path:")
    print(f"  _oww_model = Model(wakeword_model_paths=['wakeword_models/{MODEL_NAME}.onnx'], vad_threshold=0.5)")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom 'hey bong' wake word model for openWakeWord"
    )
    parser.add_argument("--setup", action="store_true", help="Install dependencies and download training data")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic positive/negative clips")
    parser.add_argument("--augment", action="store_true", help="Augment clips and compute features")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if not any([args.setup, args.generate, args.augment, args.train, args.all]):
        parser.print_help()
        print("\nSpecify at least one step to run.")
        sys.exit(1)

    if args.all:
        setup()
        generate_clips()
        augment_clips()
        train_model()
        copy_model()
    else:
        if args.setup:
            setup()
        if args.generate:
            generate_clips()
        if args.augment:
            augment_clips()
        if args.train:
            train_model()
            copy_model()


if __name__ == "__main__":
    main()
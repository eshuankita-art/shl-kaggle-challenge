"""
Baseline for the SHL audio scoring task.

Each clip is a spoken recording. The goal is to predict a score from 0 to 5.
This script describes every clip with wav2vec2-base, then fits a Ridge model.

Run it in a Kaggle notebook with the GPU on and Internet on, so the
pretrained weights can download. The audio is read from
/kaggle/input/shl-hiring-assessment-2026/.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor

# wav2vec2-base reads 16 kHz audio and returns a 768-number description.
MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
# A full minute is too long for one transformer pass, so each clip is cut
# into 10-second pieces. Those pieces are averaged back into one vector.
CHUNK_SECONDS = 10
# Ridge shrinks the linear weights. 1.0 is the usual starting penalty.
RIDGE_ALPHA = 1.0
N_FOLDS = 5
RANDOM_STATE = 42
# Scores outside the legal range are pulled back onto [0, 5].
SCORE_MIN = 0.0
SCORE_MAX = 5.0


def find_data_root():
    """Locate the folder that holds train.csv, test.csv, and the wav folders."""
    candidates = [
        Path("/kaggle/input/shl-hiring-assessment-2026"),
        Path("/kaggle/input/shl-hiring-assessment-2026/Dataset_Final"),
        Path("shl-hiring-assessment-2026/Dataset_Final"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
        nested = candidate / "Dataset_Final"
        if (nested / "train.csv").exists():
            return nested

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_csv in kaggle_input.rglob("train.csv"):
            return train_csv.parent

    raise FileNotFoundError(
        "Could not find train.csv under /kaggle/input/shl-hiring-assessment-2026/"
    )


def read_table(path):
    """Read a csv into a list of rows, keeping the file's row order."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_wav(path):
    """Read a mono wav and return samples scaled to the range -1 to 1."""
    import wave

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path.name} is not 16-bit audio")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio / 32768.0

    if sample_rate != SAMPLE_RATE:
        # wav2vec2 expects 16 kHz. Stretch or squeeze the clip onto that rate.
        duration = len(audio) / sample_rate
        target_length = int(round(duration * SAMPLE_RATE))
        source_times = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        target_times = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        audio = np.interp(target_times, source_times, audio).astype(np.float32)

    return audio


def find_wav(root, filename):
    """Return the wav path. Training and test clips share some names."""
    for folder in ("train", "test"):
        path = root / folder / filename
        if path.exists():
            return path
    return None


def embed_audio(audio, model, processor, device):
    """Turn one clip into one 768-number vector by averaging wav2vec2 over time."""
    chunk_size = CHUNK_SECONDS * SAMPLE_RATE
    if len(audio) <= chunk_size:
        chunks = [audio]
    else:
        chunks = [
            audio[start : start + chunk_size]
            for start in range(0, len(audio), chunk_size)
        ]
        # Drop a tiny leftover. A slice under one second is not a useful example.
        chunks = [chunk for chunk in chunks if len(chunk) >= SAMPLE_RATE]

    if not chunks:
        return None

    vectors = []
    weights = []
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.inference_mode():
            hidden = model(input_values).last_hidden_state
        # Average the time steps inside this piece, then keep the piece's length
        # so a short tail does not count as much as a full 10-second piece.
        vectors.append(hidden.mean(dim=1).squeeze(0).cpu().numpy())
        weights.append(len(chunk))

    return np.average(np.stack(vectors), axis=0, weights=weights).astype(np.float32)


def embed_files(filenames, root, model, processor, device):
    """Embed each filename once. Missing files are recorded and skipped."""
    embeddings = {}
    missing = []
    for index, filename in enumerate(filenames, start=1):
        if filename in embeddings:
            continue
        path = find_wav(root, filename)
        if path is None:
            missing.append(filename)
            continue
        audio = load_wav(path)
        vector = embed_audio(audio, model, processor, device)
        if vector is None:
            missing.append(filename)
            continue
        embeddings[filename] = vector
        if index % 25 == 0 or index == len(filenames):
            print(f"Embedded {index} of {len(filenames)} clips")
    return embeddings, missing


def make_model():
    """Scale each of the 768 numbers, then fit a straight line with a penalty."""
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def main():
    root = find_data_root()
    print(f"Reading data from {root}")

    train_rows = read_table(root / "train.csv")
    test_rows = read_table(root / "test.csv")
    sample_rows = read_table(root / "sample_submission.csv")

    # The Data tab lists 769 train wavs, 216 test wavs, and a sample file.
    # The sample file is the submission template, and it is shorter than test.csv.
    print(f"train.csv rows: {len(train_rows)}")
    print(f"test.csv rows: {len(test_rows)}")
    print(f"sample_submission.csv rows: {len(sample_rows)}")
    if len(sample_rows) != len(test_rows):
        print(
            "Row mismatch: submission.csv follows sample_submission.csv, "
            "not test.csv."
        )

    train_labels = np.array([float(row["label"]) for row in train_rows], dtype=np.float64)
    fallback_score = float(train_labels.mean())
    print(f"Average training score, used when a clip is missing: {fallback_score:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Every training clip is needed for the fit. Submission clips that are not
    # already in training are added only when the wav is actually on disk.
    needed = []
    seen = set()
    for row in train_rows + sample_rows:
        name = row["filename"]
        if name not in seen:
            needed.append(name)
            seen.add(name)

    embeddings, missing = embed_files(needed, root, model, processor, device)
    print(f"Clips with an embedding: {len(embeddings)}")
    print(f"Clips with no wav file: {len(missing)}")

    train_names = [row["filename"] for row in train_rows]
    missing_train = [name for name in train_names if name not in embeddings]
    if missing_train:
        raise RuntimeError(f"Training clips with no wav: {missing_train[:5]}")

    features = np.stack([embeddings[name] for name in train_names])

    # Five-fold CV shuffles the clips so the printed score is not an artifact
    # of the order in train.csv. Lower RMSE is better. RMSE is in score points.
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = cross_val_score(
        make_model(),
        features,
        train_labels,
        cv=folds,
        scoring="neg_root_mean_squared_error",
    )
    fold_rmse = -fold_scores
    print("5-fold CV RMSE by fold:", ", ".join(f"{value:.4f}" for value in fold_rmse))
    print(f"5-fold CV RMSE: {fold_rmse.mean():.4f}")

    # The folds were only for the score. The submitted model sees every training clip.
    fitted = make_model()
    fitted.fit(features, train_labels)

    predictions = []
    used_audio = 0
    used_fallback = 0
    for row in sample_rows:
        vector = embeddings.get(row["filename"])
        if vector is None:
            score = fallback_score
            used_fallback += 1
        else:
            score = float(fitted.predict(vector.reshape(1, -1))[0])
            used_audio += 1
        predictions.append(float(np.clip(score, SCORE_MIN, SCORE_MAX)))

    print(f"Submission rows from audio: {used_audio}")
    print(f"Submission rows set to the training average: {used_fallback}")

    if Path("/kaggle/working").exists():
        output_path = Path("/kaggle/working/submission.csv")
    else:
        output_path = Path("submission.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
        writer.writeheader()
        for row, score in zip(sample_rows, predictions):
            writer.writerow({"filename": row["filename"], "label": f"{score:.6f}"})

    # Confirm the saved file matches the sample template before you submit it.
    saved_rows = read_table(output_path)
    same_names = [row["filename"] for row in saved_rows] == [
        row["filename"] for row in sample_rows
    ]
    blank_scores = [row["label"] for row in saved_rows if row["label"] == ""]
    print(f"Wrote {output_path}")
    print(f"Columns: filename, label")
    print(f"Rows match the sample file: {len(saved_rows) == len(sample_rows) and same_names}")
    print(f"Blank scores: {len(blank_scores)}")


if __name__ == "__main__":
    main()

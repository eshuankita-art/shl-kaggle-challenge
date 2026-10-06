"""
Score spoken answers for grammar.

Each clip is about 45 to 60 seconds. The label is a continuous score from 0 to 5.
wav2vec2-base describes the sound. Whisper-base writes the words. Four text
counts are added, and a Ridge model with penalty 100 predicts the score.

Training clips are read only from train/. Test clips are read only from test/.
Many filenames appear in both folders, but those files are different recordings.

Run this in a Kaggle notebook with the GPU on and Internet on, so the
pretrained weights can download. The data folder is the competition input
that contains Dataset_Final.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor

# wav2vec2-base returns a 768-number description of 16 kHz audio.
AUDIO_MODEL = "facebook/wav2vec2-base"
# Whisper-base writes the spoken words down as text.
TEXT_MODEL = "openai/whisper-base"
SAMPLE_RATE = 16000
# A full minute is too long for one pass, so the audio is cut into pieces.
EMBED_CHUNK_SECONDS = 10
TEXT_CHUNK_SECONDS = 30
# This penalty is the one used for the public RMSE of 0.5696. Leave it fixed.
RIDGE_ALPHA = 100.0
N_FOLDS = 5
RANDOM_STATE = 42
SCORE_MIN = 0.0
SCORE_MAX = 5.0


def find_data_root():
    """Find the folder that holds train.csv. Search Kaggle first, then a local copy."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for train_csv in kaggle_input.rglob("train.csv"):
            return train_csv.parent

    local = Path("shl-hiring-assessment-2026") / "Dataset_Final"
    if (local / "train.csv").is_file():
        return local

    raise FileNotFoundError(
        "Could not find train.csv under /kaggle/input or "
        "shl-hiring-assessment-2026/Dataset_Final"
    )


def cache_directory():
    """Keep caches on Kaggle. Off Kaggle, do not write cache files."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    return None


def submission_path():
    """Write the scored file where Kaggle can submit it, or in the current folder."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "submission.csv"
    return Path("submission.csv")


def read_rows(path):
    """Read a csv into a list of rows, keeping the file's row order."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clip_path(root, split, filename):
    """Point at train/ or test/ only. A shared name is not the same recording."""
    path = root / split / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} audio: {filename}")
    return path


def load_wav(path):
    """Read a 16 kHz wav and scale 16-bit samples to the range -1 to 1."""
    import wave

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio / 32768.0


def rmse(y_true, y_pred):
    """Root mean squared error. Lower is better. The unit is score points."""
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(error ** 2)))


def pearson(y_true, y_pred):
    """Correlation of predictions with true scores. Higher is better."""
    left = np.asarray(y_true, dtype=np.float64)
    right = np.asarray(y_pred, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    return float((left * right).sum() / np.sqrt((left ** 2).sum() * (right ** 2).sum()))


def embed_audio(audio, model, processor, device):
    """Turn one clip into one 768-number vector."""
    # Cut the clip into 10-second pieces. A leftover under one second is left out.
    chunk_size = EMBED_CHUNK_SECONDS * SAMPLE_RATE
    chunks = [audio[start:start + chunk_size] for start in range(0, len(audio), chunk_size)]
    chunks = [chunk for chunk in chunks if len(chunk) >= SAMPLE_RATE]

    vectors = []
    weights = []
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(inputs.input_values.to(device)).last_hidden_state
        # Average the time steps inside this piece.
        vectors.append(hidden.mean(dim=1).squeeze(0).cpu().numpy())
        # A short piece should count less than a full 10-second piece.
        weights.append(len(chunk))

    return np.average(np.stack(vectors), axis=0, weights=weights).astype(np.float32)


def transcribe(audio, model, processor, device):
    """Write down the words. Whisper reads about 30 seconds at a time."""
    chunk_size = TEXT_CHUNK_SECONDS * SAMPLE_RATE
    pieces = []
    forced = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) < SAMPLE_RATE:
            continue
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            token_ids = model.generate(
                inputs.input_features.to(device),
                forced_decoder_ids=forced,
                max_new_tokens=224,
            )
        pieces.append(processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip())
    return " ".join(piece for piece in pieces if piece)


def text_counts(transcript):
    """Four plain counts from the transcript."""
    # "like" is counted every time, including when it is a real word.
    words = re.findall(r"[A-Za-z']+", transcript.lower())
    sentences = [part for part in re.split(r"[.!?]+", transcript) if part.strip()]
    word_count = float(len(words))
    sentence_count = float(len(sentences) if sentences else (1 if words else 0))
    words_per_sentence = word_count / sentence_count if sentence_count else 0.0
    fillers = {"um", "uh", "uhm", "er", "ah", "hmm", "like"}
    filler_count = float(sum(word in fillers for word in words))
    filler_count += float(len(re.findall(r"\byou know\b", transcript.lower())))
    return np.array(
        [word_count, sentence_count, words_per_sentence, filler_count],
        dtype=np.float32,
    )


def load_embeddings(path):
    """Read a saved filename-to-vector cache. A missing file means nothing is cached."""
    if path is None or not path.is_file():
        return {}
    stored = np.load(path, allow_pickle=True)
    return {str(name): vector for name, vector in zip(stored["names"], stored["vectors"])}


def save_embeddings(path, names, embeddings):
    """Save one split. Train and test must not share a file, or names would collide."""
    np.savez_compressed(
        path,
        names=np.array(names),
        vectors=np.stack([embeddings[name] for name in names]),
    )


def load_transcripts(path):
    """Read a saved filename-to-text cache."""
    if path is None or not path.is_file():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): text for name, text in stored.items()}


def save_transcripts(path, names, transcripts):
    """Save transcripts for one split only."""
    payload = {name: transcripts[name] for name in names}
    path.write_text(json.dumps(payload), encoding="utf-8")


def describe_split(names, root, split, embeddings, transcripts, emb_cache, text_cache, device):
    """Describe each clip in this split, reusing a cache when every name is already stored."""
    need_audio = any(name not in embeddings for name in names)
    if need_audio:
        print("Loading", AUDIO_MODEL, "for", split)
        audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
        audio_model = AutoModel.from_pretrained(AUDIO_MODEL).to(device)
        audio_model.eval()
        for index, name in enumerate(names, start=1):
            if name not in embeddings:
                audio = load_wav(clip_path(root, split, name))
                embeddings[name] = embed_audio(audio, audio_model, audio_processor, device)
            if index % 25 == 0 or index == len(names):
                print(f"{split} audio descriptions {index} of {len(names)}")
        if emb_cache is not None:
            save_embeddings(emb_cache, names, embeddings)
        del audio_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"Reusing cached {split} audio descriptions")

    need_text = any(name not in transcripts for name in names)
    if need_text:
        print("Loading", TEXT_MODEL, "for", split)
        text_processor = AutoProcessor.from_pretrained(TEXT_MODEL)
        text_model = AutoModelForSpeechSeq2Seq.from_pretrained(TEXT_MODEL).to(device)
        text_model.eval()
        for index, name in enumerate(names, start=1):
            if name not in transcripts:
                audio = load_wav(clip_path(root, split, name))
                transcripts[name] = transcribe(audio, text_model, text_processor, device)
            if index % 25 == 0 or index == len(names):
                print(f"{split} transcripts {index} of {len(names)}")
        if text_cache is not None:
            save_transcripts(text_cache, names, transcripts)
        del text_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"Reusing cached {split} transcripts")


def combine(embedding, transcript):
    """The first 768 numbers describe the sound. The last 4 describe the words."""
    return np.concatenate([embedding, text_counts(transcript)])


def make_model():
    """Scale every column, then fit a straight line with penalty 100."""
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def main():
    root = find_data_root()
    print("Reading data from", root)

    # Train rows and test rows stay in csv order. The submission follows test.csv.
    train_rows = read_rows(root / "train.csv")
    test_rows = read_rows(root / "test.csv")
    y_train = np.array([float(row["label"]) for row in train_rows], dtype=np.float64)
    train_names = [row["filename"] for row in train_rows]
    test_names = [row["filename"] for row in test_rows]
    print("train.csv rows:", len(train_rows))
    print("test.csv rows:", len(test_rows))

    # Separate files so a name that exists in both folders cannot overwrite itself.
    cache_root = cache_directory()
    if cache_root is None:
        train_emb_cache = None
        train_text_cache = None
        test_emb_cache = None
        test_text_cache = None
        print("No /kaggle/working directory, so caches are skipped")
    else:
        train_emb_cache = cache_root / "wav2vec_embeddings.npz"
        train_text_cache = cache_root / "whisper_transcripts.json"
        test_emb_cache = cache_root / "test_wav2vec_embeddings.npz"
        test_text_cache = cache_root / "test_whisper_transcripts.json"

    # Load each cache on its own. Never describe a test clip from the training file.
    train_embeddings = load_embeddings(train_emb_cache)
    train_transcripts = load_transcripts(train_text_cache)
    test_embeddings = load_embeddings(test_emb_cache)
    test_transcripts = load_transcripts(test_text_cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    describe_split(
        train_names,
        root,
        "train",
        train_embeddings,
        train_transcripts,
        train_emb_cache,
        train_text_cache,
        device,
    )
    describe_split(
        test_names,
        root,
        "test",
        test_embeddings,
        test_transcripts,
        test_emb_cache,
        test_text_cache,
        device,
    )

    # This count should be large. It shows the test folder was actually used.
    shared = [name for name in test_names if name in train_embeddings]
    changed = sum(
        not np.allclose(train_embeddings[name], test_embeddings[name]) for name in shared
    )
    print("Shared names whose test description differs from training:", changed, "of", len(shared))

    x_train = np.stack([
        combine(train_embeddings[name], train_transcripts[name]) for name in train_names
    ])
    x_test = np.stack([
        combine(test_embeddings[name], test_transcripts[name]) for name in test_names
    ])

    # Fit on every training row, then score those same rows. This number is optimistic.
    model = make_model()
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    print("Training RMSE:", round(rmse(y_train, train_pred), 4))
    print("Training Pearson:", round(pearson(y_train, train_pred), 4))

    # Five folds, shuffled, so the fair score is not an artifact of csv order.
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(make_model(), x_train, y_train, cv=folds)
    print("5-fold CV RMSE:", round(rmse(y_train, cv_pred), 4))
    print("5-fold CV Pearson:", round(pearson(y_train, cv_pred), 4))

    # The folds were only for the score. The submitted scores use the full-data fit.
    test_pred = np.clip(model.predict(x_test), SCORE_MIN, SCORE_MAX)
    output_path = submission_path()
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
        writer.writeheader()
        for row, score in zip(test_rows, test_pred):
            writer.writerow({"filename": row["filename"], "label": f"{float(score):.6f}"})

    saved = read_rows(output_path)
    saved_names = [row["filename"] for row in saved]
    print("Wrote", output_path)
    print("Test rows read from test/:", len(test_names))
    print("Rows match test.csv:", saved_names == test_names and len(saved) == len(test_rows))


if __name__ == "__main__":
    main()

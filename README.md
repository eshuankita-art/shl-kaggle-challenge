# SHL grammar scoring

Spoken answers last about 45 to 60 seconds. The task is to score grammar on a continuous scale from 0 to 5. The training set has 769 clips. The test set has 216 clips.

## Metric

The leaderboard metric is RMSE. Lower is better. Pearson correlation is the second metric. Higher is better.

## Approach

Each clip is turned into a wav2vec2-base embedding by cutting the audio into 10-second pieces, mean-pooling each piece, and averaging the pieces by length. Whisper-base writes a transcript in 30-second pieces. Four counts come from that transcript: words, sentences, words per sentence, and filler words. The fillers are um, uh, uhm, er, ah, hmm, like, and the phrase "you know". The word "like" is counted every time, including when it is a real word. Those 768 audio numbers and 4 counts go into a standardized Ridge regression with penalty 100.

## Data bug

212 filenames appear in both `train/` and `test/`, but the audio files are different recordings. Test clips must be read from `test/`. Reading the training copy for a shared name scores the wrong recording.

## Results

These numbers are from the Kaggle runs that used penalty 100 and read every test clip from `test/`:

- Training RMSE: 0.5093
- 5-fold CV RMSE: 0.8076
- 5-fold CV Pearson: 0.821
- Public leaderboard RMSE: 0.5696

## How to run

Open a Kaggle notebook with GPU and Internet turned on. The data path is the competition input that contains `Dataset_Final`. Paste `baseline.py` into a cell, or run the file. The script writes `submission.csv` under `/kaggle/working`, and it keeps separate cache files for the training and test descriptions.

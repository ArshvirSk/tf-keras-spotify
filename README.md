# tf-keras-spotify — TensorFlow/Keras success-model notebook (G5)

Executed Jupyter notebook ([spotify_keras_success_model.ipynb](spotify_keras_success_model.ipynb), outputs included): a Keras MLP predicting a track's `log10(Spotify Streams)` from its 19-dim cross-platform footprint (YouTube / TikTok / Apple Music / AirPlay / playlists / Shazam / Pandora / Soundcloud / Track Score), on my own 2024 Spotify dashboard dataset (4,600 tracks).

## Results (TensorFlow 2.21, CPU)

| Metric | Value |
|---|---|
| Test MAE | **0.197 log10 units** ≈ ×1.57 typical ratio error |
| Best val MAE | 0.194 |
| Spearman rank corr (test) | **0.927** |
| Epochs (EarlyStopping) | 215 |

## What it demonstrates (the TF/Keras workflow, deliberately)

- `tf.keras.layers.Normalization().adapt()` **inside** the model — preprocessing travels with the weights
- `tf.data` pipelines (`from_tensor_slices`, `shuffle`, `batch`, `prefetch(AUTOTUNE)`)
- `EarlyStopping` (restore-best-weights) + `ReduceLROnPlateau`
- Evaluation beyond the metric: **MAE by true-popularity decile** — the model over-predicts the quietest tracks (median-imputed zeros are signal-carrying), and the notebook names that failure and its first fix (missing-indicator column)
- Data cleaning as first-class work: the CSV exports Indian digit grouping (`,`` separators), log10 transforms for 5-orders-of-magnitude targets, winsorization

## Run

```bash
uv venv && uv pip install tensorflow-cpu scikit-learn pandas matplotlib scipy
jupyter execute spotify_keras_success_model.ipynb   # or open and run all
```

Dataset: `C:/ASK_Main/ASK/Dashboards/Spotify 2024/spotify-dataset.csv` (my Power BI dashboard source, 4,600 tracks × 29 columns).

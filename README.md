# A-Share Neural Network

Deep learning pipeline for predicting **next-day direction** of A-share stocks.

This project trains an LSTM-Transformer ensemble on ~3M sequences built from ~5800 A-share stocks. The goal is simple: given recent stock + market context, estimate the probability that a stock goes **up on the next trading day**.

## What this repo does

- Collects and caches A-share market data from Tushare Pro
- Builds engineered features for individual stocks, market, sector, margin, and northbound flow
- Trains a **1-day-only** deep model on large-scale walk-forward splits
- Saves multiple trained models for ensemble inference
- Includes a terminal monitor for training progress
- Runs permutation importance to identify useful / neutral / harmful features

## Current status

**What works now**
- Data collection pipeline works
- Large-scale training pipeline works
- 10-model ensemble training works
- Feature importance analysis works
- Trained model checkpoints can be loaded from `models/`

**What still needs a small glue layer**
- The recommended production inference path is to use the **pretrained ensemble**
- A dedicated helper like `predict_with_ensemble(symbol)` still needs to be wired up cleanly
- `predict_stock()` exists, but it is **not** the preferred deployment path because it retrains a small model from scratch instead of using the pretrained ensemble

## Quick start

### 1. Clone

```bash
git clone https://github.com/heyixuan2/ashare-neural-network.git
cd ashare-neural-network
```

### 2. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install the core packages manually:

```bash
pip install torch tushare numpy pandas python-dotenv
```

### 3. Train

```bash
python scripts/train.py
```

### 4. Monitor training

```bash
python scripts/monitor.py
```

## Recommended way to use the trained models

Use the pretrained ensemble from `models/`.

```python
from tools.price_predictor import StockPredictor

models = StockPredictor.load_ensemble("models/", n_models=10)
```

The intended inference flow is:

1. Fetch recent stock prices
2. Fetch market / sector / northbound / extra data
3. Build features with `_feature_engineer(...)`
4. Build the last 30-day sequence
5. Load the 10 pretrained models
6. Run each model once
7. Average the predictions

That gives you a deployable probability `P(up)`.

## Why 1-day only?

Earlier versions predicted 1d / 3d / 5d together. In testing, 3-day and 5-day targets were close to random, so the project was simplified to focus all model capacity on the only horizon that showed usable signal: **next trading day**.

## High-level model design

Current model family:

```text
Input features → Linear + LayerNorm + GELU
→ LSTM (1 layer)
→ Transformer Encoder (2 layers, 4 heads)
→ Attention Pooling
→ MLP head
→ P(up tomorrow)
```

The model is intentionally small (~140K params) because daily stock prediction is a weak-signal problem and larger models tended to overfit quickly.

## Data and features

### Universe
- ~5800 A-share stocks
- 2022-01-01 to present
- Forward-adjusted prices (前复权)

### Feature groups
The model uses stock-level and market-context features such as:

- Returns and moving-average ratios
- Volatility and ATR
- RSI / MACD / Bollinger / KDJ
- Candlestick structure
- OBV and gap features
- Industry encoding
- Calendar signals
- Market return + relative strength
- Valuation and turnover
- Money flow
- Margin trading features
- Sector return and sector-relative strength
- Northbound flow (HSGT)

Latest training version uses **48 features** after removing several low-value / harmful features identified by permutation importance.

## Training setup

- Walk-forward temporal split
- 5 trading-day gap between train / val / test to prevent label leakage
- Memory-mapped `.npy` pipeline for large datasets
- AdamW optimizer
- Gradient accumulation
- Early stopping
- 10 random seeds for stability testing
- Permutation importance on the best model

## Results, in plain English

This is not a “90% accuracy” kind of problem. A-share daily direction is noisy and close to efficient.

What matters here is not raw overall accuracy alone, but whether **high-confidence predictions** have a measurable edge.

Recent training runs showed:
- overall accuracy is only modestly above 50%
- high-confidence buckets can show materially better win rate
- best runs reached roughly **5%–12% edge** in stronger confidence regions

So the project is better thought of as a **signal ranking / confidence filtering system**, not a magic all-stocks-always-right predictor.

## Project structure

```text
ashare-neural-network/
├── README.md
├── scripts/
│   ├── train.py                # user-facing training entrypoint
│   └── monitor.py              # user-facing monitor entrypoint
├── tools/
│   ├── train_predictor.py      # data collection + training pipeline
│   ├── price_predictor.py      # feature engineering + model definition + loading
│   └── monitor_training.py     # terminal dashboard implementation
├── models/                     # checkpoints, meta, loss curves, feature importance
│   └── splits/                 # train/val/test data (.npy, mmap)
├── data/ashare_daily/          # permanent raw OHLCV backup
└── .cache/                     # Tushare API cache
```

## Recommended next step for deployment

The main missing production step is a helper like:

```python
predict_with_ensemble(symbol)
```

That function should:
- fetch stock + global context data
- build the latest 48-dim feature tensor
- load the 10 pretrained models
- average predictions
- return a final probability and confidence bucket

Once that is added, the trained system is ready to plug into a frontend or API service.

## Version history

- **V2**: multi-horizon (1d/3d/5d), much larger model
- **V3**: switched to 1d-only, much smaller model
- **V4**: simpler loss / threshold logic / seed stability search
- **V4.1**: removed harmful features, improved warmup / regularization, 48-feature setup

## Requirements

- Python 3.10+
- PyTorch (MPS / CUDA / CPU)
- tushare
- numpy
- pandas
- python-dotenv

## License

MIT

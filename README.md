# A-Share Neural Network

LSTM-Transformer hybrid model for A-share stock 1-day price direction prediction.

## Architecture (V4)

```
Input (51-dim, seq_len=30) → Linear(64) + LayerNorm + GELU + Dropout(0.2)
  → LSTM(64, 1 layer) + LayerNorm + Positional Encoding
  → TransformerEncoder(2 layers, 4 heads, ffn=256, dropout=0.4)
  → LayerNorm → Attention Pooling
  → Dropout(0.5) → Linear(64→16) + GELU → Linear(16→1) → Sigmoid
```

**~140K parameters** — intentionally small to match low signal-to-noise ratio in daily stock data.

## Version History

| Version | Params | Config | Targets | Loss | LR Schedule | Key Change |
|---------|--------|--------|---------|------|-------------|------------|
| V2 | 1.2M | 128/8/4/3 | 1d+3d+5d | Asymmetric Focal | Cosine+Warmup | Baseline |
| V3 | 140K | 64/4/2/1 | 1d only | Asymmetric Focal | Cosine+Warmup(3ep) | Smaller model, +SWA |
| V4 | 140K | 64/4/2/1 | 1d only | Plain BCE | Constant 1e-4 | Remove SWA, temp scaling |

**Why 1d only?** V2 showed 3d/5d predictions ≈ random (48.5%/47.8% on test set). All capacity now focuses on the only horizon with detectable signal.

**V3→V4 changes:**
- LR: 3e-4 → 1e-4 (V3 best epoch consistently at warmup lr ≈ 1e-4)
- Batch: 1024 → 256 micro × 4 accumulation (implicit regularization via gradient noise)
- Loss: Asymmetric Focal BCE → plain BCE (fewer hyperparameters, same signal strength)
- Removed SWA (hurt probability calibration, negligible accuracy gain)
- Added post-hoc temperature scaling + optimal threshold search
- Label smoothing: 0.10 → 0.05

## Features (51 dimensions)

| Category | Features | Dims |
|----------|----------|------|
| Returns | 1-day, 5-day, 20-day log returns | 3 |
| Moving Averages | Price / MA ratio (5, 20, 60-day) | 3 |
| Volatility | Rolling std of returns (5, 20-day) | 2 |
| RSI | Relative Strength Index (14-day) | 1 |
| MACD | DIF, histogram (normalized by price) | 2 |
| Bollinger | Price position within Bollinger Band (20-day) | 1 |
| Volume | Volume / MA ratio (5, 20-day) | 2 |
| Candlestick | Body ratio, upper shadow, lower shadow | 3 |
| KDJ | K, D, J values (9-day, normalized to 0-1) | 3 |
| ATR | Average True Range ratio (14-day) | 1 |
| Price Position | Close within N-day high-low range (10, 30-day) | 2 |
| OBV | On-Balance Volume (20-day z-score, clipped) | 1 |
| Gap | Opening gap / previous close | 1 |
| Industry | Hash-based sector embedding (4-dim, isolated RNG) | 4 |
| Calendar | Day-of-week sin/cos, month sin/cos (cyclical) | 4 |
| Market | Index return (沪深300), stock-vs-index relative strength | 2 |
| Fundamental | PE, PB (log-transformed), dividend yield, turnover rate | 4 |
| Money Flow | Net inflow, big-order net ratio, small-order net ratio, big-order share | 4 |
| Margin | 融资余额, 融资净买入比, 融券占比, is_margin flag | 4 |
| Sector | 申万行业 daily return, stock-vs-sector differential | 2 |
| HSGT | Northbound flow (沪股通+深股通), flow / 5-day MA ratio | 2 |

## Data Pipeline

- **Source**: Tushare Pro API (~5800 A-share stocks, 2022-01-01 to present)
- **Price adjustment**: Forward-adjusted (前复权) via per-day `adj_factor`
- **Normalization**: Rolling 60-day z-score per feature (constant features like industry encoding preserved raw)
- **Sequence length**: 30 trading days
- **Labels**: Binary (up=1 / down=0) with ATR-adaptive threshold (`max(0.002, ATR_20d × 0.15)`); ambiguous moves → NaN → excluded
- **Storage**: Stream-to-disk `.bin` → memory-mapped `.npy` (handles 3M+ sequences without RAM bottleneck)

## Training (V4)

- **Temporal split** (walk-forward, no shuffling):
  - Train: ≤ 2025-06-25
  - Val: 2025-07-03 ~ 2025-10-24
  - Test: ≥ 2025-10-31
  - 5 trading-day gap between each split to prevent label boundary leakage
- **Loss**: `nn.functional.binary_cross_entropy` with valid-sample masking (labels at 0.5 = unknown, skipped)
- **Optimizer**: AdamW (lr=1e-4 constant, weight_decay=5e-2)
- **Batch**: 256 micro-batch × 4 gradient accumulation = 1024 effective batch
- **Label smoothing**: 0.05 (labels clipped to [0.05, 0.95])
- **Regularization**: Dropout 0.2 (input) + 0.4 (transformer) + 0.5 (head), weight decay 5e-2, grad clip 1.0
- **Early stopping**: patience=15 on val_loss
- **Post-hoc calibration**: Temperature scaling (T searched on val set, NLL minimized) + optimal classification threshold (F1 maximized on val set)
- **Ensemble**: 3 models with seeds {42, 43, 44}, max 60 epochs each

## V3 Baseline Results

```
              Test Acc  Win Rate(>55%)  Edge    High-Conf(≥0.70)
Model 0 (42): 50.8%    54.3%          +4.7%   67.7%
Model 1 (43): 50.5%    52.3%          +2.7%   63.5%
Model 2 (44): 52.3%    55.1%          +5.5%   —
```

Key observations: overall accuracy near 50% (expected for efficient market), but high-confidence predictions (P>0.7) show 60-68% actual win rate. V4 targets improving calibration so these confidence buckets become more reliable.

## Usage

```bash
# Train ensemble (3 models, ~20h each max)
python tools/train_predictor.py

# Monitor training progress in real-time
python monitor_training.py
```

## Project Structure

```
ashare-neural-network/
├── tools/
│   ├── train_predictor.py    # Data collection + training pipeline
│   └── price_predictor.py    # Feature engineering + model definition + inference
├── monitor_training.py       # Real-time training dashboard
├── models/                   # Trained weights, meta, loss curves
│   └── splits/               # Train/val/test data (.npy, mmap)
├── data/ashare_daily/        # Permanent raw data backup
└── .cache/                   # Tushare API response cache
```

## Requirements

- Python 3.10+
- PyTorch (MPS / CUDA / CPU)
- tushare, numpy, pandas, python-dotenv

## License

MIT

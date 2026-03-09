# A-Share Neural Network

LSTM-Transformer hybrid model for A-share stock 1-day price direction prediction.

## Architecture (V4)

```
Input (51-dim) → Linear(64) → LSTM(1 layer) → Transformer(2 layers, 4 heads)
  → Attention Pooling → Dropout(0.5) → Head(64→16→1) → Sigmoid
```

**140K parameters** — intentionally small to match low signal-to-noise ratio in daily stock data.

## Evolution

| Version | Architecture | Targets | Key Change | Best Edge |
|---------|-------------|---------|------------|-----------|
| V2 | 1.2M params, 128/8/4/3 | 1d+3d+5d | Focal Loss + SWA | +2.2% |
| V3 | 140K params, 64/4/2/1 | 1d only | Smaller model, stronger regularization | +5.5% |
| V4 | 140K params, 64/4/2/1 | 1d only | Constant LR, plain BCE, temperature scaling | TBD |

**Why 1d only?** V2 showed 3d/5d predictions ≈ random (48.5%/47.8% on test). All capacity now focuses on the only horizon with signal.

## Features (51 dimensions)

| Category | Features | Count |
|----------|----------|-------|
| Technical | Returns, MA ratios, RSI, MACD, Bollinger, KDJ, ATR, OBV, gap | 25 |
| Industry | Hash-encoded sector embedding | 4 |
| Calendar | Day-of-week, month (cyclical sin/cos) | 4 |
| Market | Index return, relative strength | 2 |
| Fundamental | PE, PB, dividend yield, turnover (log-transformed) | 4 |
| Money Flow | Net inflow, big/small order ratio | 4 |
| Margin | 融资余额, 融券比, 融资净买入, is_margin flag | 4 |
| Sector | Sector return, stock-vs-sector | 2 |
| HSGT | Northbound flow, normalized ratio | 2 |

## Data Pipeline

- **Source**: Tushare Pro API (~5800 A-share stocks, 2022-present)
- **Price adjustment**: Forward-adjusted (前复权) via `adj_factor`
- **Normalization**: Rolling 60-day z-score (constant features preserved raw)
- **Sequence length**: 30 trading days
- **Labels**: Binary (up/down) with ATR-adaptive threshold
- **Stream-to-disk**: Handles 3M+ sequences without RAM bottleneck

## Training (V4)

- **Temporal split**: Train ≤2025-06-25 / Val 2025-07-03~10-24 / Test ≥2025-10-31
- **5-day gap** between splits to prevent label boundary leakage
- **Loss**: Plain BCE with valid-sample masking (0.5 = unknown label, skipped)
- **Optimizer**: AdamW (lr=1e-4 constant, wd=5e-2)
- **Batch**: 256 micro × 4 accumulation = 1024 effective
- **Regularization**: Dropout 0.5, weight decay 5e-2, label smoothing 0.05
- **Post-hoc calibration**: Temperature scaling on val set + optimal threshold search
- **Ensemble**: 3 models (seeds 42, 43, 44), patience=15, max 60 epochs

## V3 Results (baseline for V4)

```
              Test Acc  Win Rate  Edge    High-Conf(0.70+)
Model 0 (42): 50.8%    54.3%    +4.7%   67.7%
Model 1 (43): 50.5%    52.3%    +2.7%   63.5%
Model 2 (44): 52.3%    55.1%    +5.5%   —
```

## Usage

```bash
# Train (3 ensemble models)
python tools/train_predictor.py

# Monitor training progress
python monitor_training.py
```

## Requirements

- Python 3.10+
- PyTorch (MPS/CUDA/CPU)
- tushare, numpy, pandas

## License

MIT

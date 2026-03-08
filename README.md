# A-Share Neural Network

LSTM-Transformer hybrid model for A-share stock price prediction.

## Architecture

```
Input (49-dim) → Linear(128) → LSTM(3 layers) → Transformer(4 layers, 8 heads)
  → Attention Pooling → Shared FC(64) → 3 prediction heads (1d/3d/5d)
```

## Features (49 dimensions)

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

## Data Pipeline

- **Source**: Tushare Pro API (~5800 A-share stocks, 2022-present)
- **Price adjustment**: Forward-adjusted (前复权) via `adj_factor`
- **Normalization**: Rolling 60-day z-score (constant features preserved)
- **Sequence length**: 30 trading days
- **Labels**: Binary (up/down) with ATR-adaptive threshold

## Training

- **Temporal split**: Train ≤2025-06-25 / Val 2025-07-03~10-24 / Test ≥2025-10-31
- **5-day gap** between splits to prevent label boundary leakage
- **Loss**: Asymmetric Focal BCE (γ=1.5, FP penalty=1.3)
- **Optimizer**: AdamW + CosineAnnealingWarmRestarts + SWA
- **Ensemble**: 3 models (seeds 42, 43, 44)
- **Stream-to-disk**: Handles 5M+ sequences without RAM bottleneck

## Usage

```bash
# Train
python tools/train_predictor.py

# Monitor
python monitor_training.py
```

## Requirements

- Python 3.10+
- PyTorch (MPS/CUDA/CPU)
- tushare, numpy, pandas

## License

MIT

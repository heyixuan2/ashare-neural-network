"""
Stock Price Direction Predictor — LSTM + Transformer (V3, 1d-only)
Predicts probability of price going UP in the next trading day.

Architecture:
- Feature engineering: 51 features (technical + fundamental + market-wide)
- Model: LSTM(1 layer) → Transformer(2 layers, 4 heads) → single head
- Output: P(up) for 1-day horizon
"""
import os
import json
import time
import hashlib
import math
import numpy as np
from typing import Optional, Dict, Any

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_model_cache = {}


def _feature_engineer(prices: list, market_data: dict = None, industry: str = None,
                      extra_data: dict = None, hsgt_data: dict = None,
                      sector_data: dict = None) -> np.ndarray:
    """
    Create features from OHLCV data + market-wide + industry + calendar.
    Input: list of dicts with open/high/low/close/volume/date
    Output: (N, F) numpy array
    """
    n = len(prices)
    if n < 60:
        return np.array([])

    close = np.array([p["close"] for p in prices], dtype=float)
    high = np.array([p["high"] for p in prices], dtype=float)
    low = np.array([p["low"] for p in prices], dtype=float)
    open_ = np.array([p["open"] for p in prices], dtype=float)
    volume = np.array([p["volume"] for p in prices], dtype=float)

    features = []

    # Returns
    ret1 = np.zeros(n); ret1[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)
    ret5 = np.zeros(n); ret5[5:] = (close[5:] - close[:-5]) / np.maximum(close[:-5], 1e-8)
    ret20 = np.zeros(n); ret20[20:] = (close[20:] - close[:-20]) / np.maximum(close[:-20], 1e-8)
    features.extend([ret1, ret5, ret20])

    # Moving averages ratio — keep 5, 20, 60 (drop 10, too correlated with 5)
    for w in [5, 20, 60]:
        ma = np.convolve(close, np.ones(w)/w, mode='full')[:n]
        ma[:w-1] = ma[w-1]
        ratio = (close - ma) / np.maximum(ma, 1e-8)
        features.append(ratio)

    # Volatility
    for w in [5, 20]:
        vol = np.zeros(n)
        for i in range(w, n):
            vol[i] = np.std(ret1[i-w+1:i+1])
        vol[:w] = vol[w] if w < n else 0
        features.append(vol)

    # RSI
    def calc_rsi(arr, period=14):
        rsi = np.full(n, 50.0)
        deltas = np.diff(arr, prepend=arr[0])
        for i in range(period, n):
            window = deltas[i-period+1:i+1]
            gains = np.mean(np.maximum(0, window))
            losses = np.mean(np.maximum(0, -window))
            if losses < 1e-10:
                rsi[i] = 100
            else:
                rsi[i] = 100 - 100 / (1 + gains / losses)
        return rsi / 100  # normalize to 0-1
    # RSI — keep only 14 (7 too correlated, r>0.85)
    features.append(calc_rsi(close, 14))

    # MACD
    def ema(arr, span):
        result = np.zeros(n)
        result[0] = arr[0]
        k = 2 / (span + 1)
        for i in range(1, n):
            result[i] = arr[i] * k + result[i-1] * (1 - k)
        return result

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    dif = ema12 - ema26
    dea = ema(dif, 9)
    macd = 2 * (dif - dea)
    # MACD — keep DIF and histogram only (DEA ≈ DIF smoothed, r>0.95)
    features.append(dif / np.maximum(close, 1e-8))
    features.append(macd / np.maximum(close, 1e-8))

    # Bollinger Band position
    bb_mid = np.convolve(close, np.ones(20)/20, mode='full')[:n]
    bb_mid[:19] = bb_mid[19]
    bb_std = np.zeros(n)
    for i in range(20, n):
        bb_std[i] = np.std(close[i-19:i+1])
    bb_std[:20] = bb_std[20] if 20 < n else 1
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (close - bb_lower) / np.maximum(bb_upper - bb_lower, 1e-8)
    features.append(np.clip(bb_pos, 0, 1))

    # Volume features
    vol_ma5 = np.convolve(volume, np.ones(5)/5, mode='full')[:n]
    vol_ma5[:4] = vol_ma5[4]
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='full')[:n]
    vol_ma20[:19] = vol_ma20[19]
    vol_ratio_5 = volume / np.maximum(vol_ma5, 1)
    vol_ratio_20 = volume / np.maximum(vol_ma20, 1)
    features.extend([vol_ratio_5, vol_ratio_20])

    # Candlestick features
    body = (close - open_) / np.maximum(np.abs(high - low), 1e-8)
    upper_shadow = (high - np.maximum(close, open_)) / np.maximum(np.abs(high - low), 1e-8)
    lower_shadow = (np.minimum(close, open_) - low) / np.maximum(np.abs(high - low), 1e-8)
    features.extend([body, upper_shadow, lower_shadow])

    # KDJ
    k_vals = np.full(n, 50.0)
    d_vals = np.full(n, 50.0)
    for i in range(9, n):
        low9 = np.min(low[i-8:i+1])
        high9 = np.max(high[i-8:i+1])
        rsv = (close[i] - low9) / max(high9 - low9, 1e-8) * 100
        k_vals[i] = 2/3 * k_vals[i-1] + 1/3 * rsv
        d_vals[i] = 2/3 * d_vals[i-1] + 1/3 * k_vals[i]
    j_vals = 3 * k_vals - 2 * d_vals
    features.extend([k_vals/100, d_vals/100, j_vals/100])

    # Trend strength (ADX-like: ratio of directional move to range)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr
    atr_ma = np.convolve(atr, np.ones(14)/14, mode='full')[:n]
    atr_ma[:13] = atr_ma[13] if 13 < n else 1
    features.append(atr_ma / np.maximum(close, 1e-8))  # ATR ratio

    # Price position within range (where is close relative to N-day high/low)
    for w in [10, 30]:
        pos = np.zeros(n)
        for i in range(w, n):
            h_w = np.max(high[i-w:i+1])
            l_w = np.min(low[i-w:i+1])
            pos[i] = (close[i] - l_w) / max(h_w - l_w, 1e-8)
        pos[:w] = 0.5
        features.append(pos)

    # Volume trend (OBV-like: cumulative volume direction)
    obv = np.zeros(n)
    for i in range(1, n):
        obv[i] = obv[i-1] + (volume[i] if close[i] > close[i-1] else -volume[i])
    obv_norm = np.zeros(n)
    for i in range(20, n):
        std = np.std(obv[i-19:i+1])
        obv_norm[i] = (obv[i] - np.mean(obv[i-19:i+1])) / max(std, 1e-8)
    obv_norm[:20] = 0
    features.append(np.clip(obv_norm, -3, 3) / 3)  # normalize to [-1, 1]

    # Gap (opening gap as fraction of previous close)
    gap = np.zeros(n)
    gap[1:] = (open_[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)
    features.append(gap)

    # Industry encoding (hash-based, 4 dims)
    # Uses separate RandomState to avoid corrupting global numpy seed
    if industry:
        ind_hash = hash(industry) % 10000
        rng = np.random.RandomState(ind_hash)  # isolated RNG
        ind_features = rng.randn(4) * 0.5
        for dim in range(4):
            features.append(np.full(n, ind_features[dim]))
    else:
        for dim in range(4):
            features.append(np.zeros(n))

    # Calendar effects (cyclical encoding — no discontinuity at boundaries)
    day_of_week_sin = np.zeros(n)
    day_of_week_cos = np.zeros(n)
    month_sin = np.zeros(n)
    month_cos = np.zeros(n)
    for i, p in enumerate(prices):
        date_str = p.get("date", "")
        if len(date_str) >= 8:
            try:
                from datetime import datetime as dt
                d = dt.strptime(date_str[:8], "%Y%m%d") if len(date_str) == 8 else dt.strptime(date_str[:10], "%Y-%m-%d")
                dow = d.weekday()  # 0=Mon, 4=Fri
                mon = d.month
                day_of_week_sin[i] = np.sin(2 * np.pi * dow / 5)
                day_of_week_cos[i] = np.cos(2 * np.pi * dow / 5)
                month_sin[i] = np.sin(2 * np.pi * mon / 12)
                month_cos[i] = np.cos(2 * np.pi * mon / 12)
            except:
                pass
    features.extend([day_of_week_sin, day_of_week_cos, month_sin, month_cos])

    # Market-wide features (if available)
    if market_data:
        idx_ret = np.zeros(n)
        for i, p in enumerate(prices):
            date = p.get("date", "")
            if date in market_data:
                idx_ret[i] = market_data[date].get("idx_ret", 0)
        features.append(idx_ret)

        # Relative strength: stock return vs market return
        rel_strength = ret1 - idx_ret
        features.append(rel_strength)
    else:
        features.extend([np.zeros(n), np.zeros(n)])

    # ── Extra features from daily_basic + moneyflow ──
    if extra_data:
        # Valuation features (normalized by rolling stats in _build_sequences)
        pe_arr = np.zeros(n)
        pb_arr = np.zeros(n)
        dv_arr = np.zeros(n)  # dividend yield
        turnover_arr = np.zeros(n)
        # Fund flow features
        net_mf_arr = np.zeros(n)  # 净主力资金 (万元, will be normalized)
        big_net_arr = np.zeros(n)  # 大单净比 (already -1 to 1)
        sm_net_arr = np.zeros(n)  # 散户净比
        big_ratio_arr = np.zeros(n)  # 大单占比 (0 to 1)

        for i, p in enumerate(prices):
            date = p.get("date", "")
            if date in extra_data:
                ed = extra_data[date]
                pe_arr[i] = ed.get("pe", 0)
                pb_arr[i] = ed.get("pb", 0)
                dv_arr[i] = ed.get("dv", 0)
                turnover_arr[i] = ed.get("turnover", 0)
                net_mf_arr[i] = ed.get("net_mf", 0)
                big_net_arr[i] = ed.get("big_net", 0)
                sm_net_arr[i] = ed.get("sm_net", 0)
                big_ratio_arr[i] = ed.get("big_ratio", 0)

        # Log-transform PE/PB to reduce skew (PE can be 5 or 5000)
        pe_log = np.sign(pe_arr) * np.log1p(np.abs(pe_arr))
        pb_log = np.sign(pb_arr) * np.log1p(np.abs(pb_arr))

        # Margin trading features (融资融券)
        rzye_arr = np.zeros(n)  # 融资余额
        rz_net_arr = np.zeros(n)  # 融资净买入比
        rq_ratio_arr = np.zeros(n)  # 融券占比 (做空情绪)
        is_margin_arr = np.zeros(n)  # 当天是否在两融名单 (0/1)

        for i, p in enumerate(prices):
            date = p.get("date", "")
            if date in extra_data:
                ed = extra_data[date]
                rzye_arr[i] = ed.get("rzye", 0)
                rz_net_arr[i] = ed.get("rz_net", 0)
                rq_ratio_arr[i] = ed.get("rq_ratio", 0)
                # is_margin = 1 if this date has real margin data (rzye > 0)
                is_margin_arr[i] = 1.0 if ed.get("rzye", 0) > 0 else 0.0

        features.extend([pe_log, pb_log, dv_arr, turnover_arr,
                         net_mf_arr, big_net_arr, sm_net_arr, big_ratio_arr,
                         rzye_arr, rz_net_arr, rq_ratio_arr, is_margin_arr])
    else:
        # Pad with zeros if no extra data
        for _ in range(12):
            features.append(np.zeros(n))

    # ── Sector (申万行业指数) daily return ──
    if sector_data:
        sector_ret = np.zeros(n)
        sector_vs_stock = np.zeros(n)  # stock return - sector return
        for i, p in enumerate(prices):
            date = p.get("date", "")
            if date in sector_data:
                sector_ret[i] = sector_data[date]
                sector_vs_stock[i] = ret1[i] - sector_data[date]
        features.extend([sector_ret, sector_vs_stock])
    else:
        features.extend([np.zeros(n), np.zeros(n)])

    # ── HSGT (北向资金 / 南向资金) ──
    if hsgt_data:
        north_flow = np.zeros(n)  # 沪股通 + 深股通 (百万)
        for i, p in enumerate(prices):
            date = p.get("date", "")
            if date in hsgt_data:
                north_flow[i] = hsgt_data[date].get("hgt", 0) + hsgt_data[date].get("sgt", 0)
        # 5-day MA ratio: today's flow vs recent average (captures flow momentum)
        nf_ma5 = np.convolve(north_flow, np.ones(5)/5, mode='full')[:n]
        nf_ma5[:4] = nf_ma5[4]
        nf_ratio = north_flow / np.maximum(np.abs(nf_ma5), 1.0)
        features.extend([north_flow, nf_ratio])
    else:
        features.extend([np.zeros(n), np.zeros(n)])

    # Stack: (N, F)
    X = np.stack(features, axis=1)

    # Replace NaN/Inf — use ±4 to match clip range in _build_sequences
    X = np.nan_to_num(X, nan=0.0, posinf=4.0, neginf=-4.0)

    return X


def _create_labels(prices: list, horizons=[1, 3, 5], min_threshold=0.002) -> Dict[int, np.ndarray]:
    """
    Create binary labels with ADAPTIVE threshold per-stock.
    Threshold = max(min_threshold, ATR_20d * 0.15)

    Why adaptive:
    - Bank stocks: daily vol ~0.3%, tiny moves are noise
    - ChiNext stocks: daily vol ~3%, same absolute threshold is too aggressive
    - Each stock gets a threshold matching its own volatility

    Conservative multiplier (0.15): keeps ~85% of data while still filtering
    the smallest noise. Previous 0.3 dropped 96% of val data!

    Moves within ±threshold → NaN → dropped from training.
    """
    close = np.array([p["close"] for p in prices], dtype=float)
    high = np.array([p["high"] for p in prices], dtype=float)
    low = np.array([p["low"] for p in prices], dtype=float)
    n = len(close)

    # Compute per-day adaptive threshold based on 20-day ATR
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr
    # Rolling 20-day mean ATR, as ratio of close price
    atr_ratio = np.zeros(n)
    for i in range(20, n):
        atr_ratio[i] = np.mean(atr[i-19:i+1]) / max(close[i], 1e-8)
    atr_ratio[:20] = atr_ratio[20] if 20 < n else min_threshold

    labels = {}
    for h in horizons:
        y = np.full(n, np.nan)
        for i in range(n - h):
            ret = (close[i + h] - close[i]) / max(close[i], 1e-8)
            thresh = max(min_threshold, atr_ratio[i] * 0.15)
            if ret > thresh:
                y[i] = 1.0
            elif ret < -thresh:
                y[i] = 0.0
        labels[h] = y
    return labels


def _build_sequences(X: np.ndarray, y: Dict[int, np.ndarray], seq_len=30, norm_window=60):
    """
    Create sequences with ROLLING WINDOW normalization.

    Why rolling (not expanding):
    - Stock statistics drift over time (regime change)
    - 2022 mean/std is irrelevant for 2024 data
    - Rolling 60-day window captures recent regime only

    Why not full-dataset normalization:
    - Uses future data = look-ahead bias = cheating

    Each sequence normalized using the PRECEDING norm_window days only.
    """
    n = X.shape[0]
    seqs = []
    targets = {h: [] for h in y}
    valid_indices = []  # track which price indices survived NaN filtering

    # Start from max(seq_len, norm_window) to have enough history
    start = max(seq_len, norm_window)

    for i in range(start, n):
        # Rolling window: use [i-norm_window : i] for stats (NO future data)
        lookback = X[i-norm_window:i]
        mean = lookback.mean(axis=0)
        std = lookback.std(axis=0)

        # Normalize the sequence window
        raw_seq = X[i-seq_len:i].copy()

        # For constant features (std ≈ 0, e.g. industry hash, is_margin):
        # Keep raw values instead of normalizing to 0.
        # This preserves cross-stock discriminative power.
        const_mask = std < 1e-6
        std = np.maximum(std, 1e-8)
        norm_seq = (raw_seq - mean) / std
        # Restore constant features to their raw values
        if const_mask.any():
            norm_seq[:, const_mask] = raw_seq[:, const_mask]

        # Clip extreme values (>4 sigma likely noise), but skip constant features
        clip_mask = ~const_mask
        norm_seq[:, clip_mask] = np.clip(norm_seq[:, clip_mask], -4, 4)

        # LABEL ALIGNMENT FIX:
        # Sequence X[i-30:i] sees features through day i-1 (latest close = close[i-1])
        # y[h][i-1] = (close[i-1+h] - close[i-1]) / close[i-1]
        # This means: "given data up to day i-1, predict return over next h days"
        # Before fix: used y[h][i] which skipped a day (2-step-ahead)
        label_idx = i - 1

        # Only require 1d label valid (primary horizon)
        # For 3d/5d: if NaN, fill with 0.5 (model learns to be uncertain)
        # This preserves MUCH more data in val/test periods
        if label_idx < 0 or np.isnan(y[1][label_idx]):
            continue  # must have 1d label

        seqs.append(norm_seq)
        valid_indices.append(label_idx)  # the price index this sequence's label refers to
        for h in y:
            val = y[h][label_idx]
            targets[h].append(val if not np.isnan(val) else 0.5)

    if not seqs:
        empty_X = np.array([], dtype=np.float32).reshape(0, seq_len, X.shape[1])
        empty_y = {h: np.array([], dtype=np.float32) for h in y}
        return empty_X, empty_y, np.array([], dtype=np.int64)

    return (np.array(seqs, dtype=np.float32),
            {h: np.array(v, dtype=np.float32) for h, v in targets.items()},
            np.array(valid_indices, dtype=np.int64))


class StockPredictor:
    """LSTM + Transformer hybrid model for stock prediction — V3 (1d-only)

    Architecture must match train_predictor.py HybridModel exactly:
    - InputProj(input_dim→64, dropout=0.2) → LSTM(1 layer, 64)
    - Transformer(2 layers, 4 heads, ff=256, dropout=0.4)
    - AttentionPooling → Dropout(0.5) → head (64→16→1)
    """

    # Default architecture matches V3 train_predictor.py
    TRAIN_ARCH = {"hidden_dim": 64, "n_heads": 4, "n_layers": 2, "lstm_layers": 1}

    def __init__(self, input_dim, seq_len=30, hidden_dim=64, n_heads=4, n_layers=2, lstm_layers=1):
        import torch
        import torch.nn as nn

        self.seq_len = seq_len
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        class HybridModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_heads, n_layers, lstm_layers=1):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                )
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                                     num_layers=lstm_layers, dropout=0.0)
                self.lstm_norm = nn.LayerNorm(hidden_dim)
                self.pos_enc = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim*4, dropout=0.4,
                    batch_first=True, activation='gelu'
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.out_norm = nn.LayerNorm(hidden_dim)
                self.attn_pool = nn.Linear(hidden_dim, 1)
                # Single head — 1d prediction only
                self.head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dim, 16), nn.GELU(),
                    nn.Linear(16, 1), nn.Sigmoid())

            def forward(self, x):
                x = self.input_proj(x)
                h, _ = self.lstm(x)
                h = self.lstm_norm(h) + self.pos_enc
                t = self.out_norm(self.transformer(h))
                w = torch.softmax(self.attn_pool(t), dim=1)
                p = (t * w).sum(dim=1)
                return self.head(p)

        self.model = HybridModel(input_dim, hidden_dim, n_heads, n_layers, lstm_layers).to(self.device)

    @classmethod
    def load_trained(cls, model_dir, device=None):
        """Load a model trained by train_predictor.py"""
        import torch, json
        from pathlib import Path
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "predictor_meta.json").read_text())
        predictor = cls(
            input_dim=meta["input_dim"],
            seq_len=meta["seq_len"],
            hidden_dim=meta["hidden_dim"],
            n_heads=meta["n_heads"],
            n_layers=meta["n_layers"],
            lstm_layers=meta.get("lstm_layers", 1),
        )
        state = torch.load(model_dir / "predictor_best.pt", map_location=predictor.device, weights_only=True)
        predictor.model.load_state_dict(state)
        predictor.model.eval()
        return predictor

    @classmethod
    def load_ensemble(cls, model_dir, n_models=3):
        """Load ensemble of models trained by train_predictor.py"""
        import torch, json
        from pathlib import Path
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "predictor_meta.json").read_text())
        predictors = []
        for i in range(n_models):
            suffix = f"_{i}" if i > 0 else ""
            pt_path = model_dir / f"predictor_best{suffix}.pt"
            if not pt_path.exists():
                continue
            p = cls(
                input_dim=meta["input_dim"],
                seq_len=meta["seq_len"],
                hidden_dim=meta["hidden_dim"],
                n_heads=meta["n_heads"],
                n_layers=meta["n_layers"],
            lstm_layers=meta.get("lstm_layers", 1),
        )
            state = torch.load(pt_path, map_location=p.device, weights_only=True)
            p.model.load_state_dict(state)
            p.model.eval()
            predictors.append(p)
        return predictors

    def train_model(self, X_seq, y_dict, epochs=200, lr=0.001):
        import torch
        import torch.nn as nn

        # Time-series split: walk-forward (last 20% for val, strict temporal order)
        n = len(X_seq)
        split = int(n * 0.8)

        X_train_np = X_seq[:split]
        X_val = torch.FloatTensor(X_seq[split:]).to(self.device)

        y1_train = np.clip(y_dict[1][:split], 0.1, 0.9)  # label smoothing
        y1_val = torch.clamp(torch.FloatTensor(y_dict[1][split:]).to(self.device), 0.1, 0.9)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=epochs, pct_start=0.1,
            anneal_strategy='cos', final_div_factor=100
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state = None
        patience = 10
        no_improve = 0
        batch_size = 64

        for epoch in range(epochs):
            self.model.train()
            indices = np.random.permutation(len(X_train_np))
            epoch_loss, n_batches = 0, 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start+batch_size]
                X_batch = torch.FloatTensor(X_train_np[batch_idx]).to(self.device)
                y_batch = torch.FloatTensor(y1_train[batch_idx]).to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred.squeeze(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                vp = self.model(X_val)
                val_loss = criterion(vp.squeeze(), y1_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Compute accuracy on validation set
        self.model.eval()
        y1_val_hard = torch.FloatTensor(y_dict[1][split:]).to(self.device)
        with torch.no_grad():
            vp = self.model(X_val)
            acc1 = ((vp.squeeze() > 0.5).float() == y1_val_hard).float().mean().item()

        return {
            "epochs_trained": epoch + 1,
            "val_loss": round(best_val_loss, 4),
            "accuracy_1d": round(acc1 * 100, 1),
        }

    def predict(self, X_seq_last):
        """Predict on the last sequence — 1d only"""
        import torch
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_seq_last).unsqueeze(0).to(self.device)
            pred = self.model(x)
            return {
                "1d": round(pred.item() * 100, 1),
            }


def predict_stock(symbol: str, prices: list) -> Dict[str, Any]:
    """
    Full prediction pipeline:
    1. Feature engineering
    2. Train model on historical data
    3. Predict next day direction
    Returns: probabilities + training metrics
    """
    cache_key = f"pred_{symbol}_{len(prices)}"
    cached = None
    cache_path = os.path.join(CACHE_DIR, f"pred_{hashlib.md5(cache_key.encode()).hexdigest()}.json")

    # Check cache (4 hour TTL)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                obj = json.load(f)
            if time.time() - obj.get("ts", 0) < 14400:
                return obj["data"]
        except:
            pass

    if len(prices) < 120:
        return {"error": "需要至少120天数据用于训练"}

    try:
        # Feature engineering
        X = _feature_engineer(prices)
        if X.size == 0:
            return {"error": "特征工程失败"}

        # Labels
        y = _create_labels(prices)

        # Sequences
        seq_len = 30
        X_seq, y_seq, _valid_indices = _build_sequences(X, y, seq_len=seq_len)

        if len(X_seq) < 50:
            return {"error": "数据序列不足"}

        # NO additional normalization here — _build_sequences already does
        # rolling 60-day z-score + clip ±4. Double normalization would distort features.

        # Build & train — V3: smaller model, 1d only
        input_dim = X_seq.shape[2]
        predictor = StockPredictor(input_dim=input_dim, seq_len=seq_len)
        metrics = predictor.train_model(X_seq, y_seq, epochs=150, lr=0.0005)

        # Predict
        probs = predictor.predict(X_seq[-1])

        # Confidence based on how far from 50%
        confidence = {
            "1d": round(abs(probs["1d"] - 50) / 50 * 100, 1),
        }

        result = {
            "symbol": symbol,
            "predictions": probs,  # P(up) in %
            "confidence": confidence,
            "directions": {
                "1d": "涨" if probs["1d"] > 50 else "跌",
            },
            "training": metrics,
            "model": "LSTM-Transformer V3 (1d-only)",
            "features": input_dim,
            "sequences": len(X_seq),
            "timestamp": int(time.time()),
        }

        # Cache
        try:
            with open(cache_path, "w") as f:
                json.dump({"ts": time.time(), "data": result}, f)
        except:
            pass

        return result

    except Exception as e:
        return {"error": str(e)}

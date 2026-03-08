#!/usr/bin/env python3
"""
Large-scale Stock Price Predictor Training Script
Trains on ALL A-share stocks via Tushare (5000+ stocks × 2 years)

MEMORY-EFFICIENT: streams data to disk, never holds all sequences in RAM.
"""
import os
import sys
import json
import time
import math
import random
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import tushare as ts
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
pro = ts.pro_api(TUSHARE_TOKEN)

CACHE_DIR = PROJECT_ROOT / ".cache"
PERMANENT_DATA = PROJECT_ROOT / "data" / "ashare_daily"  # permanent backup, never auto-delete
MODEL_DIR = PROJECT_ROOT / "models"
SPLIT_DIR = MODEL_DIR / "splits"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SPLIT_DIR.mkdir(exist_ok=True)

LOG_FILE = MODEL_DIR / "training_log.jsonl"

def log(msg):
    ts_str = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"ts": time.time(), "msg": msg}, ensure_ascii=False) + "\n")


# ============================================================
# Phase 1: Data Collection - Stream to disk
# ============================================================

def fetch_stock_list():
    cache_path = CACHE_DIR / "all_stocks_list.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 86400):
        return json.loads(cache_path.read_text())
    log("Fetching stock list from Tushare (including delisted)...")
    # Include delisted (D) and paused (P) stocks to avoid survivorship bias
    dfs = []
    for status in ['L', 'D', 'P']:
        try:
            df = pro.stock_basic(exchange='', list_status=status, fields='ts_code,name,industry,market')
            if df is not None and not df.empty:
                dfs.append(df)
        except:
            pass
    df = pd.concat(dfs, ignore_index=True) if dfs else dfs[0]
    stocks = df.to_dict('records')
    cache_path.write_text(json.dumps(stocks, ensure_ascii=False))
    log(f"Got {len(stocks)} stocks")
    return stocks


def fetch_daily_data(ts_code):
    """Fetch OHLCV only (legacy, used by old cache)"""
    cache_key = hashlib.md5(f"{ts_code}_20220101".encode()).hexdigest()
    cache_path = CACHE_DIR / f"daily_{cache_key}.json"
    perm_path = PERMANENT_DATA / f"{ts_code.replace('.', '_')}.json"

    for p in [cache_path, perm_path]:
        if p.exists():
            if p == perm_path or (time.time() - p.stat().st_mtime < 86400 * 7):
                try:
                    return json.loads(p.read_text())
                except:
                    pass
    try:
        # Fetch raw daily + adj_factor for forward-adjustment (前复权)
        df = pro.daily(ts_code=ts_code, start_date="20220101")
        if df is None or df.empty:
            return []
        df = df.sort_values("trade_date")

        # Forward adjustment: price_qfq = price_raw * (adj_factor / latest_adj_factor)
        # This makes all historical prices comparable to the latest price
        try:
            adj_df = pro.adj_factor(ts_code=ts_code, start_date="20220101")
            if adj_df is not None and not adj_df.empty:
                adj_map = dict(zip(adj_df["trade_date"], adj_df["adj_factor"]))
                latest_adj = max(adj_map.values())
                for col in ["open", "high", "low", "close"]:
                    df[col] = df.apply(
                        lambda r: float(r[col]) * adj_map.get(r["trade_date"], latest_adj) / latest_adj,
                        axis=1)
        except Exception:
            pass  # Fall back to unadjusted if adj_factor fails

        records = [{"date": r["trade_date"], "open": float(r["open"]), "high": float(r["high"]),
                     "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["vol"])}
                    for _, r in df.iterrows()]
        cache_path.write_text(json.dumps(records))
        PERMANENT_DATA.mkdir(parents=True, exist_ok=True)
        perm_path.write_text(json.dumps(records))
        return records
    except Exception as e:
        return []


def fetch_extra_data(ts_code):
    """Fetch daily_basic + moneyflow + margin_detail for a stock.
    Returns {date: {field: val}}
    Cache validation: only use cache if it has ALL expected fields."""
    cache_path = CACHE_DIR / f"extra_{ts_code.replace('.', '_')}.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 86400 * 7):
        try:
            cached = json.loads(cache_path.read_text())
            # Validate cache has ALL required fields (basic + flow + margin)
            # Check MULTIPLE dates, not just the first - some dates may be incomplete
            if cached:
                dates = sorted(cached.keys())
                # Check a date from the middle of the range (avoids edge effects)
                check_date = dates[len(dates) // 2]
                sample = cached[check_date]
                if "big_net" in sample and "rzye" in sample:
                    return cached
                # else: incomplete cache, re-fetch
        except:
            pass

    result = {}
    got_basic = False
    got_flow = False
    got_margin = False
    max_retries = 5

    # daily_basic - retry until success
    for attempt in range(max_retries):
        try:
            df = pro.daily_basic(ts_code=ts_code, start_date="20220101",
                                 fields="trade_date,turnover_rate_f,volume_ratio,pe_ttm,pb,ps_ttm,dv_ttm,total_mv,circ_mv")
            if df is not None and not df.empty:
                got_basic = True
                for _, r in df.iterrows():
                    d = r["trade_date"]
                    result[d] = {
                        "turnover": float(r["turnover_rate_f"]) if pd.notna(r["turnover_rate_f"]) else 0,
                        "vol_ratio": float(r["volume_ratio"]) if pd.notna(r["volume_ratio"]) else 1,
                        "pe": float(r["pe_ttm"]) if pd.notna(r["pe_ttm"]) else 0,
                        "pb": float(r["pb"]) if pd.notna(r["pb"]) else 0,
                        "ps": float(r["ps_ttm"]) if pd.notna(r["ps_ttm"]) else 0,
                        "dv": float(r["dv_ttm"]) if pd.notna(r["dv_ttm"]) else 0,
                        "total_mv": float(r["total_mv"]) if pd.notna(r["total_mv"]) else 0,
                        "circ_mv": float(r["circ_mv"]) if pd.notna(r["circ_mv"]) else 0,
                    }
            else:
                got_basic = True  # stock exists but no daily_basic data (rare)
            break
        except Exception as e:
            wait = min(15 * (attempt + 1), 65)
            time.sleep(wait)

    # Pause between API calls to avoid shared rate limit
    # time.sleep(0.15)  # removed: API latency is natural throttle

    # moneyflow - retry until success
    for attempt in range(max_retries):
        try:
            df = pro.moneyflow(ts_code=ts_code, start_date="20220101")
            if df is not None and not df.empty:
                got_flow = True
                for _, r in df.iterrows():
                    d = r["trade_date"]
                    if d not in result:
                        result[d] = {}
                    buy_big = (float(r.get("buy_elg_amount", 0) or 0) + float(r.get("buy_lg_amount", 0) or 0))
                    sell_big = (float(r.get("sell_elg_amount", 0) or 0) + float(r.get("sell_lg_amount", 0) or 0))
                    buy_sm = (float(r.get("buy_sm_amount", 0) or 0) + float(r.get("buy_md_amount", 0) or 0))
                    sell_sm = (float(r.get("sell_sm_amount", 0) or 0) + float(r.get("sell_md_amount", 0) or 0))
                    total = buy_big + sell_big + buy_sm + sell_sm
                    result[d]["net_mf"] = float(r.get("net_mf_amount", 0) or 0)
                    result[d]["big_net"] = (buy_big - sell_big) / max(total, 1)
                    result[d]["sm_net"] = (buy_sm - sell_sm) / max(total, 1)
                    result[d]["big_ratio"] = (buy_big + sell_big) / max(total, 1)
            else:
                got_flow = True
            break
        except Exception as e:
            wait = min(15 * (attempt + 1), 65)
            time.sleep(wait)

    # Pause between API calls
    # time.sleep(0.15)  # removed: API latency is natural throttle

    # margin_detail: 融资融券
    # Retry until success - never fake data with zeros
    for attempt in range(max_retries):
        try:
            df = pro.margin_detail(ts_code=ts_code, start_date="20220101")
            if df is not None and not df.empty:
                # 两融标的 - has real margin data
                got_margin = True
                margin_dates = set()
                for _, r in df.iterrows():
                    d = r["trade_date"]
                    margin_dates.add(d)
                    if d not in result:
                        result[d] = {}
                    rzye = float(r.get("rzye", 0) or 0)
                    rqye = float(r.get("rqye", 0) or 0)
                    rzmre = float(r.get("rzmre", 0) or 0)
                    rzche = float(r.get("rzche", 0) or 0)
                    result[d]["rzye"] = rzye
                    result[d]["rz_net"] = (rzmre - rzche) / max(rzmre + rzche, 1)
                    result[d]["rq_ratio"] = rqye / max(rzye + rqye, 1)
                # Fill zeros for dates without margin (stock may have been
                # added/removed from 两融 list over time - partial coverage)
                for d in list(result.keys()):
                    if d not in margin_dates and "rzye" not in result[d]:
                        result[d]["rzye"] = 0
                        result[d]["rz_net"] = 0
                        result[d]["rq_ratio"] = 0
            else:
                # Empty result = genuinely not a 两融标的
                # Zero IS the truth here: this stock has no margin trading
                got_margin = True
                for d in list(result.keys()):
                    result[d]["rzye"] = 0
                    result[d]["rz_net"] = 0
                    result[d]["rq_ratio"] = 0
            break
        except Exception as e:
            # API error (rate limit, network, etc.) - retry with backoff
            wait = min(15 * (attempt + 1), 65)  # 15s, 30s, 45s, 60s, 65s
            log(f"    margin_detail {ts_code} attempt {attempt+1} failed: {str(e)[:80]}, retrying in {wait}s")
            time.sleep(wait)

    # Only cache if ALL three APIs succeeded - never save partial data
    if result and got_basic and got_flow and got_margin:
        cache_path.write_text(json.dumps(result))
    return result


def fetch_hsgt_data():
    """Fetch 北向/南向 资金 (market-wide, one call)"""
    cache_path = CACHE_DIR / "hsgt_flow.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 86400 * 7):
        try:
            return json.loads(cache_path.read_text())
        except:
            pass
    result = {}
    try:
        df = pro.moneyflow_hsgt(start_date="20220101")
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                result[r["trade_date"]] = {
                    "hgt": float(r.get("hgt", 0) or 0),  # 沪股通 (百万)
                    "sgt": float(r.get("sgt", 0) or 0),  # 深股通 (百万)
                }
    except:
        pass
    if result:
        cache_path.write_text(json.dumps(result))
    return result


def collect_data():
    """Stream-to-disk data collection. Never holds all data in RAM."""
    flag_file = MODEL_DIR / "data_ready.flag"

    # Check if splits already exist
    if flag_file.exists() and (SPLIT_DIR / "X_train.npy").exists():
        age = time.time() - flag_file.stat().st_mtime
        if age < 86400 * 3:
            log("Loading cached split data...")
            result = {}
            for split in ["train", "val", "test"]:
                result[f"X_{split}"] = np.load(SPLIT_DIR / f"X_{split}.npy", mmap_mode='r')
                result[f"y_{split}"] = {h: np.load(SPLIT_DIR / f"y_{split}_{h}d.npy", mmap_mode='r') for h in [1,3,5]}
            log(f"Loaded: train={len(result['X_train'])}, val={len(result['X_val'])}, test={len(result['X_test'])}")
            return result

    from tools.price_predictor import _feature_engineer, _create_labels, _build_sequences

    stocks = fetch_stock_list()
    n = len(stocks)
    seq_len = 30

    # ============================================================
    # TEMPORAL SPLIT - Walk-forward, no look-ahead (Plan B: 85/8/7)
    # Train: 2022-01 ~ 2025-06-30 (3.5 years)
    # Val:   2025-07 ~ 2025-10-31 (4 months)
    # Test:  2025-11 ~ 2026-03-06 (5 months)
    #
    # Every stock goes through ALL splits based on DATE.
    # Dates come from valid_indices (NaN-skipped samples removed).
    # Normalization: rolling 60-day z-score (NOT expanding window).
    # ============================================================
    # Gap of 5 trading days between splits to avoid label boundary leakage
    # (5d label from last train sample would peek into val period)
    TRAIN_END = "20250625"   # train labels use close up to ~20250702
    VAL_START = "20250703"   # 5-day gap after TRAIN_END
    VAL_END = "20251024"     # val labels use close up to ~20251031
    TEST_START = "20251031"  # 5-day gap after VAL_END
    # Test: everything after TEST_START
    log(f"Temporal split: train<={TRAIN_END}, gap, val={VAL_START}~{VAL_END}, gap, test>={TEST_START}")
    split_map = None  # not used anymore - split by date per-sequence

    # Open binary files for streaming writes
    # Use 'ab' (append) if resuming from checkpoint, 'wb' (fresh) otherwise
    writers = {}
    counts = {"train": 0, "val": 0, "test": 0}
    feature_dim = None

    write_mode = "ab" if (SPLIT_DIR / "progress.json").exists() else "wb"
    for split in ["train", "val", "test"]:
        writers[split] = {
            "X": open(SPLIT_DIR / f"X_{split}.bin", write_mode),
            "y1": open(SPLIT_DIR / f"y1_{split}.bin", write_mode),
            "y3": open(SPLIT_DIR / f"y3_{split}.bin", write_mode),
            "y5": open(SPLIT_DIR / f"y5_{split}.bin", write_mode),
        }

    # Fetch market-wide features for ALL three major indices
    # 60xxxx.SH → 上证指数 000001.SH
    # 00xxxx.SZ → 深证成指 399001.SZ
    # 30xxxx.SZ → 创业板指 399006.SZ
    INDEX_MAP = {
        "000001.SH": "上证指数",
        "399001.SZ": "深证成指",
        "399006.SZ": "创业板指",
    }
    all_market_features = {}
    for idx_code, idx_name in INDEX_MAP.items():
        log(f"Fetching market index: {idx_name} ({idx_code})...")
        try:
            safe_name = idx_code.replace(".", "_")
            idx_cache = CACHE_DIR / f"index_daily_{safe_name}.json"
            if idx_cache.exists() and (time.time() - idx_cache.stat().st_mtime < 86400 * 7):
                idx_data = json.loads(idx_cache.read_text())
            else:
                idx_df = pro.index_daily(ts_code=idx_code, start_date='20220101')
                idx_df = idx_df.sort_values('trade_date')
                idx_data = {r['trade_date']: {
                    'idx_ret': float(r['pct_chg']) / 100,
                } for _, r in idx_df.iterrows()}
                idx_cache.write_text(json.dumps(idx_data))
            all_market_features[idx_code] = idx_data
            log(f"  {idx_name}: {len(idx_data)} days")
        except Exception as e:
            log(f"  Warning: could not fetch {idx_name}: {e}")

    def get_market_for_stock(ts_code):
        """Match stock to its corresponding exchange index"""
        code = ts_code.split(".")[0]
        if code.startswith("30"):
            return all_market_features.get("399006.SZ", {})
        elif code.startswith("00"):
            return all_market_features.get("399001.SZ", {})
        else:
            return all_market_features.get("000001.SH", {})

    # ── 申万行业指数 (板块涨跌, 31 sectors) ──
    log("Fetching SW industry index data (31 sectors)...")
    sw_cache = CACHE_DIR / "sw_industry_daily.json"
    if sw_cache.exists() and (time.time() - sw_cache.stat().st_mtime < 86400 * 7):
        sw_data = json.loads(sw_cache.read_text())
    else:
        sw_data = {}  # {industry_name: {date: pct_change}}
        try:
            idx_list = pro.index_classify(level="L1", src="SW2021")
            if idx_list is not None:
                for _, row in idx_list.iterrows():
                    code = row["index_code"]
                    name = row["industry_name"]
                    try:
                        df = pro.sw_daily(ts_code=code, start_date="20220101")
                        if df is not None and not df.empty:
                            sw_data[name] = {r["trade_date"]: float(r.get("pct_change", 0) or 0) / 100
                                             for _, r in df.iterrows()}
                        # time.sleep(0.15)  # removed: API latency is natural throttle  # gentle rate limit
                    except:
                        pass
                log(f"  SW sectors: {len(sw_data)} industries loaded")
        except Exception as e:
            log(f"  Warning: could not fetch SW data: {e}")
        if sw_data:
            sw_cache.write_text(json.dumps(sw_data))

    # ── Stock → SW sector mapping (index_member based) ──
    sw_map_cache = CACHE_DIR / "sw_stock_map.json"
    if sw_map_cache.exists() and (time.time() - sw_map_cache.stat().st_mtime < 86400 * 7):
        sw_stock_map = json.loads(sw_map_cache.read_text())
    else:
        log("Building stock → SW sector mapping via index_member...")
        sw_stock_map = {}  # ts_code → industry_name
        try:
            idx_list = pro.index_classify(level="L1", src="SW2021")
            if idx_list is not None:
                for _, row in idx_list.iterrows():
                    code = row["index_code"]
                    name = row["industry_name"]
                    try:
                        members = pro.index_member(index_code=code)
                        if members is not None:
                            current = members[members["is_new"] == "Y"]
                            for _, m in current.iterrows():
                                sw_stock_map[m["con_code"]] = name
                        time.sleep(0.3)
                    except:
                        pass
        except Exception as e:
            log(f"  Warning: could not build SW mapping: {e}")
        if sw_stock_map:
            sw_map_cache.write_text(json.dumps(sw_stock_map, ensure_ascii=False))
        log(f"  Mapped {len(sw_stock_map)} stocks to SW sectors")

    # daily_info skipped: requires 1000 API calls (one per trading day), too slow

    # Check for resume checkpoint
    progress_file = SPLIT_DIR / "progress.json"
    resume_from = 0
    if progress_file.exists():
        try:
            prog = json.loads(progress_file.read_text())
            resume_from = prog["stock_index"]
            counts = prog["counts"]
            success = prog["success"]
            errors = prog["errors"]
            total_seqs = prog["total_seqs"]
            feature_dim = prog["feature_dim"]
            log(f"Resuming from stock #{resume_from} (checkpoint found)")
            log(f"  Previous: {success} ok, {errors} err, {sum(counts.values())} seqs")

            # CRITICAL: truncate .bin files to match checkpoint counts
            # Process may have written beyond checkpoint before crashing
            bytes_per_seq_X = seq_len * feature_dim * 4  # float32
            bytes_per_label = 4  # float32
            for split in ["train", "val", "test"]:
                expected_X = counts[split] * bytes_per_seq_X
                expected_y = counts[split] * bytes_per_label
                for fname, expected in [
                    (f"X_{split}.bin", expected_X),
                    (f"y1_{split}.bin", expected_y),
                    (f"y3_{split}.bin", expected_y),
                    (f"y5_{split}.bin", expected_y),
                ]:
                    fpath = SPLIT_DIR / fname
                    if fpath.exists():
                        actual = fpath.stat().st_size
                        if actual > expected:
                            log(f"  Truncating {fname}: {actual} → {expected} bytes")
                            with open(fpath, "r+b") as f:
                                f.truncate(expected)
        except Exception as e:
            log(f"  Warning: checkpoint recovery failed: {e}, starting fresh")
            resume_from = 0

    if resume_from == 0:
        success = 0
        errors = 0
        total_seqs = 0

    log(f"Starting data collection for {n} stocks (streaming to disk)...")
    api_calls = 0
    extra_api_calls = 0

    for i, stock in enumerate(stocks):
        if i < resume_from:
            continue  # skip already-processed stocks
        ts_code = stock["ts_code"]

        # Rate limit for actual API calls only
        cache_key = hashlib.md5(f"{ts_code}_20220101".encode()).hexdigest()
        cache_path = CACHE_DIR / f"daily_{cache_key}.json"
        is_cached = cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 86400 * 7)

        if not is_cached:
            api_calls += 1
            if api_calls > 1 and api_calls % 500 == 0:
                log(f"  Rate limit pause at API call #{api_calls}... sleeping 30s")
                time.sleep(30)

        prices = fetch_daily_data(ts_code)
        if len(prices) < 120:
            errors += 1
            continue

        # Fetch extra data (daily_basic + moneyflow + margin_detail = 3 API calls)
        extra_cache = CACHE_DIR / f"extra_{ts_code.replace('.', '_')}.json"
        extra_is_cached = False
        if extra_cache.exists() and (time.time() - extra_cache.stat().st_mtime < 86400 * 7):
            try:
                cached = json.loads(extra_cache.read_text())
                if cached:
                    dates = sorted(cached.keys())
                    check_date = dates[len(dates) // 2]
                    sample = cached[check_date]
                    if "big_net" in sample and "rzye" in sample:
                        extra_is_cached = True
            except:
                pass
        if not extra_is_cached:
            extra_api_calls += 3  # daily_basic + moneyflow + margin_detail
            # Tested: API response time (~0.5-1.2s each) is the bottleneck
            # Zero sleep between stocks; API latency provides natural throttle
            # Brief cooldown every 1000 calls as safety margin
            if extra_api_calls > 1 and extra_api_calls % 1000 == 0:
                log(f"  Extra API checkpoint at #{extra_api_calls}... sleeping 15s")
                time.sleep(15)
        extra_data = fetch_extra_data(ts_code)

        # Validate extra data completeness — skip stock if API failed
        if extra_data:
            sample_date = sorted(extra_data.keys())[len(extra_data) // 2]
            sample = extra_data[sample_date]
            if "big_net" not in sample or "rzye" not in sample:
                errors += 1
                continue  # Skip — incomplete extra data would poison the model
        else:
            errors += 1
            continue  # Skip — no extra data at all

        try:
            stock_market = get_market_for_stock(ts_code)
            stock_industry = stock.get("industry", "")
            # Match stock to SW sector via index_member mapping (not industry name)
            sw_sector_name = sw_stock_map.get(ts_code, "")
            sector_daily = sw_data.get(sw_sector_name, {})
            X = _feature_engineer(prices, market_data=stock_market, industry=stock_industry,
                                  extra_data=extra_data, sector_data=sector_daily)
            if X.size == 0 or len(X) < 60:
                errors += 1
                continue

            y = _create_labels(prices)
            # _build_sequences returns (X_seq, y_seq, valid_indices)
            # valid_indices[k] = the price index for sequence k's label
            # This correctly handles NaN-skipped samples
            X_seq, y_seq, valid_indices = _build_sequences(X, y, seq_len=seq_len)

            if len(X_seq) < 30:
                errors += 1
                continue

            if feature_dim is None:
                feature_dim = X_seq.shape[2]

            # Get dates using valid_indices (NOT continuous indexing!)
            # valid_indices[k] = label_idx for sequence k → prices[label_idx]["date"]
            dates = [prices[int(idx)]["date"] for idx in valid_indices]

            # Split sequences by DATE - walk-forward temporal split
            for split in ["train", "val", "test"]:
                if split == "train":
                    mask = np.array([d <= TRAIN_END for d in dates])
                elif split == "val":
                    mask = np.array([VAL_START <= d <= VAL_END for d in dates])
                else:
                    mask = np.array([d >= TEST_START for d in dates])

                if mask.sum() == 0:
                    continue

                X_split = X_seq[mask]  # already float32 from _build_sequences
                writers[split]["X"].write(X_split.tobytes())
                writers[split]["y1"].write(y_seq[1][mask].astype(np.float32).tobytes())
                writers[split]["y3"].write(y_seq[3][mask].astype(np.float32).tobytes())
                writers[split]["y5"].write(y_seq[5][mask].astype(np.float32).tobytes())
                counts[split] += int(mask.sum())

            # Flush ALL writers after EVERY stock — X and y stay in sync on disk
            for split in writers:
                for f in writers[split].values():
                    f.flush()

            total_seqs += int(len(X_seq))
            success += 1

        except Exception as e:
            errors += 1
            continue

        if (i + 1) % 100 == 0:
            # Save progress checkpoint every 100 stocks
            # Writers already flushed per-stock above, so counts == file state
            progress = {"stock_index": i + 1, "counts": counts, "success": success,
                         "errors": errors, "total_seqs": total_seqs, "feature_dim": feature_dim}
            (SPLIT_DIR / "progress.json").write_text(json.dumps(progress))

        if (i + 1) % 500 == 0:
            total = sum(counts.values())
            log(f"  Progress: {i+1}/{n} stocks, {success} ok, {errors} err, {total} seqs "
                f"(tr={counts['train']} va={counts['val']} te={counts['test']})")
            # Force garbage collection to prevent memory buildup
            import gc; gc.collect()

    # Close writers
    for split in writers:
        for f in writers[split].values():
            f.close()

    total = sum(counts.values())
    log(f"Data collection complete: {success} stocks, {total} sequences, {feature_dim} features")
    log(f"Split counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    # Convert binary files to .npy (memory-mapped readable)
    log("Converting binary files to .npy format...")
    for split in ["train", "val", "test"]:
        c = counts[split]
        if c == 0:
            continue

        # X: (count, seq_len, feature_dim)
        X_raw = np.fromfile(SPLIT_DIR / f"X_{split}.bin", dtype=np.float32)
        X_arr = X_raw.reshape(c, seq_len, feature_dim)
        np.save(SPLIT_DIR / f"X_{split}.npy", X_arr)
        del X_raw, X_arr

        # y: (count,) each
        for h, name in [(1, "y1"), (3, "y3"), (5, "y5")]:
            y_raw = np.fromfile(SPLIT_DIR / f"{name}_{split}.bin", dtype=np.float32)
            np.save(SPLIT_DIR / f"y_{split}_{h}d.npy", y_raw)
            del y_raw

        # Clean up .bin files
        for f in (SPLIT_DIR / f"X_{split}.bin", SPLIT_DIR / f"y1_{split}.bin",
                  SPLIT_DIR / f"y3_{split}.bin", SPLIT_DIR / f"y5_{split}.bin"):
            f.unlink(missing_ok=True)

        log(f"  {split}: {c} sequences saved as .npy")

    import gc; gc.collect()

    # Write flag
    flag_file.write_text(json.dumps(counts))

    # Load as memory-mapped
    result = {}
    for split in ["train", "val", "test"]:
        result[f"X_{split}"] = np.load(SPLIT_DIR / f"X_{split}.npy", mmap_mode='r')
        result[f"y_{split}"] = {h: np.load(SPLIT_DIR / f"y_{split}_{h}d.npy", mmap_mode='r') for h in [1,3,5]}

    du = sum(f.stat().st_size for f in SPLIT_DIR.iterdir() if f.suffix == '.npy')
    log(f"Data ready! Disk: {du/1e9:.1f} GB")

    return result


# ============================================================
# Phase 2: Training
# ============================================================

def train(data, max_hours=72, seed=42, model_id=0):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    log(f"Random seed: {seed}, Model ID: {model_id}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Device: {device}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_train = len(X_train)
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    log(f"Train: {n_train}, Val: {len(X_val)}, Test: {len(X_test)}, Features: {input_dim}")

    # Label smoothing (reduced - labels are now clean after threshold filtering)
    smooth = 0.03

    # Custom dataset - reads from mmap on-the-fly
    class MmapDataset(torch.utils.data.Dataset):
        def __init__(self, X_mmap, y1, y3, y5, smooth=0.03):
            self.X, self.y1, self.y3, self.y5 = X_mmap, y1, y3, y5
            self.n = len(X_mmap)
            self.smooth = smooth  # label smoothing: 0→0.03, 1→0.97
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
            # Labels are now clean 0/1 (NaN samples already dropped)
            y1 = np.clip(float(self.y1[idx]), self.smooth, 1.0 - self.smooth)
            y3 = np.clip(float(self.y3[idx]), self.smooth, 1.0 - self.smooth)
            y5 = np.clip(float(self.y5[idx]), self.smooth, 1.0 - self.smooth)
            return x, torch.tensor(y1, dtype=torch.float32), torch.tensor(y3, dtype=torch.float32), torch.tensor(y5, dtype=torch.float32)

    # Hyperparameters
    batch_size = min(512, max(64, int(math.sqrt(n_train))))
    lr = 5e-5 * (batch_size / 64)  # slightly lower LR for smaller effective batch
    wd = 1e-2
    hidden_dim = 128
    n_heads = 8
    n_layers = 4
    max_epochs = 500
    patience = 30

    # Class balance check - compute pos_weight for each horizon
    # Filter out 0.5 fill labels when computing class balance
    # 0.5 = NaN-filled unknowns, not real labels
    def real_mean(arr):
        a = np.array(arr)
        real = a[a != 0.5]
        return np.mean(real) if len(real) > 0 else 0.5
    y1_mean = real_mean(y_train[1])
    y3_mean = real_mean(y_train[3])
    y5_mean = real_mean(y_train[5])
    log(f"Class balance: 1d_up={y1_mean:.3f} 3d_up={y3_mean:.3f} 5d_up={y5_mean:.3f}")

    # If imbalanced, Focal Loss alpha should compensate
    # alpha = 1 - pos_ratio (give more weight to minority class)
    focal_alpha_1d = 1.0 - y1_mean
    focal_alpha_3d = 1.0 - y3_mean
    focal_alpha_5d = 1.0 - y5_mean
    log(f"Focal alpha: 1d={focal_alpha_1d:.3f} 3d={focal_alpha_3d:.3f} 5d={focal_alpha_5d:.3f}")
    log(f"Batch size: {batch_size}, LR: {lr:.6f}, Weight decay: {wd}")

    train_ds = MmapDataset(X_train, y_train[1], y_train[3], y_train[5], smooth=smooth)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    steps_per_epoch = len(train_loader)
    log(f"Batches/epoch: {steps_per_epoch}")

    # Model
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.3)
            self.lstm_norm = nn.LayerNorm(hidden_dim)
            self.pos_enc = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
                dropout=0.3, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.attn_pool = nn.Linear(hidden_dim, 1)
            # Shared trunk - captures common patterns across horizons
            self.shared_fc = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(0.3))
            # Separate thin heads - horizon-specific
            self.head_1d = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
            self.head_3d = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
            self.head_5d = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

        def forward(self, x):
            x = self.input_proj(x)
            h, _ = self.lstm(x)
            h = self.lstm_norm(h) + self.pos_enc
            t = self.out_norm(self.transformer(h))
            w = torch.softmax(self.attn_pool(t), dim=1)
            p = (t * w).sum(dim=1)
            shared = self.shared_fc(p)  # shared representation
            return self.head_1d(shared), self.head_3d(shared), self.head_5d(shared)

    model = HybridModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # CosineAnnealingWarmRestarts - works with early stopping (no total_steps dependency)
    # T_0=10: restart every 10 epochs, T_mult=2: period doubles each restart (10, 20, 40...)
    # This gives warm restarts that can escape local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.001)
    # Warmup: manually ramp LR for first 3 epochs
    warmup_epochs = 3

    # Asymmetric Focal Loss - penalizes false positives more than false negatives
    # In trading: predicting "up" when it goes down = you lose money
    # Predicting "down" when it goes up = you just miss a trade
    # FP_weight > FN_weight
    class AsymmetricFocalBCE(nn.Module):
        def __init__(self, gamma=1.5, alpha=0.5, fp_penalty=1.3):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.fp_penalty = fp_penalty  # extra penalty for false positives
        def forward(self, pred, target):
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
            # Mask out 0.5 fill labels (NaN-filled unknowns) - they should contribute 0 loss
            # After label smoothing: real labels are 0.03 or 0.97, fills stay at 0.5
            # Use range [0.45, 0.55] to catch smoothed fills too
            valid = (target < 0.45) | (target > 0.55)
            if valid.sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            pred = pred[valid]
            target = target[valid]
            bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
            pt = torch.where(target > 0.5, pred, 1 - pred)
            alpha_t = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
            focal = alpha_t * (1 - pt) ** self.gamma * bce
            # Extra penalty: when pred > 0.5 but target = 0 (false positive)
            fp_mask = (pred > 0.5) & (target < 0.5)
            focal = torch.where(fp_mask, focal * self.fp_penalty, focal)
            return focal.mean()

    criterion_1d = AsymmetricFocalBCE(gamma=1.5, alpha=focal_alpha_1d, fp_penalty=1.3)
    criterion_3d = AsymmetricFocalBCE(gamma=1.5, alpha=focal_alpha_3d, fp_penalty=1.3)
    criterion_5d = AsymmetricFocalBCE(gamma=1.5, alpha=focal_alpha_5d, fp_penalty=1.3)

    # Gradient accumulation - effective batch size = batch_size * accum_steps
    accum_steps = 2  # effective batch = 512*2 = 1024
    log(f"Gradient accumulation: {accum_steps} steps (effective batch={batch_size*accum_steps})")

    # SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 30
    swa_active = False

    best_val_loss = float('inf')
    best_val_acc = 0
    best_state = None
    no_improve = 0
    start_time = time.time()

    history = {"train_loss": [], "val_loss": [], "val_acc_1d": [], "val_acc_3d": [],
               "val_acc_5d": [], "learning_rate": [], "grad_norm": [], "epoch_time": []}

    suffix = f"_{model_id}" if model_id > 0 else ""
    checkpoint_path = MODEL_DIR / f"predictor_best{suffix}.pt"
    meta_path = MODEL_DIR / f"predictor_meta{suffix}.json"
    loss_curve_path = MODEL_DIR / f"loss_curve{suffix}.json"

    log(f"Training: max {max_epochs} epochs, patience {patience}, SWA@epoch {swa_start}")

    for epoch in range(max_epochs):
        elapsed = time.time() - start_time
        if elapsed > max_hours * 3600:
            log(f"Time limit ({max_hours}h)")
            break

        epoch_start = time.time()
        model.train()
        train_loss = 0
        n_batches = 0
        grad_norms = []

        optimizer.zero_grad()
        for batch_idx, (bx, by1, by3, by5) in enumerate(train_loader):
            bx = bx.to(device)
            by1, by3, by5 = by1.to(device), by3.to(device), by5.to(device)

            # Data augmentation after warmup — light touch for financial data
            # Only apply ONE augmentation per batch (not all three stacked)
            if epoch > 3 and random.random() < 0.5:  # 50% of batches get augmented
                aug_type = random.random()
                if aug_type < 0.5:
                    # Gaussian noise (reduced from 0.003 to 0.001)
                    bx = bx + torch.randn_like(bx) * 0.001
                elif aug_type < 0.8:
                    # Feature masking (reduced from 10% to 5%)
                    mask = torch.ones(bx.shape[2], device=device)
                    mask[torch.rand(bx.shape[2], device='cpu') < 0.05] = 0
                    bx = bx * mask.unsqueeze(0).unsqueeze(0)
                else:
                    # Magnitude scaling (reduced from ±5% to ±2%)
                    bx = bx * (1.0 + (random.random() - 0.5) * 0.04)

            p1, p3, p5 = model(bx)
            # Weight 1d heavier — it's the most actionable prediction
            loss = 0.5 * criterion_1d(p1.squeeze(), by1) + 0.3 * criterion_3d(p3.squeeze(), by3) + 0.2 * criterion_5d(p5.squeeze(), by5)
            # Scale loss by accumulation steps
            (loss / accum_steps).backward()

            train_loss += loss.item()
            n_batches += 1

            # Step optimizer every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norms.append(gn.item() if hasattr(gn, 'item') else float(gn))
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= max(n_batches, 1)
        avg_gn = np.mean(grad_norms) if grad_norms else 0

        # LR scheduling: warmup then cosine annealing
        if epoch < warmup_epochs:
            # Linear warmup
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step(epoch - warmup_epochs)

        # Validation (batched from mmap)
        model.eval()
        val_n = len(X_val)  # use ALL val data for honest early stopping
        val_loss_sum, val_count = 0, 0
        vp1_all, vp3_all, vp5_all = [], [], []
        vy1_all, vy3_all, vy5_all = [], [], []

        with torch.no_grad():
            for vs in range(0, val_n, 512):
                ve = min(vs + 512, val_n)
                xb = torch.FloatTensor(np.array(X_val[vs:ve])).to(device)
                # Use TRUE labels for val (no smoothing - we want honest metrics)
                y1b = torch.FloatTensor(np.array(y_val[1][vs:ve])).to(device)
                y3b = torch.FloatTensor(np.array(y_val[3][vs:ve])).to(device)
                y5b = torch.FloatTensor(np.array(y_val[5][vs:ve])).to(device)

                p1, p3, p5 = model(xb)
                vl = 0.5 * criterion_1d(p1.squeeze(), y1b) + 0.3 * criterion_3d(p3.squeeze(), y3b) + 0.2 * criterion_5d(p5.squeeze(), y5b)
                val_loss_sum += vl.item() * (ve - vs)
                val_count += (ve - vs)

                vp1_all.append(p1.squeeze().cpu())
                vp3_all.append(p3.squeeze().cpu())
                vp5_all.append(p5.squeeze().cpu())
                vy1_all.append(torch.FloatTensor(np.array(y_val[1][vs:ve])))
                vy3_all.append(torch.FloatTensor(np.array(y_val[3][vs:ve])))
                vy5_all.append(torch.FloatTensor(np.array(y_val[5][vs:ve])))

        val_loss = val_loss_sum / max(val_count, 1)
        vp1 = torch.cat(vp1_all); vp3 = torch.cat(vp3_all); vp5 = torch.cat(vp5_all)
        vy1 = torch.cat(vy1_all); vy3 = torch.cat(vy3_all); vy5 = torch.cat(vy5_all)
        # Only count accuracy on REAL labels (exclude 0.5 fill values)
        # y=0.5 are NaN-filled samples that would always count as wrong
        def masked_acc(pred, target):
            valid = (target != 0.5)  # 0.5 = NaN-filled, not a real label
            if valid.sum() == 0:
                return 0.5
            return ((pred[valid] > 0.5).float() == target[valid]).float().mean().item()
        acc1 = masked_acc(vp1, vy1)
        acc3 = masked_acc(vp3, vy3)
        acc5 = masked_acc(vp5, vy5)
        avg_acc = (acc1 + acc3 + acc5) / 3

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]['lr']
        gap = train_loss - val_loss

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc_1d"].append(round(acc1*100, 2))
        history["val_acc_3d"].append(round(acc3*100, 2))
        history["val_acc_5d"].append(round(acc5*100, 2))
        history["learning_rate"].append(round(lr_now, 8))
        history["grad_norm"].append(round(avg_gn, 4))
        history["epoch_time"].append(round(epoch_time, 2))

        if (epoch + 1) % 10 == 0:
            loss_curve_path.write_text(json.dumps(history, indent=2))

        # Best check - use val_loss ONLY for early stopping (single metric, no ambiguity)
        # best_val_acc tracks accuracy AT the best val_loss epoch (not global max)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = avg_acc  # accuracy at this epoch, not global max

        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, checkpoint_path)
            meta = {
                "epoch": epoch + 1, "val_loss": round(val_loss, 4), "train_loss": round(train_loss, 4),
                "accuracy_1d": round(acc1*100, 1), "accuracy_3d": round(acc3*100, 1), "accuracy_5d": round(acc5*100, 1),
                "avg_accuracy": round(avg_acc*100, 1), "total_params": total_params, "hidden_dim": hidden_dim,
                "n_heads": n_heads, "n_layers": n_layers, "lstm_layers": 3, "input_dim": input_dim, "seq_len": seq_len,
                "train_size": n_train, "val_size": len(X_val), "batch_size": batch_size,
                "learning_rate": lr, "elapsed_hours": round(elapsed/3600, 2), "timestamp": int(time.time()),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or improved:
            log(f"Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} gap={gap:+.4f} | "
                f"acc: 1d={acc1*100:.1f}% 3d={acc3*100:.1f}% 5d={acc5*100:.1f}% | "
                f"grad={avg_gn:.3f} lr={lr_now:.2e} | {epoch_time:.1f}s | {'★' if improved else ''}")

        # SWA
        if epoch >= swa_start:
            if not swa_active:
                log(f"SWA activated at epoch {epoch+1}")
                swa_active = True
            swa_model.update_parameters(model)

        if no_improve >= patience:
            log(f"Early stopping at epoch {epoch+1} (no improve for {patience} epochs)")
            break

    # ── Final model selection ──
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if swa_active:
        log("Comparing SWA vs best checkpoint...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_model.eval()
        # Batched SWA evaluation (full val, avoid OOM)
        swa_correct = {h: 0 for h in [1,3,5]}
        swa_valid_count = {h: 0 for h in [1,3,5]}
        with torch.no_grad():
            for vs in range(0, len(X_val), 512):
                ve = min(vs + 512, len(X_val))
                xb = torch.FloatTensor(np.array(X_val[vs:ve])).to(device)
                sp1, sp3, sp5 = swa_model(xb)
                yb = {h: torch.FloatTensor(np.array(y_val[h][vs:ve])) for h in [1,3,5]}
                for sp, h in [(sp1,1),(sp3,3),(sp5,5)]:
                    valid = (yb[h] != 0.5)
                    if valid.sum() > 0:
                        swa_correct[h] += ((sp.squeeze().cpu()[valid] > 0.5).float() == yb[h][valid]).float().sum().item()
                        swa_valid_count[h] += valid.sum().item()
        swa_avg = sum(swa_correct[h] / max(swa_valid_count[h], 1) for h in [1,3,5]) / 3
        log(f"SWA avg_acc={swa_avg*100:.1f}% vs best={best_val_acc*100:.1f}%")
        if swa_avg > best_val_acc:
            log("★ SWA wins!")
            # SWA state_dict has 'module.' prefix — strip it for compatibility
            best_state = {k.replace("module.", ""): v.cpu().clone()
                          for k, v in swa_model.state_dict().items()}
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save(best_state, checkpoint_path)
    loss_curve_path.write_text(json.dumps(history, indent=2))

    # ============================================================
    # Phase 3: TEST SET EVALUATION
    # ============================================================
    log("=" * 60)
    log("TEST SET EVALUATION (never seen during training)")
    log("=" * 60)

    model.eval()
    test_n = len(X_test)  # eval ALL test samples - no cap
    tp1_all, tp3_all, tp5_all = [], [], []

    with torch.no_grad():
        for ts in range(0, test_n, 512):
            te = min(ts + 512, test_n)
            xb = torch.FloatTensor(np.array(X_test[ts:te])).to(device)
            p1, p3, p5 = model(xb)
            tp1_all.append(p1.squeeze().cpu())
            tp3_all.append(p3.squeeze().cpu())
            tp5_all.append(p5.squeeze().cpu())

    tp1 = torch.cat(tp1_all).numpy()
    tp3 = torch.cat(tp3_all).numpy()
    tp5 = torch.cat(tp5_all).numpy()
    ty1 = np.array(y_test[1][:test_n])
    ty3 = np.array(y_test[3][:test_n])
    ty5 = np.array(y_test[5][:test_n])

    def np_masked_acc(pred, target):
        valid = (target != 0.5)
        if valid.sum() == 0:
            return 0.5
        return np.mean((pred[valid] > 0.5) == target[valid])
    test_acc1 = np_masked_acc(tp1, ty1)
    test_acc3 = np_masked_acc(tp3, ty3)
    test_acc5 = np_masked_acc(tp5, ty5)
    test_avg = (test_acc1 + test_acc3 + test_acc5) / 3

    log(f"Test Accuracy:")
    log(f"  1-day: {test_acc1*100:.1f}%")
    log(f"  3-day: {test_acc3*100:.1f}%")
    log(f"  5-day: {test_acc5*100:.1f}%")
    log(f"  Average: {test_avg*100:.1f}%")

    # Detailed analysis per horizon
    for name, preds, labels in [("1-day", tp1, ty1), ("3-day", tp3, ty3), ("5-day", tp5, ty5)]:
        pred_bin = (preds > 0.5).astype(float)
        tp = np.sum((pred_bin == 1) & (labels == 1))
        fp = np.sum((pred_bin == 1) & (labels == 0))
        tn = np.sum((pred_bin == 0) & (labels == 0))
        fn = np.sum((pred_bin == 0) & (labels == 1))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        log(f"")
        log(f"── {name} ──")
        log(f"  TP={int(tp)} FP={int(fp)} TN={int(tn)} FN={int(fn)}")
        log(f"  Precision={prec*100:.1f}% Recall={rec*100:.1f}% F1={f1*100:.1f}%")

        # Confidence buckets
        log(f"  Confidence Buckets:")
        for lo, hi in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]:
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() > 10:
                log(f"    P(up)[{lo:.2f}-{hi:.2f}]: {mask.sum()} samples, actual_up={np.mean(labels[mask])*100:.1f}%")
            mask_dn = (preds <= (1-lo)) & (preds > (1-hi))
            if mask_dn.sum() > 10:
                log(f"    P(dn)[{1-hi:.2f}-{1-lo:.2f}]: {mask_dn.sum()} samples, actual_dn={100-np.mean(labels[mask_dn])*100:.1f}%")

    # Calibration
    log(f"")
    # Calibration (exclude 0.5 fills for 3d/5d)
    log(f"Calibration: pred_up_1d={tp1.mean():.3f} actual={ty1.mean():.3f}")
    ty3_real = ty3[ty3 != 0.5]; ty5_real = ty5[ty5 != 0.5]
    log(f"Calibration: pred_up_3d={tp3[ty3 != 0.5].mean():.3f} actual={ty3_real.mean():.3f}")
    log(f"Calibration: pred_up_5d={tp5[ty5 != 0.5].mean():.3f} actual={ty5_real.mean():.3f}")

    # Overfitting check
    gap = best_val_acc * 100 - test_avg * 100
    if gap > 3:
        log(f"⚠️ Overfitting: val={best_val_acc*100:.1f}% > test={test_avg*100:.1f}% (gap={gap:.1f}%)")
    else:
        log(f"✅ No overfitting: val={best_val_acc*100:.1f}% ≈ test={test_avg*100:.1f}% (gap={gap:.1f}%)")

    # Profitability
    log(f"")
    log(f"── Profitability (1-day, threshold=0.55) ──")
    long_mask = tp1 > 0.55
    if long_mask.sum() > 0:
        wr = np.mean(ty1[long_mask])
        log(f"  Trades: {long_mask.sum()}/{len(tp1)} ({long_mask.sum()/len(tp1)*100:.1f}%)")
        log(f"  Win rate: {wr*100:.1f}% vs baseline: {ty1.mean()*100:.1f}%")
        log(f"  Edge: {(wr - ty1.mean())*100:+.1f}%")

    # Update meta
    meta = json.loads(meta_path.read_text())
    meta["test_accuracy_1d"] = round(test_acc1*100, 1)
    meta["test_accuracy_3d"] = round(test_acc3*100, 1)
    meta["test_accuracy_5d"] = round(test_acc5*100, 1)
    meta["test_avg_accuracy"] = round(test_avg*100, 1)
    meta["overfit_gap"] = round(gap, 1)
    meta_path.write_text(json.dumps(meta, indent=2))

    total_time = time.time() - start_time
    log(f"Model {model_id} total time: {total_time/3600:.1f}h")
    log(f"Model {model_id} done!")
    return str(meta_path)


def train_ensemble(data, n_models=3, max_hours_per_model=24):
    """Train multiple models with different seeds, save all for ensemble inference"""
    log("=" * 60)
    log(f"ENSEMBLE TRAINING: {n_models} models")
    log("=" * 60)

    all_results = []
    for i in range(n_models):
        log(f"\n{'='*60}")
        log(f"Model {i+1}/{n_models} (seed={42+i})")
        log(f"{'='*60}")

        # Each model gets a different seed → different initialization + shuffling
        result_meta = train(data, max_hours=max_hours_per_model, seed=42+i, model_id=i)
        all_results.append(result_meta)

    # Summary
    log("\n" + "=" * 60)
    log("ENSEMBLE SUMMARY")
    log("=" * 60)
    for i, r in enumerate(all_results):
        if r and Path(r).exists():
            meta = json.loads(Path(r).read_text())
            log(f"Model {i}: test_avg={meta.get('test_avg_accuracy','?')}% "
                f"val_loss={meta.get('val_loss','?')}")


if __name__ == "__main__":
    log("=" * 60)
    log("Stock Price Predictor - Large-Scale Training V2")
    log("=" * 60)
    data = collect_data()
    # Train 3 models for ensemble (8h each max = 24h total max)
    train_ensemble(data, n_models=3, max_hours_per_model=20)

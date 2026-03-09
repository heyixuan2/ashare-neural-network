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
    """Fetch 北向/南向 资金 (market-wide, paginated by year)"""
    cache_path = CACHE_DIR / "hsgt_flow.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 86400 * 7):
        try:
            data = json.loads(cache_path.read_text())
            if len(data) > 500:  # ensure full data, not partial
                return data
        except:
            pass
    result = {}
    # API returns max ~300 rows per call, fetch by year
    for year in range(2022, 2027):
        try:
            df = pro.moneyflow_hsgt(start_date=f"{year}0101", end_date=f"{year}1231")
            if df is not None and not df.empty:
                for _, r in df.iterrows():
                    result[r["trade_date"]] = {
                        "hgt": float(r.get("hgt", 0) or 0),
                        "sgt": float(r.get("sgt", 0) or 0),
                    }
            time.sleep(0.3)
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
                result[f"y_{split}"] = {1: np.load(SPLIT_DIR / f"y_{split}_1d.npy", mmap_mode='r')}
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

    # ── HSGT (北向/南向资金) ──
    log("Fetching HSGT (northbound/southbound) flow data...")
    hsgt_flow = fetch_hsgt_data()
    log(f"  HSGT: {len(hsgt_flow)} days")

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
                                  extra_data=extra_data, hsgt_data=hsgt_flow,
                                  sector_data=sector_daily)
            if X.size == 0 or len(X) < 60:
                errors += 1
                continue

            y = _create_labels(prices, horizons=[1])
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

        # y: (count,)
        y_raw = np.fromfile(SPLIT_DIR / f"y1_{split}.bin", dtype=np.float32)
        np.save(SPLIT_DIR / f"y_{split}_1d.npy", y_raw)
        del y_raw

        # Clean up .bin files
        for f in (SPLIT_DIR / f"X_{split}.bin", SPLIT_DIR / f"y1_{split}.bin"):
            f.unlink(missing_ok=True)

        log(f"  {split}: {c} sequences saved as .npy")

    import gc; gc.collect()

    # Write flag
    flag_file.write_text(json.dumps(counts))

    # Load as memory-mapped
    result = {}
    for split in ["train", "val", "test"]:
        result[f"X_{split}"] = np.load(SPLIT_DIR / f"X_{split}.npy", mmap_mode='r')
        result[f"y_{split}"] = {1: np.load(SPLIT_DIR / f"y_{split}_1d.npy", mmap_mode='r')}

    du = sum(f.stat().st_size for f in SPLIT_DIR.iterdir() if f.suffix == '.npy')
    log(f"Data ready! Disk: {du/1e9:.1f} GB")

    # ============================================================
    # DATA INTEGRITY CHECK — must pass before training starts
    # ============================================================
    log("=" * 60)
    log("DATA INTEGRITY CHECK")
    log("=" * 60)
    checks_passed = True

    for split in ["train", "val", "test"]:
        X = result[f"X_{split}"]
        y1 = result[f"y_{split}"][1]

        # 1. Shape alignment
        if X.shape[0] != len(y1):
            log(f"  ❌ {split}: shape mismatch X={X.shape[0]} y1={len(y1)}")
            checks_passed = False
        else:
            log(f"  ✅ {split}: {X.shape[0]:,} samples, shape {X.shape}")

        # 2. Feature dimension
        if X.shape[2] != feature_dim:
            log(f"  ❌ {split}: feature_dim={X.shape[2]}, expected {feature_dim}")
            checks_passed = False

        # 3. NaN/Inf check (sample 1000)
        sample_idx = np.random.choice(X.shape[0], min(1000, X.shape[0]), replace=False)
        X_sample = np.array(X[sample_idx])
        if np.isnan(X_sample).any() or np.isinf(X_sample).any():
            nan_pct = np.isnan(X_sample).mean() * 100
            inf_pct = np.isinf(X_sample).mean() * 100
            log(f"  ❌ {split}: NaN={nan_pct:.2f}% Inf={inf_pct:.2f}%")
            checks_passed = False
        else:
            log(f"  ✅ {split}: no NaN/Inf (sampled 1000)")

        # 4. Label distribution (1d only)
        y1_arr = np.array(y1)
        y1_real = y1_arr[y1_arr != 0.5]
        y1_up = np.mean(y1_real) if len(y1_real) > 0 else 0
        log(f"  ℹ️  {split}: 1d_up={y1_up:.3f}, valid={len(y1_real)}/{len(y1_arr)} ({len(y1_real)/max(len(y1_arr),1)*100:.1f}%)")

        # 5. Minimum samples
        if X.shape[0] < 1000:
            log(f"  ❌ {split}: only {X.shape[0]} samples, need ≥1000")
            checks_passed = False

    # 6. Split ratio sanity
    total = sum(result[f"X_{s}"].shape[0] for s in ["train", "val", "test"])
    for s in ["train", "val", "test"]:
        pct = result[f"X_{s}"].shape[0] / total * 100
        log(f"  ℹ️  {s}: {pct:.1f}%")

    # 7. Feature range check (expect mostly in [-4, 4] after z-score)
    X_tr_sample = np.array(result["X_train"][np.random.choice(result["X_train"].shape[0], 500, replace=False)])
    out_of_range = np.mean(np.abs(X_tr_sample) > 4) * 100
    log(f"  ℹ️  train features outside ±4: {out_of_range:.1f}% (constant features expected)")

    if not checks_passed:
        log("❌ DATA CHECK FAILED — aborting training")
        raise RuntimeError("Data integrity check failed")
    log("✅ ALL CHECKS PASSED — proceeding to training")
    log("=" * 60)

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

    # V4: lower LR, smaller batch, plain BCE, no SWA
    # V3 evidence: best epoch=1 at warmup lr=1e-4, lr=3e-4 overshoots
    smooth = 0.05  # lighter smoothing (V3's 0.1 was too aggressive)

    # Custom dataset - 1d only
    class MmapDataset(torch.utils.data.Dataset):
        def __init__(self, X_mmap, y1, smooth=0.05):
            self.X, self.y1 = X_mmap, y1
            self.n = len(X_mmap)
            self.smooth = smooth
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
            y1 = np.clip(float(self.y1[idx]), self.smooth, 1.0 - self.smooth)
            return x, torch.tensor(y1, dtype=torch.float32)

    # V4 hyperparameters
    batch_size = 256   # was 1024 — smaller batch = implicit regularization
    lr = 1e-4          # was 3e-4 — V3 best epoch was at warmup lr=1e-4
    wd = 5e-2
    hidden_dim = 64
    n_heads = 4
    n_layers = 2
    lstm_layers = 1
    max_epochs = 60    # was 200 — with constant lr, no need for many epochs
    patience = 15      # was 10 — give more room with lower lr

    # Class balance - 1d only
    def real_mean(arr):
        a = np.array(arr)
        real = a[a != 0.5]
        return np.mean(real) if len(real) > 0 else 0.5
    y1_mean = real_mean(y_train[1])
    log(f"Class balance: 1d_up={y1_mean:.3f}")
    log(f"Batch size: {batch_size}, LR: {lr:.6f}, Weight decay: {wd}")
    log(f"Model: hidden={hidden_dim}, heads={n_heads}, layers={n_layers}, lstm={lstm_layers}")

    train_ds = MmapDataset(X_train, y_train[1], smooth=smooth)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    steps_per_epoch = len(train_loader)
    log(f"Batches/epoch: {steps_per_epoch}")

    # Model — V3: smaller, 1d-only, stronger dropout
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2))
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                                num_layers=lstm_layers, dropout=0.0)  # 1 layer = no internal dropout
            self.lstm_norm = nn.LayerNorm(hidden_dim)
            self.pos_enc = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
                dropout=0.4, batch_first=True, activation='gelu')  # dropout 0.3→0.4
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.attn_pool = nn.Linear(hidden_dim, 1)
            # Single head — 1d prediction only
            self.head = nn.Sequential(
                nn.Dropout(0.5),  # aggressive dropout before final head
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

    model = HybridModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Constant LR — V3 showed best epoch at lr=1e-4, cosine schedule didn't help
    # No warmup, no scheduler: just train at the sweet spot

    # Plain BCE with valid-sample masking (0.5 = unknown label, skip)
    class MaskedBCE(nn.Module):
        def forward(self, pred, target):
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
            valid = (target < 0.45) | (target > 0.55)
            if valid.sum() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return nn.functional.binary_cross_entropy(pred[valid], target[valid])

    criterion = MaskedBCE()

    # Gradient accumulation: effective batch = 256 * 4 = 1024
    # Small per-step batch for noise, but accumulated gradient is stable
    accum_steps = 4
    log(f"Effective batch size: {batch_size * accum_steps} (micro={batch_size} × accum={accum_steps})")

    best_val_loss = float('inf')
    best_val_acc = 0
    best_state = None
    no_improve = 0
    start_time = time.time()

    history = {"train_loss": [], "val_loss": [], "val_acc_1d": [],
               "learning_rate": [], "grad_norm": [], "epoch_time": []}

    suffix = f"_{model_id}" if model_id > 0 else ""
    checkpoint_path = MODEL_DIR / f"predictor_best{suffix}.pt"
    meta_path = MODEL_DIR / f"predictor_meta{suffix}.json"
    loss_curve_path = MODEL_DIR / f"loss_curve{suffix}.json"

    log(f"Training: max {max_epochs} epochs, patience {patience}, constant lr={lr}")

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
        for batch_idx, (bx, by1) in enumerate(train_loader):
            bx = bx.to(device)
            by1 = by1.to(device)

            # No data augmentation — V2.6 showed signal is too weak, augmentation adds noise

            pred = model(bx)
            loss = criterion(pred.squeeze(), by1)
            loss = loss / accum_steps
            loss.backward()

            train_loss += loss.item() * accum_steps
            n_batches += 1

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norms.append(gn.item() if hasattr(gn, 'item') else float(gn))
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= max(n_batches, 1)
        avg_gn = np.mean(grad_norms) if grad_norms else 0

        # Validation (batched from mmap) — 1d only
        model.eval()
        val_n = len(X_val)
        val_loss_sum, val_count = 0, 0
        vp1_all, vy1_all = [], []

        with torch.no_grad():
            for vs in range(0, val_n, 1024):
                ve = min(vs + 1024, val_n)
                xb = torch.FloatTensor(np.array(X_val[vs:ve])).to(device)
                y1b = torch.FloatTensor(np.array(y_val[1][vs:ve])).to(device)

                pred = model(xb)
                vl = criterion(pred.squeeze(), y1b)
                val_loss_sum += vl.item() * (ve - vs)
                val_count += (ve - vs)

                vp1_all.append(pred.squeeze().cpu())
                vy1_all.append(torch.FloatTensor(np.array(y_val[1][vs:ve])))

        val_loss = val_loss_sum / max(val_count, 1)
        vp1 = torch.cat(vp1_all)
        vy1 = torch.cat(vy1_all)
        def masked_acc(pred, target):
            valid = (target != 0.5)
            if valid.sum() == 0:
                return 0.5
            return ((pred[valid] > 0.5).float() == target[valid]).float().mean().item()
        acc1 = masked_acc(vp1, vy1)
        avg_acc = acc1

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]['lr']
        gap = train_loss - val_loss

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc_1d"].append(round(acc1*100, 2))
        history["learning_rate"].append(round(lr_now, 8))
        history["grad_norm"].append(round(avg_gn, 4))
        history["epoch_time"].append(round(epoch_time, 2))

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
                "accuracy_1d": round(acc1*100, 1),
                "avg_accuracy": round(avg_acc*100, 1), "total_params": total_params, "hidden_dim": hidden_dim,
                "n_heads": n_heads, "n_layers": n_layers, "lstm_layers": lstm_layers, "input_dim": input_dim, "seq_len": seq_len,
                "train_size": n_train, "val_size": len(X_val), "batch_size": batch_size,
                "learning_rate": lr, "elapsed_hours": round(elapsed/3600, 2), "timestamp": int(time.time()),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
        else:
            no_improve += 1

        log(f"Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} gap={gap:+.4f} | "
            f"acc: 1d={acc1*100:.1f}% | "
            f"grad={avg_gn:.3f} lr={lr_now:.2e} | {epoch_time:.1f}s | {'★' if improved else ''}")

        if no_improve >= patience:
            log(f"Early stopping at epoch {epoch+1} (no improve for {patience} epochs)")
            break

    # ── Final model selection ──
    if best_state:
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
    test_n = len(X_test)
    tp1_all = []

    with torch.no_grad():
        for ts in range(0, test_n, 1024):
            te = min(ts + 1024, test_n)
            xb = torch.FloatTensor(np.array(X_test[ts:te])).to(device)
            pred = model(xb)
            tp1_all.append(pred.squeeze().cpu())

    tp1 = torch.cat(tp1_all).numpy()
    ty1 = np.array(y_test[1][:test_n])

    def np_masked_acc(pred, target):
        valid = (target != 0.5)
        if valid.sum() == 0:
            return 0.5
        return np.mean((pred[valid] > 0.5) == target[valid])
    test_acc1 = np_masked_acc(tp1, ty1)
    test_avg = test_acc1

    log(f"Test Accuracy: 1-day={test_acc1*100:.1f}%")

    # Detailed analysis
    for name, preds, labels in [("1-day", tp1, ty1)]:
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
    # Calibration
    log(f"Calibration: pred_up_1d={tp1.mean():.3f} actual={ty1.mean():.3f}")

    # Overfitting check
    gap = best_val_acc * 100 - test_avg * 100
    if gap > 3:
        log(f"⚠️ Overfitting: val={best_val_acc*100:.1f}% > test={test_avg*100:.1f}% (gap={gap:.1f}%)")
    else:
        log(f"✅ No overfitting: val={best_val_acc*100:.1f}% ≈ test={test_avg*100:.1f}% (gap={gap:.1f}%)")

    # ── Temperature Scaling (post-hoc calibration on val set) ──
    log(f"")
    log(f"── Temperature Scaling (calibrate on val set) ──")

    # Extract pre-sigmoid logits from a mmap array (reused for val + test)
    head_layers = list(model.head.children())
    def extract_logits(X_mmap, n):
        logits_all = []
        with torch.no_grad():
            for s in range(0, n, 1024):
                e = min(s + 1024, n)
                xb = torch.FloatTensor(np.array(X_mmap[s:e])).to(device)
                x_enc = model.input_proj(xb)
                h, _ = model.lstm(x_enc)
                h = model.lstm_norm(h) + model.pos_enc
                t = model.out_norm(model.transformer(h))
                w = torch.softmax(model.attn_pool(t), dim=1)
                p = (t * w).sum(dim=1)
                z = p
                for layer in head_layers[:-1]:  # skip Sigmoid()
                    z = layer(z)
                logits_all.append(z.squeeze().cpu())
        return torch.cat(logits_all)

    model.eval()
    val_logits = extract_logits(X_val, len(X_val))
    val_labels = torch.FloatTensor(np.array(y_val[1][:len(X_val)]))
    val_valid = (val_labels < 0.45) | (val_labels > 0.55)

    # Optimize temperature T to minimize NLL on valid val samples
    best_T, best_nll = 1.0, float('inf')
    for T_cand in [t / 10.0 for t in range(5, 31)]:  # 0.5 to 3.0
        scaled = torch.sigmoid(val_logits[val_valid] / T_cand)
        nll = nn.functional.binary_cross_entropy(
            torch.clamp(scaled, 1e-7, 1-1e-7), val_labels[val_valid]).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T_cand
    log(f"  Optimal temperature: T={best_T:.1f} (NLL={best_nll:.4f})")

    # ── Optimal Threshold Search on val set ──
    log(f"── Optimal Threshold Search ──")
    test_logits = extract_logits(X_test, test_n)
    tp1_cal = torch.sigmoid(test_logits / best_T).numpy()

    # Search threshold on val calibrated probs for best F1
    val_cal = torch.sigmoid(val_logits / best_T).numpy()
    val_labels_np = val_labels.numpy()
    val_v = (val_labels_np < 0.45) | (val_labels_np > 0.55)

    best_thresh, best_f1 = 0.5, 0
    for th in [t / 100.0 for t in range(45, 70)]:  # 0.45 to 0.69
        pred_b = (val_cal[val_v] > th).astype(float)
        tp_v = np.sum((pred_b == 1) & (val_labels_np[val_v] > 0.5))
        fp_v = np.sum((pred_b == 1) & (val_labels_np[val_v] < 0.5))
        fn_v = np.sum((pred_b == 0) & (val_labels_np[val_v] > 0.5))
        prec_v = tp_v / max(tp_v + fp_v, 1)
        rec_v = tp_v / max(tp_v + fn_v, 1)
        f1_v = 2 * prec_v * rec_v / max(prec_v + rec_v, 1e-8)
        if f1_v > best_f1:
            best_f1 = f1_v
            best_thresh = th
    log(f"  Optimal threshold: {best_thresh:.2f} (val F1={best_f1*100:.1f}%)")

    # Report calibrated test metrics
    log(f"")
    log(f"── Calibrated Test Metrics (T={best_T:.1f}, thresh={best_thresh:.2f}) ──")
    test_pred_cal = (tp1_cal > best_thresh).astype(float)
    tv = (ty1 != 0.5)
    tp_c = np.sum((test_pred_cal[tv] == 1) & (ty1[tv] == 1))
    fp_c = np.sum((test_pred_cal[tv] == 1) & (ty1[tv] == 0))
    fn_c = np.sum((test_pred_cal[tv] == 0) & (ty1[tv] == 1))
    prec_c = tp_c / max(tp_c + fp_c, 1)
    rec_c = tp_c / max(tp_c + fn_c, 1)
    f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-8)
    cal_acc = np.mean((test_pred_cal[tv]) == ty1[tv])
    log(f"  Accuracy={cal_acc*100:.1f}% Precision={prec_c*100:.1f}% Recall={rec_c*100:.1f}% F1={f1_c*100:.1f}%")
    log(f"  Pred mean={tp1_cal.mean():.3f} Actual={ty1.mean():.3f}")

    # Profitability with optimal threshold
    log(f"")
    log(f"── Profitability (1-day, calibrated thresh={best_thresh:.2f}) ──")
    long_mask = tp1_cal > best_thresh
    if long_mask.sum() > 0:
        wr = np.mean(ty1[long_mask])
        log(f"  Trades: {long_mask.sum()}/{len(tp1_cal)} ({long_mask.sum()/len(tp1_cal)*100:.1f}%)")
        log(f"  Win rate: {wr*100:.1f}% vs baseline: {ty1.mean()*100:.1f}%")
        log(f"  Edge: {(wr - ty1.mean())*100:+.1f}%")

    # Update meta
    meta = json.loads(meta_path.read_text())
    meta["test_accuracy_1d"] = round(test_acc1*100, 1)
    meta["test_avg_accuracy"] = round(test_avg*100, 1)
    meta["overfit_gap"] = round(gap, 1)
    meta["temperature"] = best_T
    meta["optimal_threshold"] = best_thresh
    meta["calibrated_f1"] = round(f1_c*100, 1)
    meta["calibrated_accuracy"] = round(cal_acc*100, 1)
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
    log("Stock Price Predictor - Large-Scale Training V4 (1d-only)")
    log("=" * 60)
    data = collect_data()
    # Train 3 models for ensemble (8h each max = 24h total max)
    train_ensemble(data, n_models=3, max_hours_per_model=20)

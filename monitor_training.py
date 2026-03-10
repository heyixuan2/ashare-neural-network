"""
训练监控脚本 — 实时查看数据收集 + 训练进度
用法: python monitor_training.py
"""
import json, time, os, sys, glob, re, random
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
LOG = BASE / "models" / "training_log.jsonl"
OUTPUT_LOG = BASE / "models" / "training_output.log"
CACHE_DIR = BASE / ".cache"
DATA_DIR = BASE / "data" / "ashare_daily"
SPLIT_DIR = BASE / "models" / "splits"
MODEL_DIR = BASE / "models"
PROGRESS_JSON = SPLIT_DIR / "progress.json"
SEQ_LEN = 30


def get_feature_dim():
    if PROGRESS_JSON.exists():
        try:
            return json.loads(PROGRESS_JSON.read_text()).get("feature_dim", 51)
        except:
            pass
    return 51


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def progress_bar(current, total, width=40):
    ratio = min(1.0, current / max(total, 1))
    filled = int(width * ratio)
    bar = "\033[32m" + "█" * filled + "\033[90m" + "░" * (width - filled) + "\033[0m"
    return f"  {bar} {ratio*100:.1f}%"


def bar(value, width=25, lo=45, hi=65):
    ratio = max(0, min(1, (value - lo) / (hi - lo)))
    filled = int(width * ratio)
    return "█" * filled + "░" * (width - filled)


def spark(values, width=40):
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    chars = " ▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    return "".join(chars[min(8, int((v - mn) / rng * 8))] for v in sampled)


def fmt_size(bytes_val):
    if bytes_val >= 1e9:
        return f"{bytes_val/1e9:.1f}GB"
    return f"{bytes_val/1e6:.1f}MB"


# ════════════════════════════════════════════════════════════════
# Data Collection Monitor
# ════════════════════════════════════════════════════════════════

def get_collection_status():
    feature_dim = get_feature_dim()
    info = {
        "total_stocks": 5808,
        "feature_dim": feature_dim,
        "phase": "unknown",
        "extra_caches": 0,
        "extra_complete": 0,
        "extra_margin_real": 0,
        "ohlcv_files": 0,
        "ohlcv_cache_files": 0,
        "rate_limit_pauses": 0,
        "last_lines": [],
        "errors": [],
        "started": None,
        "current_stock": None,
        "progress": None,
        "split_counts": {},
        "split_sizes_mb": {},
        "recent_rate_spm": None,
        "last_progress_ts": None,
        "last_progress_line": None,
        "run_restarts": 0,
        "hsgt_status": None,
        "sw_status": None,
        "sw_map_status": None,
        "index_status": {},
        "data_quality": {},
    }

    if not OUTPUT_LOG.exists():
        return info

    lines = OUTPUT_LOG.read_text().splitlines()
    info["last_lines"] = lines[-15:] if lines else []

    progress_points = []
    for line in lines:
        if "Stock Price Predictor - Large-Scale Training" in line:
            info["run_restarts"] += 1
        if "Starting data collection" in line:
            info["phase"] = "collecting"
            try:
                info["started"] = line.split("]")[0].strip("[")
            except:
                pass
        if "Rate limit pause" in line or "Extra API rate limit" in line:
            info["rate_limit_pauses"] += 1
        if "margin_detail" in line and "failed" in line:
            info["errors"].append(line.strip())
        if "Building sequences" in line or "Streaming" in line:
            info["phase"] = "building"
        if "Training model" in line or "Epoch" in line:
            info["phase"] = "training"
        if "Data collection complete" in line:
            info["phase"] = "converting"
        if "Data ready" in line:
            info["phase"] = "ready"
        if "Progress:" in line and "stocks," in line and "seqs" in line:
            info["last_progress_line"] = line.strip()
            try:
                ts_str = line.split("]")[0].strip("[")
                hh, mm, ss = [int(x) for x in ts_str.split(":")]
                mins = hh * 60 + mm + ss / 60.0
                left = line.split("Progress:")[1].strip()
                stock_part = left.split("stocks,")[0]
                current_stock = int(stock_part.split("/")[0])
                progress_points.append((mins, current_stock))
                info["last_progress_ts"] = ts_str
                info["current_stock"] = current_stock
            except:
                pass

    # ── Cache file counts ──
    extra_files = glob.glob(str(CACHE_DIR / "extra_*.json"))
    info["extra_caches"] = len(extra_files)
    info["ohlcv_cache_files"] = len(glob.glob(str(CACHE_DIR / "daily_*.json")))

    # ── OHLCV permanent data ──
    info["ohlcv_files"] = len([f for f in glob.glob(str(DATA_DIR / "*.json"))
                               if re.match(r'[036]\d{5}_', os.path.basename(f))])

    # ── Market-wide data sources ──
    for idx_code in ["000001_SH", "399001_SZ", "399006_SZ"]:
        p = CACHE_DIR / f"index_daily_{idx_code}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                info["index_status"][idx_code] = len(d)
            except:
                info["index_status"][idx_code] = -1

    hsgt_path = CACHE_DIR / "hsgt_flow.json"
    if hsgt_path.exists():
        try:
            d = json.loads(hsgt_path.read_text())
            info["hsgt_status"] = len(d)
        except:
            info["hsgt_status"] = -1

    sw_path = CACHE_DIR / "sw_industry_daily.json"
    if sw_path.exists():
        try:
            d = json.loads(sw_path.read_text())
            info["sw_status"] = len(d)
        except:
            info["sw_status"] = -1

    sw_map_path = CACHE_DIR / "sw_stock_map.json"
    if sw_map_path.exists():
        try:
            d = json.loads(sw_map_path.read_text())
            info["sw_map_status"] = len(d)
        except:
            info["sw_map_status"] = -1

    # ── Checkpoint progress ──
    if PROGRESS_JSON.exists():
        try:
            info["progress"] = json.loads(PROGRESS_JSON.read_text())
        except:
            pass

    # ── Split file sizes / inferred counts ──
    seq_bytes = SEQ_LEN * feature_dim * 4
    for split_name in ["train", "val", "test"]:
        x_path = SPLIT_DIR / f"X_{split_name}.bin"
        y_path = SPLIT_DIR / f"y1_{split_name}.bin"
        npy_path = SPLIT_DIR / f"X_{split_name}.npy"
        if npy_path.exists():
            info["split_sizes_mb"][f"X_{split_name}"] = npy_path.stat().st_size / 1024 / 1024
            info[f"npy_{split_name}"] = True
        elif x_path.exists():
            info["split_sizes_mb"][f"X_{split_name}"] = x_path.stat().st_size / 1024 / 1024
            info["split_counts"][f"X_{split_name}"] = x_path.stat().st_size // seq_bytes
        if y_path.exists():
            info["split_sizes_mb"][f"y1_{split_name}"] = y_path.stat().st_size / 1024 / 1024
            info["split_counts"][f"y1_{split_name}"] = y_path.stat().st_size // 4

    # ── Extra data quality sample ──
    sample = random.sample(extra_files, min(50, len(extra_files))) if extra_files else []
    for f in sample:
        try:
            d = json.loads(open(f).read())
            if not d:
                continue
            dates = sorted(d.keys())
            mid = d[dates[len(dates) // 2]]
            if "big_net" in mid and "rzye" in mid:
                info["extra_complete"] += 1
                if any(d[dt].get("rzye", 0) > 0 for dt in list(d.keys())[:50]):
                    info["extra_margin_real"] += 1
        except:
            pass
    if sample:
        ratio = len(extra_files) / len(sample)
        info["extra_complete"] = int(info["extra_complete"] * ratio)
        info["extra_margin_real"] = int(info["extra_margin_real"] * ratio)

    # ── OHLCV data quality spot check ──
    ohlcv_files_list = glob.glob(str(DATA_DIR / "*.json"))
    ohlcv_sample = random.sample(ohlcv_files_list, min(5, len(ohlcv_files_list))) if ohlcv_files_list else []
    max_jump = 0
    max_jump_file = ""
    adj_ok = 0
    for f in ohlcv_sample:
        try:
            records = json.loads(open(f).read())
            if len(records) < 10:
                continue
            for i in range(1, len(records)):
                c_prev = records[i - 1]["close"]
                c_now = records[i]["close"]
                if c_prev > 0:
                    jump = abs(c_now - c_prev) / c_prev
                    if jump > max_jump:
                        max_jump = jump
                        max_jump_file = os.path.basename(f)
            adj_ok += 1
        except:
            pass
    info["data_quality"] = {
        "sampled": len(ohlcv_sample),
        "ok": adj_ok,
        "max_jump_pct": max_jump * 100,
        "max_jump_file": max_jump_file,
        "adj_clean": max_jump < 0.22,
    }

    # ── Speed from progress lines ──
    if len(progress_points) >= 2:
        a = progress_points[-2]
        b = progress_points[-1]
        dt = max(b[0] - a[0], 1e-6)
        ds = b[1] - a[1]
        info["recent_rate_spm"] = ds / dt

    return info


def draw_collection(info):
    clear()
    fdim = info["feature_dim"]

    print("\033[1;32m╔══════════════════════════════════════════════════════════════════╗")
    print(f"║       LSTM-Transformer V4 训练监控  ({fdim}维特征, 1d-only)     ║")
    print("╚══════════════════════════════════════════════════════════════════╝\033[0m")
    print()

    phase_labels = {
        "collecting": "📡 数据收集中",
        "building": "🔧 构建序列中",
        "converting": "💾 转换 .npy 中",
        "ready": "✅ 数据就绪",
        "training": "🏋️ 训练中",
        "unknown": "⏳ 启动中",
    }
    print(f"  \033[1m阶段: {phase_labels.get(info['phase'], info['phase'])}\033[0m")
    if info["started"]:
        print(f"  \033[90m开始时间: {info['started']}\033[0m")
    if info.get("run_restarts", 0) > 1:
        print(f"  \033[90m运行轮次: {info['run_restarts']} 次启动\033[0m")
    print()

    # ── Market-wide data sources ──
    print(f"  \033[1m🌐 全局数据源:\033[0m")
    for code, name in [("000001_SH", "上证指数"), ("399001_SZ", "深证成指"), ("399006_SZ", "创业板指")]:
        days = info["index_status"].get(code)
        if days and days > 0:
            print(f"    {name}: \033[32m{days} 天\033[0m")
        elif days == -1:
            print(f"    {name}: \033[31m加载失败\033[0m")
        else:
            print(f"    {name}: \033[90m未获取\033[0m")

    hsgt = info["hsgt_status"]
    if hsgt and hsgt > 0:
        print(f"    北向资金 (HSGT): \033[32m{hsgt} 天\033[0m")
    elif hsgt == -1:
        print(f"    北向资金 (HSGT): \033[31m加载失败\033[0m")
    else:
        print(f"    北向资金 (HSGT): \033[90m未获取\033[0m")

    sw = info["sw_status"]
    sw_map = info["sw_map_status"]
    if sw and sw > 0:
        print(f"    申万行业指数: \033[32m{sw} 个行业\033[0m", end="")
        if sw_map and sw_map > 0:
            print(f"  映射 {sw_map} 只股票")
        else:
            print()
    else:
        print(f"    申万行业指数: \033[90m未获取\033[0m")
    print()

    # ── OHLCV data ──
    ohlcv_pct = info["ohlcv_files"] / max(info["total_stocks"], 1) * 100
    print(f"  \033[1m📊 OHLCV 数据 (前复权):\033[0m")
    print(f"    永久文件: {info['ohlcv_files']:,} / {info['total_stocks']:,} ({ohlcv_pct:.0f}%)")
    print(f"    缓存文件: {info['ohlcv_cache_files']:,}")
    print(f"    {progress_bar(info['ohlcv_files'], info['total_stocks'], 40)}")
    dq = info.get("data_quality", {})
    if dq.get("sampled", 0) > 0:
        adj_icon = "\033[32m✓\033[0m" if dq["adj_clean"] else "\033[31m✗\033[0m"
        print(f"    复权抽查: {adj_icon} {dq['ok']}/{dq['sampled']} 正常, 最大日跳变={dq['max_jump_pct']:.1f}%", end="")
        if not dq["adj_clean"]:
            print(f" \033[31m({dq['max_jump_file']})\033[0m")
        else:
            print()
    print()

    # ── Extra API data ──
    extra_pct = info["extra_caches"] / max(info["total_stocks"], 1) * 100
    print(f"  \033[1m📡 Extra API (basic + moneyflow + margin):\033[0m")
    print(f"    缓存数: {info['extra_caches']:,} / {info['total_stocks']:,} ({extra_pct:.0f}%)")
    print(f"    {progress_bar(info['extra_caches'], info['total_stocks'], 40)}")
    print(f"    完整率: ~{info['extra_complete']:,} / {info['extra_caches']:,} "
          f"({info['extra_complete'] / max(info['extra_caches'], 1) * 100:.0f}%)")
    print(f"    两融标的: ~{info['extra_margin_real']:,}")
    print(f"    Rate limit 暂停: {info['rate_limit_pauses']} 次")
    if info["errors"]:
        print(f"    \033[31mMargin 错误: {len(info['errors'])} 次\033[0m")
    print()

    # ── Split status ──
    print(f"  \033[1m💾 Split 写盘状态:\033[0m")
    prog = info.get("progress") or {}
    if prog:
        counts = prog.get("counts", {})
        total_seqs = sum(counts.values())
        print(f"    Checkpoint: stock #{prog.get('stock_index', '?'):,} / {info['total_stocks']:,} "
              f"({prog.get('stock_index', 0) / max(info['total_stocks'], 1) * 100:.1f}%)")
        print(f"    Feature dim: {prog.get('feature_dim', '?')}")
        print(f"    序列总数: {total_seqs:,}  (成功={prog.get('success', '?'):,}  失败={prog.get('errors', '?')})")
        print(f"    分割:  train={counts.get('train', 0):,}  val={counts.get('val', 0):,}  test={counts.get('test', 0):,}")
        if total_seqs > 0:
            tr_pct = counts.get("train", 0) / total_seqs * 100
            va_pct = counts.get("val", 0) / total_seqs * 100
            te_pct = counts.get("test", 0) / total_seqs * 100
            print(f"    比例:  train={tr_pct:.1f}%  val={va_pct:.1f}%  test={te_pct:.1f}%")

    # Check npy vs bin
    has_npy = any(info.get(f"npy_{s}") for s in ["train", "val", "test"])
    if has_npy:
        print(f"    格式: \033[32m.npy (已转换)\033[0m")
    else:
        xtr = info['split_counts'].get('X_train', 0)
        ytr = info['split_counts'].get('y1_train', 0)
        xva = info['split_counts'].get('X_val', 0)
        yva = info['split_counts'].get('y1_val', 0)
        xte = info['split_counts'].get('X_test', 0)
        yte = info['split_counts'].get('y1_test', 0)
        print(f"    格式: .bin  推测计数: tr={xtr:,}/{ytr:,} va={xva:,}/{yva:,} te={xte:,}/{yte:,}  (X/y1)")
        aligned = (xtr == ytr and xva == yva and xte == yte)
        align_color = '\033[32m' if aligned else '\033[31m'
        print(f"    对齐检查: {align_color}{'OK' if aligned else 'MISMATCH'}\033[0m")

    total_disk = sum(info['split_sizes_mb'].get(f"X_{s}", 0) for s in ["train", "val", "test"])
    print(f"    磁盘: train={info['split_sizes_mb'].get('X_train', 0):.1f}MB "
          f"val={info['split_sizes_mb'].get('X_val', 0):.1f}MB "
          f"test={info['split_sizes_mb'].get('X_test', 0):.1f}MB "
          f"(合计 {total_disk:.1f}MB)")
    print()

    # ── ETA ──
    print(f"  \033[1m⏱️ 进度预估:\033[0m")
    current = info.get('current_stock') or (prog.get('stock_index', 0) if prog else 0)
    remaining = max(info['total_stocks'] - current, 0)
    if info.get('recent_rate_spm') and info['recent_rate_spm'] > 0:
        eta_mins = remaining / info['recent_rate_spm']
        print(f"    速率: {info['recent_rate_spm']:.1f} stocks/min")
        print(f"    剩余: ~{remaining:,} 只")
        print(f"    ETA:  ~{eta_mins:.0f} 分钟 ({eta_mins / 60:.1f} 小时)")
    elif current > 0:
        print(f"    已处理: {current:,} / {info['total_stocks']:,}  剩余: {remaining:,}")
    else:
        print(f"    暂无足够信息")
    if info.get('last_progress_line'):
        print(f"    \033[90m{info['last_progress_line']}\033[0m")
    print()

    # ── Recent log ──
    print(f"  \033[1m📋 最近日志:\033[0m")
    for line in info["last_lines"][-10:]:
        if "ERROR" in line or "failed" in line:
            print(f"    \033[31m{line}\033[0m")
        elif "sleeping" in line or "Rate limit" in line:
            print(f"    \033[33m{line}\033[0m")
        elif "✅" in line or "completed" in line.lower() or "ok" in line.lower():
            print(f"    \033[32m{line}\033[0m")
        else:
            print(f"    \033[90m{line}\033[0m")


# ════════════════════════════════════════════════════════════════
# Training Monitor
# ════════════════════════════════════════════════════════════════

def parse_epochs(model_id=None):
    epochs = []
    if not LOG.exists():
        return epochs
    for line in LOG.read_text().splitlines():
        try:
            d = json.loads(line)
            msg = d.get("msg", "")
            if not msg.startswith("Epoch"):
                continue
            parts = msg.split("|")
            epoch = int(parts[0].strip().split()[1])

            loss_part = parts[1].strip()
            train_loss = float(loss_part.split("train=")[1].split()[0])
            val_loss = float(loss_part.split("val=")[1].split()[0])
            gap = float(loss_part.split("gap=")[1].split()[0])

            acc_part = parts[2].strip()
            acc_1d = float(acc_part.split("1d=")[1].split("%")[0])

            grad_part = parts[3].strip()
            grad = float(grad_part.split("grad=")[1].split()[0])
            lr = float(grad_part.split("lr=")[1].split()[0])

            time_s = float(parts[4].strip().replace("s", "").split()[0])
            is_best = "★" in msg

            epochs.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "gap": gap, "acc_1d": acc_1d,
                "grad": grad, "lr": lr, "time": time_s, "best": is_best
            })
        except:
            pass
    return epochs


def parse_status():
    msgs = []
    if not LOG.exists():
        return msgs
    for line in LOG.read_text().splitlines():
        try:
            d = json.loads(line)
            msg = d.get("msg", "")
            if not msg.startswith("Epoch"):
                msgs.append(msg)
        except:
            pass
    return msgs


def parse_test_results(status_msgs):
    """Extract test set evaluation results from log messages."""
    results = {}
    current_horizon = None
    in_test = False
    for m in status_msgs:
        if "TEST SET EVALUATION" in m:
            in_test = True
        if not in_test:
            continue
        if "Test Accuracy:" in m:
            continue
        if "Test Accuracy: 1-day=" in m:
            try:
                results["test_1d"] = float(m.split("1-day=")[1].split("%")[0])
                results["test_avg"] = results["test_1d"]
            except:
                pass
        if "── 1-day ──" in m:
            current_horizon = "1d"
        if current_horizon and "Precision=" in m:
            try:
                prec = float(m.split("Precision=")[1].split("%")[0])
                rec = float(m.split("Recall=")[1].split("%")[0])
                f1 = float(m.split("F1=")[1].split("%")[0])
                results[f"prec_{current_horizon}"] = prec
                results[f"rec_{current_horizon}"] = rec
                results[f"f1_{current_horizon}"] = f1
            except:
                pass
        if current_horizon and "TP=" in m:
            try:
                tp = int(m.split("TP=")[1].split()[0])
                fp = int(m.split("FP=")[1].split()[0])
                tn = int(m.split("TN=")[1].split()[0])
                fn = int(m.split("FN=")[1].split()[0])
                results[f"tp_{current_horizon}"] = tp
                results[f"fp_{current_horizon}"] = fp
                results[f"tn_{current_horizon}"] = tn
                results[f"fn_{current_horizon}"] = fn
            except:
                pass
        if "Win rate:" in m:
            try:
                results["win_rate"] = float(m.split("Win rate:")[1].split("%")[0])
            except:
                pass
        if "Edge:" in m:
            try:
                results["edge"] = float(m.split("Edge:")[1].split("%")[0])
            except:
                pass
        if "Trades:" in m:
            try:
                results["trades"] = int(m.split("Trades:")[1].split("/")[0])
                results["total_test"] = int(m.split("/")[1].split("(")[0])
            except:
                pass
        if "No overfitting" in m or "Overfitting" in m:
            results["overfit_msg"] = m.strip()
        if "Calibration: pred_up_1d" in m:
            try:
                pred = float(m.split("pred_up_1d=")[1].split()[0])
                actual = float(m.split("actual=")[1].strip())
                results["cal_1d_pred"] = pred
                results["cal_1d_actual"] = actual
            except:
                pass
        if "Best long threshold:" in m:
            try:
                results["best_conf_thresh"] = m.split("threshold:")[1].split("(")[0].strip()
                results["best_conf_edge"] = m.split("edge=")[1].split(")")[0].strip() if "edge=" in m else ""
            except:
                pass
        if "Val optimal:" in m:
            try:
                results["val_thresh"] = m.split("thresh=")[1].split("(")[0].strip()
                results["val_edge"] = m.split("edge=")[1].split(")")[0].strip() if "edge=" in m else ""
            except:
                pass
        if "Test result:" in m and "trades" in m:
            try:
                results["test_thresh_trades"] = m.split("→")[1].split("trades")[0].strip()
                results["test_thresh_edge"] = m.split("edge=")[1].strip() if "edge=" in m else ""
            except:
                pass
    return results


def parse_ensemble_info(status_msgs):
    """Extract ensemble-level info: which model, how many done."""
    models = []
    current_model = None
    for m in status_msgs:
        if "ENSEMBLE TRAINING:" in m:
            try:
                n = int(m.split(":")[1].strip().split()[0])
                models = [None] * n
            except:
                pass
        if "Model ID:" in m:
            try:
                mid = int(m.split("Model ID:")[1].strip())
                current_model = mid
            except:
                pass
        if current_model is not None and f"Model {current_model} done!" in m:
            models[current_model] = "done"
        if "ENSEMBLE SUMMARY" in m:
            return {"total": len(models), "done": sum(1 for x in models if x == "done"),
                    "current": current_model, "models": models}
    return {"total": len(models) if models else 1, "done": sum(1 for x in models if x == "done"),
            "current": current_model, "models": models}


def load_meta_files():
    """Load all predictor_meta*.json files for ensemble summary."""
    metas = []
    for i in range(10):
        suffix = f"_{i}" if i > 0 else ""
        p = MODEL_DIR / f"predictor_meta{suffix}.json"
        if p.exists():
            try:
                metas.append(json.loads(p.read_text()))
            except:
                pass
    return metas


def draw_training(epochs, status_msgs):
    clear()
    feature_dim = get_feature_dim()

    print("\033[1;32m╔══════════════════════════════════════════════════════════════════╗")
    print(f"║       LSTM-Transformer V4 训练监控  ({feature_dim}维特征, 1d-only)     ║")
    print("╚══════════════════════════════════════════════════════════════════╝\033[0m")
    print()

    # ── Model / training config ──
    info = {}
    for m in status_msgs:
        if "Device:" in m:
            info["device"] = m.split("Device:")[1].strip()
        if "Train:" in m and "Val:" in m and "Features:" in m:
            info["data"] = m.strip()
        if "Model:" in m and "parameters" in m:
            info["params"] = m.strip()
        if "Class balance" in m:
            info["balance"] = m.strip()
        if "Batch size" in m or "Effective batch" in m:
            info["batch"] = m.strip()
        if "seed" in m.lower() and "Model" in m:
            info["seed"] = m.strip()
        if "Training:" in m and "patience" in m:
            info["plan"] = m.strip()
        if "Early stopping" in m:
            info["stop"] = m.strip()

    # ── Ensemble status ──
    ens = parse_ensemble_info(status_msgs)
    if ens["total"] > 1:
        print(f"  \033[1m🎯 Ensemble: 模型 {(ens['current'] or 0) + 1}/{ens['total']}  "
              f"已完成: {ens['done']}/{ens['total']}\033[0m")
        print()

    print(f"  \033[36m{'─' * 64}\033[0m")
    for key in ["device", "seed", "data", "params", "balance", "batch", "plan"]:
        if key in info:
            print(f"  \033[36m│\033[0m {info[key]}")
    if "stop" in info:
        print(f"  \033[36m│\033[0m \033[31m{info['stop']}\033[0m")
    print(f"  \033[36m{'─' * 64}\033[0m")
    print()

    latest = epochs[-1]
    best = min(epochs, key=lambda e: e["val_loss"])
    since_best = latest["epoch"] - best["epoch"]

    print(f"  \033[1m当前 Epoch {latest['epoch']}\033[0m  ", end="")
    if latest["best"]:
        print("\033[1;33m★ NEW BEST\033[0m")
    else:
        patience = 15
        print(f"\033[90m(best=E{best['epoch']}, patience {since_best}/{patience})\033[0m")
    print()

    # ── Loss ──
    print(f"  \033[1mLoss:\033[0m")
    print(f"    Train:  {latest['train_loss']:.6f}")
    print(f"    Val:    {latest['val_loss']:.6f}")
    gap_color = "\033[31m" if abs(latest['gap']) > 0.15 else "\033[33m" if abs(latest['gap']) > 0.10 else "\033[32m"
    print(f"    Gap:    {gap_color}{latest['gap']:+.6f}\033[0m", end="")
    if abs(latest['gap']) > 0.15:
        print("  ⚠️ 过拟合风险")
    elif abs(latest['gap']) > 0.10:
        print("  ⚡ 轻度过拟合")
    else:
        print("  ✅ 健康")
    print()

    # ── Accuracy (1d only) ──
    print(f"  \033[1mAccuracy (1d):\033[0m")
    val = latest["acc_1d"]
    color = "\033[32m" if val > 55 else "\033[33m" if val > 50 else "\033[31m"
    icon = "✅" if val > 55 else "〰️" if val > 50 else "❌"
    print(f"    1天: {color}{val:5.1f}%\033[0m  {bar(val)}  {icon}")
    print()

    # ── Training dynamics ──
    print(f"  \033[1m训练动态:\033[0m")
    print(f"    Gradient Norm: {latest['grad']:.4f}", end="")
    if latest['grad'] > 0.5:
        print("  \033[31m⚠️ 梯度爆炸\033[0m")
    elif latest['grad'] < 0.01:
        print("  \033[33m⚠️ 梯度消失\033[0m")
    else:
        print("  \033[32m✅ 稳定\033[0m")
    print(f"    Learning Rate: {latest['lr']:.2e}")
    print(f"    Epoch Time:    {latest['time']:.0f}s ({latest['time'] / 60:.1f} min)")
    print()

    # ── Val loss plateau detection ──
    if len(epochs) >= 10:
        recent_10 = [e["val_loss"] for e in epochs[-10:]]
        best_recent = min(recent_10)
        worst_recent = max(recent_10)
        plateau_range = worst_recent - best_recent
        print(f"  \033[1m📈 Val Loss 趋势 (近10轮):\033[0m")
        print(f"    范围: {best_recent:.6f} ~ {worst_recent:.6f}  (波动={plateau_range:.6f})")
        if plateau_range < 0.002:
            print(f"    \033[33m⚠️ 已进入平台期 — val_loss 近乎停滞\033[0m")
        elif all(recent_10[i] >= recent_10[i - 1] - 0.001 for i in range(1, len(recent_10))):
            print(f"    \033[31m↗ 上升趋势 — 过拟合加剧\033[0m")
        else:
            print(f"    \033[32m正常波动\033[0m")
        print()

    # ── Sparklines ──
    print(f"  \033[1m趋势 (E1 → E{latest['epoch']}):\033[0m")
    tl = [e["train_loss"] for e in epochs]
    vl = [e["val_loss"] for e in epochs]
    a1 = [e["acc_1d"] for e in epochs]
    gn = [e["grad"] for e in epochs]
    lrs = [e["lr"] for e in epochs]
    print(f"    TrainLoss: {spark(tl)}  {tl[0]:.3f}→{tl[-1]:.3f}")
    print(f"    Val Loss:  {spark(vl)}  {vl[0]:.3f}→{vl[-1]:.3f}")
    print(f"    1d Acc:    {spark(a1)}  {a1[0]:.1f}→{a1[-1]:.1f}%")
    print(f"    GradNorm:  {spark(gn)}  {gn[0]:.3f}→{gn[-1]:.3f}")
    print(f"    LR:        {spark(lrs)}  {lrs[0]:.2e}→{lrs[-1]:.2e}")
    print()

    # ── Records ──
    print(f"  \033[1m🏆 记录:\033[0m")
    best_acc_1d = max(epochs, key=lambda e: e["acc_1d"])
    print(f"    最低 Val Loss:  E{best['epoch']}  {best['val_loss']:.6f}  (1d={best['acc_1d']:.1f}%)")
    print(f"    最高 1d Acc:    E{best_acc_1d['epoch']}  {best_acc_1d['acc_1d']:.1f}%")
    print()

    # ── Delta ──
    if len(epochs) >= 2:
        prev = epochs[-2]
        print(f"  \033[1mΔ 与上一轮:\033[0m")
        d_vl = latest['val_loss'] - prev['val_loss']
        d_1d = latest['acc_1d'] - prev['acc_1d']
        vc = "\033[32m" if d_vl < 0 else "\033[31m"
        print(f"    Val Loss: {vc}{d_vl:+.6f}\033[0m | 1d: {d_1d:+.1f}%")
        print()

    # ── Epoch table (last 20 + all best) ──
    print(f"  \033[1m近期 Epoch:\033[0m")
    header = f"  \033[90m{'Ep':>4} │ {'TrLoss':>9} │ {'VaLoss':>9} │ {'Gap':>8} │ {'1d%':>5} │ {'Grad':>6} │ {'LR':>9} │\033[0m"
    print(header)
    print(f"  \033[90m{'─' * 4}─┼─{'─' * 9}─┼─{'─' * 9}─┼─{'─' * 8}─┼─{'─' * 5}─┼─{'─' * 6}─┼─{'─' * 9}─┤\033[0m")

    show_epochs = epochs[-20:] if len(epochs) > 20 else epochs
    if len(epochs) > 20:
        print(f"  \033[90m  ... ({len(epochs) - 20} earlier epochs hidden)\033[0m")

    def ac(v):
        if v > 55:
            return f"\033[32m{v:5.1f}\033[0m"
        elif v > 50:
            return f"\033[33m{v:5.1f}\033[0m"
        return f"\033[31m{v:5.1f}\033[0m"

    for e in show_epochs:
        star = "\033[33m★\033[0m" if e["best"] else " "
        gc = "\033[31m" if abs(e["gap"]) > 0.15 else "\033[33m" if abs(e["gap"]) > 0.10 else "\033[32m"
        print(f"  {e['epoch']:4d} \033[90m│\033[0m {e['train_loss']:9.6f} \033[90m│\033[0m "
              f"{e['val_loss']:9.6f} \033[90m│\033[0m {gc}{e['gap']:+8.5f}\033[0m \033[90m│\033[0m "
              f"{ac(e['acc_1d'])} \033[90m│\033[0m {e['grad']:6.4f} \033[90m│\033[0m "
              f"{e['lr']:9.2e} \033[90m│\033[0m{star}")
    print()

    # ── Overfit analysis ──
    if len(epochs) >= 5:
        recent_gaps = [abs(e["gap"]) for e in epochs[-5:]]
        early_gaps = [abs(e["gap"]) for e in epochs[:5]]
        avg_recent = sum(recent_gaps) / len(recent_gaps)
        avg_early = sum(early_gaps) / len(early_gaps)
        print(f"  \033[1m过拟合分析:\033[0m")
        print(f"    Gap 趋势: 前期={avg_early:.4f} → 近期={avg_recent:.4f}", end="")
        if avg_recent > avg_early * 1.3:
            print("  \033[31m↑ 加剧\033[0m")
        elif avg_recent < avg_early * 0.8:
            print("  \033[32m↓ 减轻\033[0m")
        else:
            print("  \033[33m→ 稳定\033[0m")
        print()

    # ── Test results (if training finished) ──
    test = parse_test_results(status_msgs)
    if test.get("test_avg"):
        print(f"  \033[1;35m{'═' * 64}\033[0m")
        print(f"  \033[1;35m TEST SET 评估结果\033[0m")
        print(f"  \033[1;35m{'═' * 64}\033[0m")
        print(f"    Accuracy (1d): {test.get('test_1d', 0):.1f}%")
        if "prec_1d" in test:
            print(f"    1d: P={test['prec_1d']:.1f}% R={test['rec_1d']:.1f}% "
                  f"F1={test['f1_1d']:.1f}%  "
                  f"(TP={test.get('tp_1d', 0)} FP={test.get('fp_1d', 0)} "
                  f"TN={test.get('tn_1d', 0)} FN={test.get('fn_1d', 0)})")
        if test.get("win_rate"):
            print(f"    \033[1m盈利 (1d, >55%置信):\033[0m "
                  f"胜率={test['win_rate']:.1f}%  edge={test.get('edge', 0):+.1f}%  "
                  f"交易={test.get('trades', 0)}/{test.get('total_test', 0)}")
        if test.get("cal_1d_pred"):
            cal_diff = abs(test["cal_1d_pred"] - test["cal_1d_actual"])
            cal_icon = "✅" if cal_diff < 0.05 else "⚠️"
            print(f"    校准: pred_mean={test['cal_1d_pred']:.3f} actual={test['cal_1d_actual']:.3f}  {cal_icon}")
        if test.get("best_conf_thresh"):
            print(f"    \033[1;36m最佳置信阈值:\033[0m {test['best_conf_thresh']} "
                  f"(test edge={test.get('best_conf_edge', '?')})")
        if test.get("val_thresh"):
            print(f"    \033[1;36mVal最优阈值:\033[0m {test['val_thresh']} "
                  f"(val edge={test.get('val_edge', '?')})")
        if test.get("test_thresh_trades"):
            print(f"    \033[1;36m应用到Test:\033[0m "
                  f"{test['test_thresh_trades']} trades, edge={test.get('test_thresh_edge', '?')}")
        if test.get("overfit_msg"):
            print(f"    {test['overfit_msg']}")
        print()

    # ── Time ──
    total_time = sum(e["time"] for e in epochs)
    avg_time = total_time / len(epochs)
    print(f"  \033[1m⏱️ 时间:\033[0m")
    print(f"    每Epoch: {avg_time:.0f}s | 已用: {total_time / 3600:.1f}h | patience: {since_best}/15")
    if since_best < 15 and not test.get("test_avg"):
        remaining = (15 - since_best) * avg_time
        print(f"    最多还需: {remaining / 3600:.1f}h (当前模型)")
    if ens["total"] > 1:
        models_left = ens["total"] - ens["done"] - 1
        if models_left > 0:
            est_per_model = total_time
            print(f"    剩余模型: {models_left} 个, 预计 ~{models_left * est_per_model / 3600:.1f}h")

    # ── Ensemble summary ──
    metas = load_meta_files()
    if len(metas) > 1:
        print()
        print(f"  \033[1;36m{'═' * 64}\033[0m")
        print(f"  \033[1;36m ENSEMBLE 汇总 ({len(metas)} 模型)\033[0m")
        print(f"  \033[1;36m{'═' * 64}\033[0m")
        print(f"  \033[90m  {'#':>2} {'Model':>5} {'Seed':>5} │ {'Val%':>5} │ {'Test%':>6} │ {'Edge':>6} │ {'Ep':>3}\033[0m")
        print(f"  \033[90m  {'─'*2}─{'─'*5}─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*3}\033[0m")

        ranked = sorted(enumerate(metas), key=lambda x: x[1].get("test_avg_accuracy", 0), reverse=True)
        for rank, (i, meta) in enumerate(ranked):
            test_acc = meta.get("test_avg_accuracy", 0)
            edge = meta.get("test_edge", 0)
            seed = 42 + i
            marker = "\033[33m★\033[0m" if rank == 0 else " "
            tc = "\033[32m" if test_acc > 52 else "\033[33m" if test_acc > 50 else "\033[31m"
            ec = "\033[32m" if edge > 1 else "\033[33m" if edge > 0 else "\033[31m"
            print(f"  {rank+1:>3} M{i:>3d}  {seed:>4}  \033[90m│\033[0m "
                  f"{meta.get('accuracy_1d', 0):5.1f} \033[90m│\033[0m "
                  f"{tc}{test_acc:5.1f}%\033[0m \033[90m│\033[0m "
                  f"{ec}{edge:+5.1f}%\033[0m \033[90m│\033[0m "
                  f"E{meta.get('epoch', '?'):>2}{marker}")

        test_accs = [m.get("test_avg_accuracy", 0) for m in metas if isinstance(m.get("test_avg_accuracy"), (int, float))]
        if test_accs:
            good = [a for a in test_accs if a > 52]
            print(f"    \033[1m信号稳定性: {len(good)}/{len(test_accs)} 模型 test>52%\033[0m")
            print(f"    \033[1m平均={sum(test_accs)/len(test_accs):.1f}% "
                  f"std={__import__('statistics').stdev(test_accs) if len(test_accs)>1 else 0:.1f}% "
                  f"max={max(test_accs):.1f}% min={min(test_accs):.1f}%\033[0m")


def main():
    refresh = 15
    print(f"\033[32m🔄 监控启动... 每 {refresh}s 刷新 (Ctrl+C 退出)\033[0m\n")

    try:
        while True:
            epochs = parse_epochs()

            if epochs:
                status = parse_status()
                draw_training(epochs, status)
            else:
                info = get_collection_status()
                draw_collection(info)

            now = datetime.now().strftime("%H:%M:%S")
            pid_check = os.popen(
                "ps aux | grep train_predictor | grep python | grep -v grep | awk '{print $2}'"
            ).read().strip()
            pid_status = f"\033[32mPID {pid_check}\033[0m" if pid_check else "\033[31m未运行\033[0m"
            print(f"\n  \033[90m🕐 {now}  |  {pid_status}  |  刷新 {refresh}s  |  Ctrl+C 退出\033[0m")

            time.sleep(refresh)
    except KeyboardInterrupt:
        print("\n\n  👋 监控已停止")


if __name__ == "__main__":
    main()

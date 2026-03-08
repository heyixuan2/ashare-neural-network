"""
训练监控脚本 — 实时查看数据收集 + 训练进度
用法: python monitor_training.py
"""
import json, time, os, sys, glob
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
LOG = BASE / "models" / "training_log.jsonl"
OUTPUT_LOG = BASE / "models" / "training_output.log"
CACHE_DIR = BASE / ".cache"
DATA_DIR = BASE / "data" / "ashare_daily"
SPLIT_DIR = BASE / "models" / "splits"
PROGRESS_JSON = SPLIT_DIR / "progress.json"
SEQ_LEN = 30
FEATURE_DIM = 49

def clear():
    os.system("clear" if os.name != "nt" else "cls")

# ── Data Collection Monitor ──

def get_collection_status():
    """Parse training_output.log for data collection progress"""
    info = {
        "total_stocks": 5808,
        "phase": "unknown",
        "extra_caches": 0,
        "extra_complete": 0,
        "extra_margin_real": 0,
        "ohlcv_files": 0,
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
            except: pass
        if "Rate limit pause" in line or "Extra API rate limit" in line:
            info["rate_limit_pauses"] += 1
        if "margin_detail" in line and "failed" in line:
            info["errors"].append(line.strip())
        if "Building sequences" in line or "Streaming" in line:
            info["phase"] = "building"
        if "Training model" in line or "Epoch" in line:
            info["phase"] = "training"
        if "stocks processed" in line:
            try:
                info["current_stock"] = line.split("stocks processed")[0].strip().split()[-1]
            except: pass
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
    
    # Count caches
    extra_files = glob.glob(str(CACHE_DIR / "extra_*.json"))
    info["extra_caches"] = len(extra_files)

    # Checkpoint progress
    if PROGRESS_JSON.exists():
        try:
            info["progress"] = json.loads(PROGRESS_JSON.read_text())
        except:
            pass

    # Split file sizes / inferred counts
    seq_bytes = SEQ_LEN * FEATURE_DIM * 4
    for split_name in ["train", "val", "test"]:
        x_path = SPLIT_DIR / f"X_{split_name}.bin"
        y_path = SPLIT_DIR / f"y1_{split_name}.bin"
        if x_path.exists():
            info["split_sizes_mb"][f"X_{split_name}"] = x_path.stat().st_size / 1024 / 1024
            info["split_counts"][f"X_{split_name}"] = x_path.stat().st_size // seq_bytes
        if y_path.exists():
            info["split_sizes_mb"][f"y1_{split_name}"] = y_path.stat().st_size / 1024 / 1024
            info["split_counts"][f"y1_{split_name}"] = y_path.stat().st_size // 4
    
    # Validate completeness (sample 50 max for speed)
    import random
    sample = random.sample(extra_files, min(50, len(extra_files))) if extra_files else []
    for f in sample:
        try:
            d = json.loads(open(f).read())
            if not d: continue
            dates = sorted(d.keys())
            mid = d[dates[len(dates)//2]]
            if "big_net" in mid and "rzye" in mid:
                info["extra_complete"] += 1
                if any(d[dt].get("rzye", 0) > 0 for dt in list(d.keys())[:50]):
                    info["extra_margin_real"] += 1
        except: pass
    
    # Scale up from sample
    if sample:
        ratio = len(extra_files) / len(sample)
        info["extra_complete"] = int(info["extra_complete"] * ratio)
        info["extra_margin_real"] = int(info["extra_margin_real"] * ratio)
    
    # Only count actual stock files (start with 0/3/6 digit codes), not daily_* caches
    import re
    info["ohlcv_files"] = len([f for f in glob.glob(str(DATA_DIR / "*.json"))
                                if re.match(r'[036]\d{5}_', os.path.basename(f))])

    # Recent speed from progress lines (stocks per minute)
    if len(progress_points) >= 2:
        a = progress_points[-2]
        b = progress_points[-1]
        dt = max(b[0] - a[0], 1e-6)
        ds = b[1] - a[1]
        info["recent_rate_spm"] = ds / dt

    return info

def draw_collection(info):
    """Draw data collection dashboard"""
    clear()
    
    print("\033[1;32m╔══════════════════════════════════════════════════════════════════╗")
    print("║         🧠  LSTM-Transformer V2.6 训练监控  (49维)             ║")
    print("╚══════════════════════════════════════════════════════════════════╝\033[0m")
    print()
    
    # Phase indicator
    phase_labels = {
        "collecting": "📡 数据收集中",
        "building": "🔧 构建序列中",
        "training": "🏋️ 训练中",
        "unknown": "⏳ 启动中",
    }
    print(f"  \033[1m阶段: {phase_labels.get(info['phase'], info['phase'])}\033[0m")
    if info["started"]:
        print(f"  \033[90m开始时间: {info['started']}\033[0m")
    print()
    
    # OHLCV data
    ohlcv_pct = info["ohlcv_files"] / max(info["total_stocks"], 1) * 100
    print(f"  \033[1m📊 OHLCV 数据:\033[0m")
    print(f"    文件数: {info['ohlcv_files']:,} / {info['total_stocks']:,} ({ohlcv_pct:.0f}%)")
    ohlcv_bar = progress_bar(info["ohlcv_files"], info["total_stocks"], 40)
    print(f"    {ohlcv_bar}")
    print()
    
    # Extra API data
    extra_pct = info["extra_caches"] / max(info["total_stocks"], 1) * 100
    print(f"  \033[1m📡 Extra API 数据 (daily_basic + moneyflow + margin):\033[0m")
    print(f"    缓存数: {info['extra_caches']:,} / {info['total_stocks']:,} ({extra_pct:.0f}%)")
    extra_bar = progress_bar(info["extra_caches"], info["total_stocks"], 40)
    print(f"    {extra_bar}")
    print(f"    完整率: {info['extra_complete']:,} / {info['extra_caches']:,} "
          f"({info['extra_complete']/max(info['extra_caches'],1)*100:.0f}%)")
    print(f"    两融标的: ~{info['extra_margin_real']:,}")
    print(f"    Rate limit 暂停: {info['rate_limit_pauses']} 次")
    if info["errors"]:
        print(f"    \033[31mMargin 错误: {len(info['errors'])} 次\033[0m")
    print()
    
    # Checkpoint / split status
    print(f"  \033[1m💾 Split 写盘状态:\033[0m")
    prog = info.get("progress") or {}
    if prog:
        counts = prog.get("counts", {})
        print(f"    Checkpoint: stock #{prog.get('stock_index', '?'):,} | seqs={prog.get('total_seqs', '?'):,} | F={prog.get('feature_dim', '?')}")
        print(f"    Progress计数: tr={counts.get('train', 0):,} va={counts.get('val', 0):,} te={counts.get('test', 0):,}")
    xtr = info['split_counts'].get('X_train', 0); ytr = info['split_counts'].get('y1_train', 0)
    xva = info['split_counts'].get('X_val', 0);   yva = info['split_counts'].get('y1_val', 0)
    xte = info['split_counts'].get('X_test', 0);  yte = info['split_counts'].get('y1_test', 0)
    print(f"    Bin计数:      tr={xtr:,}/{ytr:,} va={xva:,}/{yva:,} te={xte:,}/{yte:,}  (X/y1)")
    aligned = (xtr == ytr and xva == yva and xte == yte)
    align_color = '\033[32m' if aligned else '\033[31m'
    print(f"    对齐检查: {align_color}{'OK' if aligned else 'MISMATCH'}\033[0m")
    print(f"    文件体积: Xtr={info['split_sizes_mb'].get('X_train', 0):.1f}MB Xva={info['split_sizes_mb'].get('X_val', 0):.1f}MB Xte={info['split_sizes_mb'].get('X_test', 0):.1f}MB")
    print()

    # ETA / speed
    print(f"  \033[1m⏱️ 预估:\033[0m")
    remaining = max(info['total_stocks'] - (info.get('current_stock') or 0), 0)
    if info.get('recent_rate_spm') and info['recent_rate_spm'] > 0:
        eta_mins = remaining / info['recent_rate_spm']
        print(f"    当前速率: {info['recent_rate_spm']:.1f} stocks/min")
        print(f"    剩余股票: ~{remaining:,} 只")
        print(f"    ETA: ~{eta_mins:.0f} 分钟 ({eta_mins/60:.1f} 小时)")
    elif info["extra_caches"] > 0 and info["extra_caches"] < info["total_stocks"]:
        remaining = info["total_stocks"] - info["extra_caches"]
        cycles_left = remaining / 50
        eta_mins = cycles_left * (62 + 50) / 60
        print(f"    剩余: ~{remaining:,} 只")
        print(f"    预计: ~{eta_mins:.0f} 分钟 ({eta_mins/60:.1f} 小时)")
    else:
        print(f"    暂无足够信息")
    if info.get('run_restarts', 0) > 1:
        print(f"    运行轮次: {info['run_restarts']} 次启动记录")
    if info.get('last_progress_line'):
        print(f"    最近进度: {info['last_progress_line']}")
    print()
    
    # Recent log
    print(f"  \033[1m📋 最近日志:\033[0m")
    for line in info["last_lines"][-8:]:
        # Color code
        if "ERROR" in line or "failed" in line:
            print(f"    \033[31m{line}\033[0m")
        elif "sleeping" in line or "Rate limit" in line:
            print(f"    \033[33m{line}\033[0m")
        elif "✅" in line or "completed" in line.lower():
            print(f"    \033[32m{line}\033[0m")
        else:
            print(f"    \033[90m{line}\033[0m")

def progress_bar(current, total, width=40):
    ratio = min(1.0, current / max(total, 1))
    filled = int(width * ratio)
    bar = "\033[32m" + "█" * filled + "\033[90m" + "░" * (width - filled) + "\033[0m"
    return f"  {bar} {ratio*100:.1f}%"

# ── Training Monitor (existing, enhanced) ──

def parse_epochs():
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
            acc_3d = float(acc_part.split("3d=")[1].split("%")[0])
            acc_5d = float(acc_part.split("5d=")[1].split("%")[0])
            
            grad_part = parts[3].strip()
            grad = float(grad_part.split("grad=")[1].split()[0])
            lr = float(grad_part.split("lr=")[1].split()[0])
            
            time_s = float(parts[4].strip().replace("s", ""))
            is_best = "★" in msg
            
            epochs.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "gap": gap, "acc_1d": acc_1d, "acc_3d": acc_3d, "acc_5d": acc_5d,
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

def draw_training(epochs, status_msgs):
    clear()
    
    print("\033[1;32m╔══════════════════════════════════════════════════════════════════╗")
    print("║         🧠  LSTM-Transformer V2.6 训练监控  (49维)             ║")
    print("╚══════════════════════════════════════════════════════════════════╝\033[0m")
    print()
    
    # Parse key info
    info = {}
    for m in status_msgs:
        if "Device:" in m: info["device"] = m.split("Device:")[1].strip()
        if "Train:" in m and "Val:" in m: info["data"] = m.strip()
        if "Model:" in m and "parameters" in m: info["params"] = m.strip()
        if "Class balance" in m: info["balance"] = m.strip()
        if "Focal alpha" in m: info["alpha"] = m.strip()
        if "Model ID" in m: info["model_id"] = m.strip()
        if "Batch size" in m: info["batch"] = m.strip()
        if "accumulation" in m: info["accum"] = m.strip()
        if "seed" in m.lower() and "Model" in m: info["seed"] = m.strip()
        if "Training:" in m and "patience" in m: info["plan"] = m.strip()
        if "SWA activated" in m: info["swa"] = m.strip()
        if "Early stopping" in m: info["stop"] = m.strip()
    
    # Model info
    print(f"  \033[36m{'─' * 64}\033[0m")
    for key in ["device", "seed", "data", "params", "balance", "alpha", "batch", "accum", "plan", "swa", "stop"]:
        if key in info:
            print(f"  \033[36m│\033[0m {info[key]}")
    print(f"  \033[36m{'─' * 64}\033[0m")
    print()
    
    latest = epochs[-1]
    best = min(epochs, key=lambda e: e["val_loss"])
    best_acc = max(epochs, key=lambda e: e["acc_1d"])
    since_best = latest["epoch"] - best["epoch"]
    avg_acc = (latest["acc_1d"] + latest["acc_3d"] + latest["acc_5d"]) / 3
    
    print(f"  \033[1m当前 Epoch {latest['epoch']}\033[0m  ", end="")
    if latest["best"]:
        print("\033[1;33m★ NEW BEST\033[0m")
    else:
        print(f"\033[90m(best=E{best['epoch']}, patience {since_best}/30)\033[0m")
    print()
    
    # Loss
    print(f"  \033[1mLoss:\033[0m")
    print(f"    Train:  {latest['train_loss']:.6f}")
    print(f"    Val:    {latest['val_loss']:.6f}")
    gap_color = "\033[31m" if abs(latest['gap']) > 0.15 else "\033[33m" if abs(latest['gap']) > 0.10 else "\033[32m"
    print(f"    Gap:    {gap_color}{latest['gap']:+.6f}\033[0m", end="")
    if abs(latest['gap']) > 0.15: print("  ⚠️  过拟合风险")
    elif abs(latest['gap']) > 0.10: print("  ⚡ 轻度过拟合")
    else: print("  ✅ 健康")
    print()
    
    # Accuracy
    print(f"  \033[1mAccuracy:\033[0m")
    for label, val in [("1天", latest["acc_1d"]), ("3天", latest["acc_3d"]), ("5天", latest["acc_5d"])]:
        color = "\033[32m" if val > 55 else "\033[33m" if val > 50 else "\033[31m"
        icon = "✅" if val > 55 else "〰️" if val > 50 else "❌"
        print(f"    {label}: {color}{val:5.1f}%\033[0m  {bar(val)}  {icon}")
    print(f"    平均: {avg_acc:.1f}%")
    print()
    
    # Dynamics
    print(f"  \033[1m训练动态:\033[0m")
    print(f"    Gradient Norm: {latest['grad']:.4f}", end="")
    if latest['grad'] > 0.5: print("  \033[31m⚠️ 梯度爆炸\033[0m")
    elif latest['grad'] < 0.01: print("  \033[33m⚠️ 梯度消失\033[0m")
    else: print("  \033[32m✅ 稳定\033[0m")
    print(f"    Learning Rate: {latest['lr']:.2e}")
    print(f"    Epoch Time:    {latest['time']:.0f}s ({latest['time']/60:.1f} min)")
    print()
    
    # Sparklines
    print(f"  \033[1m趋势 (E1 → E{latest['epoch']}):\033[0m")
    tl = [e["train_loss"] for e in epochs]
    vl = [e["val_loss"] for e in epochs]
    a1 = [e["acc_1d"] for e in epochs]
    a3 = [e["acc_3d"] for e in epochs]
    a5 = [e["acc_5d"] for e in epochs]
    gn = [e["grad"] for e in epochs]
    lrs = [e["lr"] for e in epochs]
    print(f"    TrainLoss: {spark(tl)}  {tl[0]:.3f}→{tl[-1]:.3f}")
    print(f"    Val Loss:  {spark(vl)}  {vl[0]:.3f}→{vl[-1]:.3f}")
    print(f"    1d Acc:    {spark(a1)}  {a1[0]:.1f}→{a1[-1]:.1f}%")
    print(f"    3d Acc:    {spark(a3)}  {a3[0]:.1f}→{a3[-1]:.1f}%")
    print(f"    5d Acc:    {spark(a5)}  {a5[0]:.1f}→{a5[-1]:.1f}%")
    print(f"    GradNorm:  {spark(gn)}  {gn[0]:.3f}→{gn[-1]:.3f}")
    print(f"    LR:        {spark(lrs)}  {lrs[0]:.2e}→{lrs[-1]:.2e}")
    print()
    
    # Records
    print(f"  \033[1m🏆 记录:\033[0m")
    print(f"    最低 Val Loss:  E{best['epoch']}  {best['val_loss']:.6f}  (1d={best['acc_1d']:.1f}%)")
    print(f"    最高 1d Acc:    E{best_acc['epoch']}  {best_acc['acc_1d']:.1f}%")
    best_3d = max(epochs, key=lambda e: e["acc_3d"])
    best_5d = max(epochs, key=lambda e: e["acc_5d"])
    print(f"    最高 3d Acc:    E{best_3d['epoch']}  {best_3d['acc_3d']:.1f}%")
    print(f"    最高 5d Acc:    E{best_5d['epoch']}  {best_5d['acc_5d']:.1f}%")
    print()
    
    # Recent epoch deltas
    if len(epochs) >= 2:
        prev = epochs[-2]
        print(f"  \033[1mΔ 与上一轮相比:\033[0m")
        print(f"    Val Loss: {latest['val_loss'] - prev['val_loss']:+.6f} | 1d: {latest['acc_1d'] - prev['acc_1d']:+.1f}% | 3d: {latest['acc_3d'] - prev['acc_3d']:+.1f}% | 5d: {latest['acc_5d'] - prev['acc_5d']:+.1f}%")
        print()

    # Epoch table
    print(f"  \033[1m全部 Epoch:\033[0m")
    header = f"  \033[90m{'Ep':>4} │ {'TrLoss':>9} │ {'VaLoss':>9} │ {'Gap':>8} │ {'1d%':>5} │ {'3d%':>5} │ {'5d%':>5} │ {'Grad':>6} │ {'LR':>9} │\033[0m"
    print(header)
    print(f"  \033[90m{'─'*4}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*9}─┤\033[0m")
    
    for e in epochs:
        star = "\033[33m★\033[0m" if e["best"] else " "
        gc = "\033[31m" if abs(e["gap"]) > 0.15 else "\033[33m" if abs(e["gap"]) > 0.10 else "\033[32m"
        def ac(v):
            return f"\033[32m{v:5.1f}\033[0m" if v > 55 else f"\033[33m{v:5.1f}\033[0m" if v > 50 else f"\033[31m{v:5.1f}\033[0m"
        print(f"  {e['epoch']:4d} \033[90m│\033[0m {e['train_loss']:9.6f} \033[90m│\033[0m {e['val_loss']:9.6f} \033[90m│\033[0m {gc}{e['gap']:+8.5f}\033[0m \033[90m│\033[0m {ac(e['acc_1d'])} \033[90m│\033[0m {ac(e['acc_3d'])} \033[90m│\033[0m {ac(e['acc_5d'])} \033[90m│\033[0m {e['grad']:6.4f} \033[90m│\033[0m {e['lr']:9.2e} \033[90m│\033[0m{star}")
    print()
    
    # Overfit analysis
    if len(epochs) >= 5:
        recent_gaps = [abs(e["gap"]) for e in epochs[-5:]]
        early_gaps = [abs(e["gap"]) for e in epochs[:5]]
        avg_recent = sum(recent_gaps) / len(recent_gaps)
        avg_early = sum(early_gaps) / len(early_gaps)
        print(f"  \033[1m过拟合分析:\033[0m")
        print(f"    Gap 趋势: 前期={avg_early:.4f} → 近期={avg_recent:.4f}", end="")
        if avg_recent > avg_early * 1.3: print("  \033[31m↑ 加剧\033[0m")
        elif avg_recent < avg_early * 0.8: print("  \033[32m↓ 减轻\033[0m")
        else: print("  \033[33m→ 稳定\033[0m")
    
    # Time
    total_time = sum(e["time"] for e in epochs)
    avg_time = total_time / len(epochs)
    print()
    print(f"  \033[1m⏱️ 时间:\033[0m")
    print(f"    每Epoch: {avg_time:.0f}s | 已用: {total_time/3600:.1f}h | patience: {since_best}/30")
    if since_best < 30:
        remaining = (30 - since_best) * avg_time
        print(f"    最多还需: {remaining/3600:.1f}h")


def main():
    refresh = 15
    print(f"\033[32m🔄 监控启动... 每 {refresh}s 刷新 (Ctrl+C 退出)\033[0m\n")
    
    try:
        while True:
            epochs = parse_epochs()
            
            if epochs:
                # Training phase
                status = parse_status()
                draw_training(epochs, status)
            else:
                # Data collection phase
                info = get_collection_status()
                draw_collection(info)
            
            now = datetime.now().strftime("%H:%M:%S")
            pid_check = os.popen("ps aux | grep train_predictor | grep python | grep -v grep | awk '{print $2}'").read().strip()
            pid_status = f"\033[32mPID {pid_check}\033[0m" if pid_check else "\033[31m未运行\033[0m"
            print(f"\n  \033[90m🕐 {now}  |  {pid_status}  |  刷新 {refresh}s  |  Ctrl+C 退出\033[0m")
            
            time.sleep(refresh)
    except KeyboardInterrupt:
        print("\n\n  👋 监控已停止")


if __name__ == "__main__":
    main()

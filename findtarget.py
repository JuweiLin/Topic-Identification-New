import pandas as pd, numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

IN_CSV  = "./qwen_mlp_ckpt/val_pred_new1.csv"
OUT_CSV = "./qwen_mlp_ckpt/val_pred_new1.csv"  # 直接回写（追加列）
SWEEP_CSV = "./qwen_mlp_ckpt/precision_sweep_30_50.csv"

TARGET_START = 0.50
TARGET_END   = 0.60
TARGET_STEP  = 0.01
THR_GRID = np.linspace(0.99, 0.01, 990)  # 搜索阈值网格
MIN_PRED_POS = 20                        # 至少要有这么多正预测才算数
KEY_METRIC = "F1"                        # 用哪个指标挑“最好”：F1 / R / ACC / P

def canon_label(x):
    if pd.isna(x): return None
    s = str(x).strip().upper().replace("$","").replace("_","")
    if s == "NEW1": return "NEW1"
    return None  # 只关心 NEW1 的二分类

def y_true_from_row(tran, label):
    if pd.notna(tran) and str(tran) in ("0","1"):
        return int(tran)
    lab = canon_label(label)
    return 1 if lab == "NEW1" else 0

def metrics_at_threshold(y_true, probs, t):
    pred = (probs >= t).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    ACC = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"t": float(t), "P": float(P), "R": float(R), "F1": float(F1), "ACC": float(ACC),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "pred_pos": int(tp+fp)}

def find_threshold_for_precision(y_true, probs, target=0.80, min_pred=20):
    # 先找 Precision≥target 的阈值里 Recall 最大的
    best = None
    for t in THR_GRID:
        pred = (probs >= t).astype(int)
        tp = int(((pred==1)&(y_true==1)).sum())
        fp = int(((pred==1)&(y_true==0)).sum())
        fn = int(((pred==0)&(y_true==1)).sum())
        if tp+fp < max(min_pred,1):  # 避免样本太少
            continue
        P = tp/(tp+fp) if tp+fp>0 else 0.0
        R = tp/(tp+fn) if tp+fn>0 else 0.0
        if P >= target:
            if best is None or (R > best["R"]) or (R==best["R"] and P > best["P"]):
                best = {"t": float(t), "P": P, "R": R}
    if best is not None:
        return best["t"]

    # 否则返回 Precision 最高的阈值（仍要求最少正预测数）
    bestP = None
    for t in THR_GRID:
        pred = (probs >= t).astype(int)
        tp = int(((pred==1)&(y_true==1)).sum())
        fp = int(((pred==1)&(y_true==0)).sum())
        if tp+fp < max(min_pred,1):
            continue
        P = tp/(tp+fp) if tp+fp>0 else 0.0
        if bestP is None or P > bestP["P"]:
            bestP = {"t": float(t), "P": P}
    return bestP["t"] if bestP is not None else 0.5

# --- 主流程 ---
df = pd.read_csv(IN_CSV)
if "prob_NEW1" not in df.columns:
    raise ValueError("CSV 里缺少 prob_NEW1 列")

# 构造二分类真值 y_true
y_true = np.array([y_true_from_row(row.get("tran"), row.get("label")) for _, row in df.iterrows()], dtype=int)
probs  = df["prob_NEW1"].values.astype(float)

# 汇总：对每个 Precision 目标找阈值并记指标
rows = []
targets = np.arange(int(TARGET_START*100), int(TARGET_END*100)+1, int(TARGET_STEP*100)) / 100.0
for tgt in targets:
    t = find_threshold_for_precision(y_true, probs, target=tgt, min_pred=MIN_PRED_POS)
    m = metrics_at_threshold(y_true, probs, t)
    rows.append({"targetP": float(tgt), **m})

sweep = pd.DataFrame(rows)
sweep.to_csv(SWEEP_CSV, index=False, encoding="utf-8-sig")
print(f"Saved sweep → {SWEEP_CSV}")

# 选“最好”的 target（按 KEY_METRIC 最大）
best_idx = sweep[KEY_METRIC].values.argmax()
best_row = sweep.iloc[best_idx].to_dict()
t_best = float(best_row["t"])
print(f"[BEST by {KEY_METRIC}] targetP={best_row['targetP']:.2f}  "
      f"t={t_best:.2f}  P={best_row['P']:.3f} R={best_row['R']:.3f} "
      f"F1={best_row['F1']:.3f} ACC={best_row['ACC']:.3f}  "
      f"(TP={int(best_row['TP'])}, FP={int(best_row['FP'])}, TN={int(best_row['TN'])}, FN={int(best_row['FN'])})")

# 用 t_best 生成一列“高精度风格”的 NEW1 预测，写回原CSV（追加列）
df["pred_NEW1_Pbest"] = (probs >= t_best).astype(int)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"Appended pred_NEW1_Pbest (t={t_best:.2f}) → {OUT_CSV}")

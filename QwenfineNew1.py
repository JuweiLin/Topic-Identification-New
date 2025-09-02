# qwen_new1_single.py
# pip install torch sentence-transformers scikit-learn pandas numpy

import os, json, math, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)


os.environ["TOKENIZERS_PARALLELISM"] = "True"

# =========================
# 配置
# =========================
TRAIN_JSONL = ["HSTN_JSONL/coded_n.jsonl","HSTN_JSONL/coded_tb.jsonl"]
VAL_JSONL   = ["HSTN_JSONL/val.jsonl"]
OUT_DIR     = "./qwen_mlp_ckpt"
PRED_CSV    = "./qwen_mlp_ckpt/val_pred_new1.csv"

MODEL_NAME  = "Qwen/Qwen3-Embedding-0.6B"
CTX_MODE    = "both"     # "both" | "prev" | "next" | "none"
USE_SWITCH  = True       # 说话人切换特征（基于 speaker_id）
USE_TRAN_Y  = True       # 优先用 JSONL 的 tran 作为监督目标
# =========================
# 配置（新增）
# =========================
PRE_WINDOW = 0          # 向上看多少句
NEXT_WINDOW = 0         # 向下看多少句
EMB_ENC_BATCH = 128      # 长文本建议把 encode 的 batch 缩小，避免显存爆
MAX_CHAR_PER_UTT = 0     # 每句裁剪到多少字符（0 表示不裁剪）
SEED        = 42
BATCH_SIZE  = 32
EVAL_BATCH  = 32
EPOCHS      = 12
LR          = 2e-5
DROPOUT     = 0.1
HIDDEN      = 512
FOCAL_GAMMA = 1.8
BALANCE     = "focal"    # "none" | "focal" | "sampler"
L2_NORM     = True

# === 评估开关（新增） ===
EVAL_P_MIN = 0.30     # Precision 目标下界
EVAL_P_MAX = 0.50     # Precision 目标上界
EVAL_P_STEP = 0.01    # 步长
EVAL_MIN_PRED = 20    # 至少命中这么多正类预测才算数（防止阈值过高只出很少正例）
EVAL_PRINT_P_TARGETS = [0.30, 0.40, 0.50]  # 控制台摘要展示哪些目标

# =========================
# 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def _canon(lbl):
    if lbl is None: return None
    return str(lbl).strip().upper().replace("$","").replace("_","")

def label_to_binary(lbl):
    s = _canon(lbl)
    return None if s is None else (1 if s == "NEW1" else 0)

def collect_dialogs(paths, pre_k=PRE_WINDOW, next_k=NEXT_WINDOW):
    """
    展平对话为逐句样本，并附带“上一窗口/下一窗口”的文本列表。
    JSONL 每行形如：
      {
        "dialogue_id": "...",
        "utterances": [{uid, speaker, speaker_id, content}, ...],
        "labels": ["NEW1", ...],
        "tran": [0/1, ...]   # 可选
      }
    """
    rows = []
    for p in paths:
        for d in read_jsonl(p):
            did   = d["dialogue_id"]
            utts  = d["utterances"]
            labs  = d.get("labels", [None]*len(utts))
            trans = d.get("tran",   [None]*len(utts))
            n = len(utts)
            for i, u in enumerate(utts):
                # 取窗口（保持时间顺序：更早的在前）
                s_pre  = max(0, i - pre_k)
                e_next = min(n, i + 1 + next_k)
                prev_list = [utts[j]["content"] for j in range(s_pre, i)]
                next_list = [utts[j]["content"] for j in range(i+1, e_next)]

                rows.append({
                    "dialogue_id": did,
                    "idx": i,
                    "uid": u.get("uid"),
                    "speaker": u.get("speaker"),
                    "speaker_id": u.get("speaker_id"),
                    "text": u.get("content"),
                    # 单句的 prev/next 仍然保留（兼容旧逻辑）
                    "prev": utts[i-1]["content"] if i>0 else None,
                    "next": utts[i+1]["content"] if i+1<n else None,
                    # 新增窗口列表
                    "prev_list": prev_list,
                    "next_list": next_list,
                    "label": labs[i],
                    "tran": trans[i] if i < len(trans) else None
                })
    return rows

def _clip_text(s):
    if s is None: return None
    if MAX_CHAR_PER_UTT and MAX_CHAR_PER_UTT > 0:
        return s[:MAX_CHAR_PER_UTT]
    return s

def compose_text(r, ctx="both"):
    """
    将窗口内的多句拼接成一个编码文本：
      PREV: <p[-k]> || ... || <p[-1]> | S{sid}: <curr> | NEXT: <n[+1]> || ... || <n[+k]>
    邻句不加说话人，只拼纯文本；中心句带说话人标签。
    """
    sid = r.get("speaker_id")
    spk_tag = f"S{sid}" if sid is not None else "SPK"
    seg = []

    if ctx in ("prev", "both"):
        prevs = r.get("prev_list")
        if not prevs:
            # 兼容没有 prev_list 的情况
            prevs = [r["prev"]] if r.get("prev") else []
        if prevs:
            prevs = [_clip_text(x) for x in prevs if x is not None]
            if prevs:
                seg.append("PREV: " + " || ".join(prevs))

    # 中心句
    center = _clip_text(r.get("text"))
    seg.append(f"{spk_tag}: {center}" if center is not None else f"{spk_tag}:")

    if ctx in ("next", "both"):
        nexts = r.get("next_list")
        if not nexts:
            nexts = [r["next"]] if r.get("next") else []
        if nexts:
            nexts = [_clip_text(x) for x in nexts if x is not None]
            if nexts:
                seg.append("NEXT: " + " || ".join(nexts))

    return " | ".join(seg)


def l2norm(X, eps=1e-9):
    X = np.asarray(X)
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

def target_from_row(r):
    """
    标签优先级：
      - 若 USE_TRAN_Y 且 r['tran'] in {0,1} → 用它
      - 否则回退到 labels 的 NEW1/非NEW1 二值化
    """
    if USE_TRAN_Y and r.get("tran") in (0,1):
        return int(r["tran"])
    return label_to_binary(r.get("label"))

def sweep_precision_targets(y_true, probs, p_min=0.30, p_max=0.50, p_step=0.01, min_pred=20):
    """
    依次对 Precision 目标（p_min..p_max，步长 p_step）找阈值 t，
    要求 Precision >= 目标且 Recall 最大；若达不到目标，取 Precision 最高的 t。
    返回 dict: key 是整数百分数（如 30/31/.../50），值是 metrics dict（含 t、P/R/F1/ACC/TP/FP/TN/FN）。
    """
    targets = np.arange(p_min, p_max + 1e-9, p_step)
    out = {}
    for tgt in targets:
        t = find_threshold_for_precision(y_true, probs, target=tgt, min_pred=min_pred)
        m = metrics_at_threshold(y_true, probs, t)
        m["t"] = float(t)
        out[int(round(tgt * 100))] = m
    return out

# =========================
# 数据集
# =========================
class New1Dataset(Dataset):
    def __init__(self, rows, embedder, ctx="both", l2=True, add_switch=True, require_labels=False):
        self.rows_all = rows
        self.embedder = embedder

        # 1) 展开文本与标签
        texts, y_list = [], []
        for r in rows:
            texts.append(compose_text(r, ctx))        # 需要你已有的 compose_text
            y_list.append(target_from_row(r))         # 需要你已有的 target_from_row

        # 2) 在完整对话上先算 switch（与过滤无关）
        switch_all = []
        last_spk = {}
        for r in rows:
            did = r["dialogue_id"]
            sid = r.get("speaker_id")
            cur = int(sid) if sid is not None else -1
            prev = last_spk.get(did, None)
            switch_all.append(1.0 if (prev is not None and cur != prev) else 0.0)
            last_spk[did] = cur

        # 3) 选择保留的索引
        if require_labels:
            keep = [i for i, y in enumerate(y_list) if y in (0, 1)]
        else:
            keep = list(range(len(rows)))

        # 4) 生成嵌入（仅对 keep 的文本）
        texts_kept = [texts[i] for i in keep]
        X = self.embedder.encode(
            texts_kept, batch_size=EMB_ENC_BATCH,
            normalize_embeddings=False, show_progress_bar=True
        )
        X = l2norm(X) if l2 else np.asarray(X)
        self.X = torch.from_numpy(np.asarray(X)).float()

        # 5) 说话人切换特征 & 标签（按 keep 切片）
        if add_switch:
            sw = [switch_all[i] for i in keep]
            self.switch = torch.tensor(sw, dtype=torch.float32).unsqueeze(1)
        else:
            self.switch = None

        y_kept = [(y_list[i] if y_list[i] in (0, 1) else -1) for i in keep]
        self.y = torch.tensor(y_kept, dtype=torch.long)
        self.has_labels = all(y in (0, 1) for y in y_kept)

        # 6) 记录对应的行（便于导出/对齐）
        self.rows = [rows[i] for i in keep]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.X[i]
        if self.switch is not None:
            x = torch.cat([x, self.switch[i]], dim=-1)
        y = self.y[i]
        return x, y


# =========================
# 模型 & 损失
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # [B, D] -> [B]
        return self.net(x).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.8, alpha=None, reduction="mean"):
        super().__init__(); self.g=gamma; self.alpha=alpha; self.red=reduction
    def forward(self, logits, y):  # y in {0,1}
        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        ce = -(y*torch.log(p) + (1-y)*torch.log(1-p))
        pt = torch.where(y==1, p, 1-p)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and self.alpha.numel()==2:
                w = torch.where(y==1, self.alpha[1], self.alpha[0])
            else:
                w = torch.where(y==1, self.alpha, 1-self.alpha)
        else:
            w = 1.0
        return (w * ((1-pt)**self.g) * ce).mean()

@torch.no_grad()
def evaluate(model, dl, device, thresh=0.5, criterion=None):
    model.eval()
    y_true_list, prob_list, val_losses = [], [], []

    for x, y in dl:
        x = x.to(device)
        logits = model(x)                              # logits 保持在 device 上
        probs_batch = torch.sigmoid(logits).detach().cpu().numpy()  # 仅用于指标的概率搬到 CPU

        y_np = y.numpy()
        m_np = (y_np >= 0)                             # numpy 掩码用于采样有标签样本
        if m_np.any():
            # 收集用于指标计算的数据（在 CPU 上走）
            y_true_list.append(y_np[m_np])
            prob_list.append(probs_batch[m_np])

            # —— 这里改动：在同一设备上计算 val loss ——
            if criterion is not None:
                m_t = torch.from_numpy(m_np).to(device)      # bool mask → device
                y_dev = y.to(device).float()                 # 标签搬到 device
                loss = criterion(logits[m_t], y_dev[m_t])    # 全部在 device 上
                val_losses.append(float(loss.item()))

    if not y_true_list:
        return None, None, None, None

    y_true = np.concatenate(y_true_list, 0)
    probs  = np.concatenate(prob_list, 0)
    preds  = (probs >= thresh).astype(int)

    P,R,F1,_ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        AUC = roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        AUC = float("nan")
    try:
        AP = average_precision_score(y_true, probs) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        AP = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    ACC = (tp + tn) / max(tp + tn + fp + fn, 1)

    metrics = {
        "P": float(P), "R": float(R), "F1": float(F1),
        "ACC": float(ACC), "AUC": float(AUC), "AP": float(AP),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "VAL_LOSS": float(np.mean(val_losses)) if val_losses else None
    }
    return metrics, y_true, probs, preds

def find_threshold_for_precision(y_true, probs, target=0.80, min_pred=20):
    """
    在验证集上找一个阈值 t，使 Precision >= target，并在满足条件的阈值中 Recall 最大。
    若没有任何阈值达标，则返回 Precision 最高的阈值。
    """
    thr_grid = np.linspace(0.99, 0.01, 99)
    best = {"t": None, "P": 0.0, "R": 0.0}
    # 先找满足 Precision 的
    for t in thr_grid:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        if tp + fp < max(min_pred, 1):
            continue
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if P >= target:
            if (R > best["R"]) or (R == best["R"] and P > best["P"]):
                best = {"t": float(t), "P": float(P), "R": float(R)}
    if best["t"] is not None:
        return best["t"]
    # 否则返回 Precision 最高的
    bestP = {"t": 0.5, "P": -1.0}
    for t in thr_grid:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        if tp + fp < max(min_pred, 1):
            continue
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if P > bestP["P"]:
            bestP = {"t": float(t), "P": float(P)}
    return bestP["t"]

def metrics_at_threshold(y_true, probs, t):
    pred = (probs >= t).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    ACC = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"P": float(P), "R": float(R), "F1": float(F1), "ACC": float(ACC),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}



# =========================
# 训练 & 推理
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) 读数据
    train_rows = collect_dialogs(TRAIN_JSONL)
    print(f"Train samples: {len(train_rows)}")
    val_rows = collect_dialogs(VAL_JSONL) if len(VAL_JSONL)>0 else []
    if val_rows: print(f"Val samples: {len(val_rows)}")

    # 2) 嵌入器
    print("Loading embedder:", MODEL_NAME)
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    # 3) 数据集
    ds_tr = New1Dataset(train_rows, embedder, ctx=CTX_MODE, l2=L2_NORM, add_switch=USE_SWITCH, require_labels=True)
    dl_val = None
    if val_rows:
        ds_va = New1Dataset(val_rows, embedder, ctx=CTX_MODE, l2=L2_NORM, add_switch=USE_SWITCH, require_labels=False)
        dl_val = DataLoader(ds_va, batch_size=EVAL_BATCH, shuffle=False, num_workers=0, pin_memory=True)


    in_dim = ds_tr.X.shape[1] + (1 if USE_SWITCH else 0)
    print("Input dim:", in_dim)

    # 4) 不平衡处理
    y = ds_tr.y.numpy()
    pos = int((y==1).sum()); neg = int((y==0).sum())
    print(f"Class counts -> pos(NEW1)={pos}  neg={neg}")

    if BALANCE == "sampler":
        w_pos = 0.5 / max(pos,1); w_neg = 0.5 / max(neg,1)
        weights = np.where(y==1, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(y), replacement=True
        )
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
        alpha = None
    else:
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        if BALANCE == "focal":
            a_pos = 1.0/math.sqrt(max(pos,1)); a_neg = 1.0/math.sqrt(max(neg,1))
            s = a_pos + a_neg
            alpha = torch.tensor([a_neg/s, a_pos/s], dtype=torch.float32, device=device)
        else:
            alpha = None

    # 5) 模型
    model = MLP(in_dim=in_dim, hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha).to(device)

    history = []
    os.makedirs(OUT_DIR, exist_ok=True)
    best_f1, best_state, best_thresh = -1.0, None, 0.5
    history = []
    os.makedirs(OUT_DIR, exist_ok=True)
    best_f1, best_state, best_thresh = -1.0, None, 0.5
    history = []
    ckpt = os.path.join(OUT_DIR, "qwen_new1_mlp.pt")

    for ep in range(1, EPOCHS+1):
        # ---- Train ----
        model.train(); losses = []
        for x, yb in dl_tr:
            x = x.to(device); yb = yb.to(device).float()
            opt.zero_grad()
            logit = model(x)
            loss = crit(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        # ---- Eval every epoch ----
        metrics05, y_true_lab, probs_lab, _ = evaluate(model, dl_val, device, 0.5, criterion=crit)
        # 扫阈值找 best F1
        best_local = (metrics05["F1"], 0.5)
        for t in np.linspace(0.01, 0.99, 99):
            preds_t = (probs_lab >= t).astype(int)
            P, R, F1, _ = precision_recall_fscore_support(y_true_lab, preds_t, average="binary", zero_division=0)
            if F1 > best_local[0]:
                best_local = (float(F1), float(t))
        f1_star, t_star = best_local
        preds_star = (probs_lab >= t_star).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_lab, preds_star, labels=[0,1]).ravel()
        acc_star = (tp + tn) / max(tp + tn + fp + fn, 1)

        # 按 30%~50% 逐一找阈值（写入 CSV；控制台只摘要打印 30/40/50）
        sweep = sweep_precision_targets(
            y_true_lab, probs_lab,
            p_min=EVAL_P_MIN, p_max=EVAL_P_MAX, p_step=EVAL_P_STEP, min_pred=EVAL_MIN_PRED
        )

        # ---- 打印 ----
        print(f"[Epoch {ep}] train_loss={train_loss:.4f}")
        print(
            f"  val @0.5  P={metrics05['P']:.3f} R={metrics05['R']:.3f} F1={metrics05['F1']:.3f} "
            f"ACC={metrics05['ACC']:.3f} AUC={metrics05['AUC']:.3f} AP={metrics05['AP']:.3f} | "
            f"Conf@0.5 TP={metrics05['TP']} FP={metrics05['FP']} TN={metrics05['TN']} FN={metrics05['FN']} "
            f"| VAL_LOSS={metrics05['VAL_LOSS'] if metrics05['VAL_LOSS'] is not None else float('nan'):.4f}"
        )
        print(
            f"  bestF1={f1_star:.3f} @ t={t_star:.2f} | "
            f"Conf(t*) TP={tp} FP={fp} TN={tn} FN={fn} | ACC={acc_star:.3f}"
        )
        # 摘要打印 30/40/50 三个目标
        parts = []
        for p_tgt in EVAL_PRINT_P_TARGETS:
            key = int(round(p_tgt * 100))
            if key in sweep:
                m = sweep[key]
                parts.append(f"P≥{key}%: t={m['t']:.2f} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} (TP={m['TP']},FP={m['FP']})")
        if parts:
            print("  precision targets → " + " | ".join(parts))

        # ---- Save best by F1 ----
        if f1_star > best_f1:
            best_f1, best_thresh = f1_star, t_star
            best_state = {
                "model": model.state_dict(),
                "in_dim": in_dim,
                "thresh": best_thresh,
                "cfg": {
                    "MODEL_NAME": MODEL_NAME,
                    "CTX_MODE": CTX_MODE,
                    "USE_SWITCH": USE_SWITCH,
                    "L2_NORM": L2_NORM,
                    "HIDDEN": HIDDEN,
                    "DROPOUT": DROPOUT,
                    "USE_TRAN_Y": USE_TRAN_Y
                }
            }
            # torch.save(best_state, ckpt)
            # print(f"[+] Saved best checkpoint → {ckpt}")

        # ---- Log row（写 CSV 用）----
        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": metrics05["VAL_LOSS"],
            "P@0.5": metrics05["P"], "R@0.5": metrics05["R"], "F1@0.5": metrics05["F1"], "ACC@0.5": metrics05["ACC"],
            "AUC": metrics05["AUC"], "AP": metrics05["AP"],
            "TP@0.5": int(metrics05["TP"]), "FP@0.5": int(metrics05["FP"]),
            "TN@0.5": int(metrics05["TN"]), "FN@0.5": int(metrics05["FN"]),
            "bestF1": f1_star, "best_t": t_star,
            "TP@t*": int(tp), "FP@t*": int(fp), "TN@t*": int(tn), "FN@t*": int(fn),
            "ACC@t*": acc_star
        }
        # 把 30%~50% 的所有目标也摊平成列
        for key_pct, m in sweep.items():
            row[f"t@P{key_pct}"]  = m["t"]
            row[f"P@P{key_pct}"]  = m["P"]
            row[f"R@P{key_pct}"]  = m["R"]
            row[f"F1@P{key_pct}"] = m["F1"]
            row[f"ACC@P{key_pct}"]= m["ACC"]
            row[f"TP@P{key_pct}"] = m["TP"]
            row[f"FP@P{key_pct}"] = m["FP"]
            row[f"TN@P{key_pct}"] = m["TN"]
            row[f"FN@P{key_pct}"] = m["FN"]

        history.append(row)

    # 训练结束：写盘
    if history:
        df = pd.DataFrame(history)
        csv_path = os.path.join(OUT_DIR, "train_log.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print("Saved log →", csv_path)




if __name__ == "__main__":
    main()

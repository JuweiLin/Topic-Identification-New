# linear_head_with_emb.py
# pip install scikit-learn sentence-transformers numpy pandas torch tqdm

import os, re, json, math
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from sentence_transformers import SentenceTransformer

import re

# ---------------- 固定路径 ----------------
TRAIN_JSONL = ["HSTN_JSONL/coded_n.jsonl", "HSTN_JSONL/coded_tb.jsonl"]
VAL_JSONL   = ["HSTN_JSONL/val.jsonl"]
OUT_DIR     = "linear_head_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 配置 ----------------
MODEL_NAME   = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LEN  = 256
SEED         = 42
K_LOCAL      = 3      # novel_local 的历史窗口大小（用最近 K 句的“上一句”向量做质心）

BACKCHANNEL = set(["ok","okay","ok.","ok!","okk","k","kk","y","yy","yes","yeah","yep","nope","uh","um","right","sure","mm"])
STOP = set(["the","a","an","of","and","to","is","are","am"])

_WH = {"who","whom","whose","which","what","why","how","where","when"}
_AUX = {"do","does","did","is","are","am","was","were","can","could","will","would",
        "shall","should","may","might","have","has","had"}
_SUBJ = {"i","you","we","he","she","they","it","there","this","that","anyone","someone","people"}

# 常见询问短语（ anywhere 匹配 ）
_PHRASES = [
    r"\bi wonder if\b", r"\bany idea\b", r"\bis it possible\b",
    r"\bcould you\b", r"\bcan you\b", r"\bwould you\b", r"\bshould we\b",
    r"\bshall we\b", r"\bdo you\b", r"\bdoes it\b", r"\bis there\b", r"\bare there\b",
    r"\bwould it be\b", r"\bcould we\b", r"\bwould we\b",
]

_CLAUSE_SPLIT = re.compile(r"\b(?:and|but|so|then|or|if|because|since|when|while|though|although|whereas)\b", re.I)

_PHRASE_RE = re.compile("|".join(_PHRASES), re.I)
_OR_NOT_RE = re.compile(r"\bor not\b", re.I)

def _tokens(s: str):
    # 只保留英文字符和空格；统一小写
    s = re.sub(r"[^A-Za-z\s]", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s, s.split()

def question_score(text: str) -> float:
    """英文无标点问句打分：0~1；对子句取最大分。"""
    if not text:
        return 0.0

    s_norm, _ = _tokens(text)
    if not s_norm:
        return 0.0

    score = 0.0
    # 子句切分（删除连接词保留两侧片段）
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(s_norm) if c.strip()]
    if not clauses:
        clauses = [s_norm]

    for c in clauses:
        _, toks = _tokens(c)
        if not toks:
            continue

        # 1) WH 词在任意位置（不要求句首）
        if any(t in _WH for t in toks):
            score = max(score, 0.72)

        # 2) 助动词 + 主语 顺序共现（倒装/一般疑问）
        #    在合并句中不要求相邻，这里用滑动窗口宽度 5 近邻匹配更稳
        win = 5
        for i, t in enumerate(toks):
            if t in _AUX:
                window = toks[i+1:i+1+win]
                if any(w in _SUBJ for w in window):
                    score = max(score, 0.65)
                    break

        # 3) 固定询问短语 / “or not”
        if _PHRASE_RE.search(c) or _OR_NOT_RE.search(c):
            score = max(score, 0.62)

        # 4) 极短问句抑制（例如单词碎片），防止误报
        if len(toks) <= 2:
            score = min(score, 0.55)

    return float(score)

def set_seed(s=SEED):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def collect_rows(paths):
    rows=[]
    for p in paths:
        for d in read_jsonl(p):
            did = d["dialogue_id"]
            utts = d["utterances"]
            labs = d.get("labels", [None]*len(utts))
            trans= d.get("tran",   [None]*len(utts))
            for i,u in enumerate(utts):
                rows.append({
                    "dialogue_id": did,
                    "idx": i,
                    "uid": u.get("uid"),
                    "speaker_id": u.get("speaker_id"),
                    "text": u.get("content") or "",
                    "prev": utts[i-1]["content"] if i>0 else "",
                    "label": labs[i],
                    "tran":  trans[i] if i < len(trans) else None
                })
    return rows

def _canon(lbl):
    if lbl is None: return None
    return str(lbl).strip().upper().replace("$","").replace("_","")

def to_y(r):
    if r.get("tran") in (0,1): return int(r["tran"])
    s = _canon(r.get("label"))
    if s is None: return None
    return 1 if s=="NEW1" else 0

def simple_tokens(t):
    t = (t or "").lower().strip()
    t = re.sub(r"https?://\S+|@\w+|#[\w-]+", " ", t)
    toks = re.findall(r"[a-zA-Z]+|\d+|[\u4e00-\u9fa5]", t)
    content = [w for w in toks if w not in STOP]
    return toks, content

def jaccard(a,b):
    A,B = set(a), set(b)
    return 0.0 if not A and not B else len(A&B)/max(len(A|B),1)

def l2norm(X, eps=1e-9):
    X = np.asarray(X); n = np.linalg.norm(X, axis=1, keepdims=True)+eps
    return X / n

def build_split(rows, embedder):
    # ---- 说话人切换（按对话遍历）----
    switch_all=[]; last={}
    for r in rows:
        did=r["dialogue_id"]; sid=r.get("speaker_id")
        cur = int(sid) if sid is not None else -1
        prev= last.get(did, None)
        switch_all.append(1.0 if (prev is not None and cur!=prev) else 0.0)
        last[did]=cur

    # ---- 基础字段 ----
    y = []
    texts=[]; prevs=[]; sids=[]
    len_char=[]; is_short=[]; is_back=[]; q_flag=[]
    toks_prev_list=[]; toks_cur_list=[]
    dids=[]

    for r in rows:
        y.append(to_y(r))
        cur = r["text"]; prv = r["prev"]; did=r["dialogue_id"]
        texts.append(cur); prevs.append(prv); sids.append(r.get("speaker_id")); dids.append(did)

        t = cur.strip().lower()
        len_char.append(math.log1p(len(t)))
        is_short.append(1 if len(t)<=2 else 0)
        is_back.append(1 if (len(t)<=6 and t in BACKCHANNEL) else 0)
        q_flag.append(question_score(cur))

        _, cp = simple_tokens(prv)
        _, cc = simple_tokens(cur)
        toks_prev_list.append(cp); toks_cur_list.append(cc)

    y = np.array([(-1 if v is None else int(v)) for v in y], dtype=int)
    speaker_switch = np.array(switch_all, dtype=np.float32)

    # ---- 稀疏相似度 & jaccard（当前 vs 上一句）----
    vec = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))
    A = vec.fit_transform(prevs)
    B = vec.transform(texts)
    num = (A.multiply(B)).sum(axis=1).A.ravel()
    den = np.sqrt((A.multiply(A)).sum(axis=1).A.ravel() * (B.multiply(B)).sum(axis=1).A.ravel()) + 1e-9
    tfidf_cos = (num/den).astype(np.float32)
    jacc = np.array([jaccard(toks_prev_list[i], toks_cur_list[i]) for i in range(len(rows))], dtype=np.float32)

    # ---- 嵌入：cur / prev 向量 + delta_cos + local_sim ----
    cur_tagged  = [f"S{sids[i] if sids[i] is not None else 'PK'}: {texts[i]}" for i in range(len(rows))]
    prev_tagged = [f"PREV: {prevs[i]}" for i in range(len(rows))]
    C = embedder.encode(cur_tagged,  batch_size=128, normalize_embeddings=False, show_progress_bar=True)
    P = embedder.encode(prev_tagged, batch_size=128, normalize_embeddings=False, show_progress_bar=True)
    C = l2norm(C).astype(np.float32); P = l2norm(P).astype(np.float32)

    # delta_cos（越大越新）
    delta_cos = (1.0 - (C*P).sum(1)).astype(np.float32)

    # local_sim（与历史质心相似；仅用已发生的“上一句”向量，避免泄漏）
    local_sim = np.zeros(len(rows), dtype=np.float32)
    last_did=None; buf=[]
    for i in range(len(rows)):
        did = dids[i]
        if did!=last_did:
            buf=[]; last_did=did
        if buf:
            mean_vec = np.mean(buf,axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec)+1e-9)
            local_sim[i] = float(np.dot(C[i], mean_vec))
        else:
            local_sim[i] = 0.0
        buf.append(P[i])             # 只用“上一句”的嵌入推进历史
        if len(buf) > K_LOCAL:
            buf.pop(0)

    # novel_* 取反
    novel_tfidf   = 1.0 - tfidf_cos
    novel_jaccard = 1.0 - jacc
    novel_local   = 1.0 - local_sim

    # ---- 9 个手工特征（新增 is_question）----
    feats = {
        "len_char":        np.array(len_char, np.float32),
        "not_short":       1.0 - np.array(is_short, np.float32),
        "not_back":        1.0 - np.array(is_back,  np.float32),
        "speaker_switch":  speaker_switch,
        "delta_cos":       delta_cos,
        "novel_tfidf":     novel_tfidf,
        "novel_jaccard":   novel_jaccard,
        "novel_local":     novel_local,
        "is_question":     np.array(q_flag, np.float32),   # <<<<<< 新增
    }
    feat_names = list(feats.keys())
    X_small = np.column_stack([feats[k] for k in feat_names])

    meta = {
        "dialogue_id":np.array([r["dialogue_id"] for r in rows]),
        "idx":        np.array([r["idx"] for r in rows]),
        "text":       np.array([r["text"] for r in rows]),
        "prev":       np.array([r["prev"] for r in rows]),
    }
    return X_small, y, feat_names, feats, meta

# -------- 单特征探针（输出混淆矩阵等）--------
def probe_one_feature(y, x, name, split):
    y = np.asarray(y); x = np.asarray(x, dtype=float)
    m = np.isin(y, [0,1])
    y, x = y[m], x[m]
    out = {"feature": name, "split": split, "support": int(len(y))}
    if len(y)==0 or len(np.unique(y))<2:
        out.update(dict(ROC_AUC=np.nan, PR_AUC=np.nan, F1_best=np.nan, t_star=np.nan,
                        TP=0,FP=0,TN=0,FN=0))
        return out

    out["ROC_AUC"] = float(roc_auc_score(y, x))
    out["PR_AUC"]  = float(average_precision_score(y, x))

    P, R, T = precision_recall_curve(y, x)
    F1s = 2*P[1:]*R[1:]/(P[1:]+R[1:]+1e-9)
    idx = int(np.nanargmax(F1s)) if len(F1s)>0 else 0
    t_star = float(T[idx]) if len(T)>0 else float(np.nan)

    pred = (x >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out.update(dict(
        F1_best=float(F1s[idx]) if len(F1s)>0 else float("nan"),
        t_star=t_star, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn)
    ))
    return out

def run_feature_probe(feat_names, feats_tr, y_tr, feats_va, y_va):
    rows = []
    for name in feat_names:
        rows.append(probe_one_feature(y_tr, feats_tr[name], name, "train"))
        rows.append(probe_one_feature(y_va, feats_va[name], name, "val"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "feature_probe.csv"), index=False, encoding="utf-8-sig")
    print("Saved →", os.path.join(OUT_DIR, "feature_probe.csv"))
    return df

# -------- 线性头训练 + 验证 --------
def fit_and_eval(X_tr, y_tr, X_va, y_va, feat_names, meta_va):
    mtr = np.isin(y_tr,[0,1]); mva = np.isin(y_va,[0,1])
    X_tr, y_tr = X_tr[mtr], y_tr[mtr]
    X_va, y_va = X_va[mva], y_va[mva]

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            penalty="l2", solver="saga", C=1.0,
            class_weight="balanced", max_iter=2000, random_state=SEED))
    ])
    clf.fit(X_tr, y_tr)

    scores = clf.predict_proba(X_va)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_va, scores)
    f1s = 2*precisions[1:]*recalls[1:]/(precisions[1:]+recalls[1:]+1e-9)
    best_idx = int(np.nanargmax(f1s))
    t_star = float(thresholds[best_idx])

    def report_at(th, tag):
        pred = (scores >= th).astype(int)
        P,R,F1,_ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_va, pred, labels=[0,1]).ravel()
        ACC = (tp + tn) / max(tp+tn+fp+fn, 1)
        AUC = roc_auc_score(y_va, scores) if len(np.unique(y_va))==2 else float("nan")
        AP  = average_precision_score(y_va, scores) if len(np.unique(y_va))==2 else float("nan")
        print(f"[{tag}] th={th:.4f}  P={P:.3f} R={R:.3f} F1={F1:.3f} ACC={ACC:.3f} AUC={AUC:.3f} AP={AP:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}")
        return {"tag":tag,"th":th,"P":P,"R":R,"F1":F1,"ACC":ACC,"AUC":AUC,"AP":AP,"TP":tp,"FP":fp,"TN":tn,"FN":fn}

    print("\n=== Validation (linear head on 9 features) ===")
    _ = report_at(t_star,  "bestF1@t*")
    _ = report_at(0.5,     "fixed@0.5")

    # 导出验证预测
    pred_star  = (scores >= t_star).astype(int)
    pred_fixed = (scores >= 0.5).astype(int)
    out = pd.DataFrame({
        "dialogue_id": meta_va["dialogue_id"][mva],
        "idx":         meta_va["idx"][mva],
        "text":        meta_va["text"][mva],
        "prev":        meta_va["prev"][mva],
        "y_true":      y_va,
        "score":       scores,
        "pred_t*":     pred_star,
        "pred_0.5":    pred_fixed
    })
    out.to_csv(os.path.join(OUT_DIR, "val_pred_linear.csv"), index=False, encoding="utf-8-sig")
    print("Saved →", os.path.join(OUT_DIR, "val_pred_linear.csv"))

    # 特征权重
    lr = clf.named_steps["lr"]; W = lr.coef_.ravel()
    feat_imp = pd.DataFrame({"feature":feat_names, "weight":W})
    feat_imp.to_csv(os.path.join(OUT_DIR, "feature_weights.csv"), index=False, encoding="utf-8-sig")
    print("Saved →", os.path.join(OUT_DIR, "feature_weights.csv"))

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading embedder:", MODEL_NAME, "on", device)
    embedder = SentenceTransformer(MODEL_NAME, device=device)
    try: embedder.max_seq_length = MAX_SEQ_LEN
    except: pass

    train_rows = collect_rows(TRAIN_JSONL)
    val_rows   = collect_rows(VAL_JSONL)
    print(f"Train samples: {len(train_rows)} | Val samples: {len(val_rows)}")

    X_tr, y_tr, feat_names, feats_tr, _       = build_split(train_rows, embedder=embedder)
    X_va, y_va, feat_names_va, feats_va, meta = build_split(val_rows,   embedder=embedder)
    assert feat_names == feat_names_va

    # ① 单特征探针（含混淆矩阵）
    _ = run_feature_probe(feat_names, feats_tr, y_tr, feats_va, y_va)

    # ② 线性头训练 + 验证
    fit_and_eval(X_tr, y_tr, X_va, y_va, feat_names, meta)

if __name__ == "__main__":
    main()

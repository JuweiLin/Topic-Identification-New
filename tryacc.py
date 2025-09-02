# rule_ti_try.py
import re, math
import pandas as pd
import numpy as np
from collections import defaultdict

# ===== 词表/启发式 =====
BACKCHANNEL = {
    "uh", "uhh", "uh-huh", "huh", "hmm", "hmmm",
    "yeah", "yea", "yep", "ok", "okay", "right", "sure",
    "mm", "mmm", "oh", "wow", "alright", "fine", "good",
}
SUMMATIVE_HINTS = {
    # 总结/收束类口头语（topic-closing 代理）
    "anyway", "so yeah", "so that’s it", "long story short",
    "in the end", "that’s about it", "that’s it", "it was really", "let me know if",
}

DISCONTINUITY_PREFIX = (
    r"^(so|well|anyway|by the way|btw|but|and)\b"  # and-prefaced 也可能标记新题
)
NEWS_ELICIT = (r"^(what(’|')?s new|what is new)\b", r"^tell me about\b")
NEWS_ANN   = (r"^(you(’|')?ll never guess|guess what)\b",)

PRONOUN_START = r"^(he|she|they|it|this|that|these|those)\b"
VERB_LIKE = {"is","are","was","were","be","am","do","does","did","have","has","had","go","went","goes","say","says","said","think","thinks","thought"}

TOKEN_RE = re.compile(r"[A-Za-z']+")
def tokens(s: str):
    return [w.lower() for w in TOKEN_RE.findall(s or "")]

def is_backchannel(text: str):
    t = tokens(text)
    if not t: return True
    if len(t) <= 2 and all(w in BACKCHANNEL for w in t):
        return True
    return False

def content_len(text: str):
    t = [w for w in tokens(text) if w not in BACKCHANNEL]
    return len(t)

def jaccard(a, b):
    A, B = set(tokens(a)), set(tokens(b))
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def sim(curr, context):
    """可替换为嵌入余弦；这里先用 Jaccard 近似"""
    if not context: return 0.0
    return max(jaccard(curr, c) for c in context[-3:])  # 取最近3句的最大相似度

def detect_topic_closing(prev_utts):
    """
    近似：最近 2–3 轮多为 BACKCHANNEL + 可能出现总结性短语，即认为进入“可新启话题”区间
    """
    if len(prev_utts) < 2: return False
    last2 = prev_utts[-2:]
    cond_bc = all(is_backchannel(u) for u in last2)
    cond_sum = any(any(h in (u or "").lower() for h in SUMMATIVE_HINTS) for u in last2)
    return cond_bc or cond_sum

def detect_discontinuity(text):
    low = (text or "").strip().lower()
    if re.search(DISCONTINUITY_PREFIX, low): return True
    for pat in NEWS_ELICIT + NEWS_ANN:
        if re.search(pat, low): return True
    return False

def detect_cohesion_types(curr, prev_utts):
    """
    返回 {"lex":bool,"ref":bool,"ell":bool}
    - 词汇衔接：与上一话题有一定词面/语义相近，同时引入新词；
    - 指代衔接：代词起句，且上一窗口出现过实体名词；
    - 省略衔接：极短/无明显动词，需依赖前句恢复。
    """
    res = {"lex": False, "ref": False, "ell": False}
    if not prev_utts: return res
    prev_ctx = prev_utts[-3:]
    curr_toks = tokens(curr)

    # 词汇衔接：适中相似（非极高=续谈，非极低=突启），同时有“新词”引入
    s = sim(curr, prev_ctx)
    prev_all = set().union(*[set(tokens(p)) for p in prev_ctx])
    new_terms = [w for w in curr_toks if w not in prev_all and w not in BACKCHANNEL]
    if 0.2 <= s <= 0.65 and len(new_terms) >= 1:
        res["lex"] = True

    # 指代衔接：代词开头 + 历史里存在名词/专名
    if re.search(PRONOUN_START, (curr or "").strip().lower()):
        # 简易：只要历史非空就判定，有需要可加 NER
        res["ref"] = True

    # 省略衔接：句长很短 / 缺少显式动词
    if content_len(curr) <= 4 and not any(v in set(curr_toks) for v in VERB_LIKE):
        res["ell"] = True

    return res

def detect_nc_ti(curr, prev_utts):
    # 无任何机制且与前文相似度很低 → NC-TI（突启）
    if detect_discontinuity(curr): return False
    coh = detect_cohesion_types(curr, prev_utts)
    if any(coh.values()): return False
    return sim(curr, prev_utts) < 0.12

def partner_uptake(next_utts, curr_text):
    """
    伙伴接纳（1–2 轮内）：
    - 非纯 backchannel，且
    - 对 NEW1 有一定内容呼应（中断/衔接/相似度≥0.2/新闻问答）或“应答+内容长度>1”
    返回 (是否接纳, 用作 NEW2 的索引相对偏移)
    """
    curr = curr_text or ""
    for k, u in enumerate(next_utts[:2], start=1):
        if u is None: continue
        if not is_backchannel(u) and (detect_discontinuity(u) or
                                      any(detect_cohesion_types(u, [curr]).values()) or
                                      jaccard(u, curr) >= 0.2 or
                                      content_len(u) > 1):
            return True, k-1
    return False, None

def label_by_paper(df):
    """
    输入: df 包含 [dialogue_id, speaker_id, text]（来自你的 val_pred_new1.csv）
    输出: 新列:
      - ti_method: {topic_closing, discontinuity, cohesion_lex/ref/ell, noncoherent, none}
      - final_label: {NEW1, NEW2, SAME, BACK}
    """
    df = df.copy()
    df["ti_method"] = "none"
    df["final_label"] = "SAME"

    # 逐对话处理
    for did, sub in df.groupby("dialogue_id", sort=False):
        idxs = list(sub.index)
        speakers = sub["speaker_id"].tolist()
        texts = sub["text"].fillna("").tolist()

        # 1) 标注 TI 机制
        for i, gi in enumerate(idxs):
            prev_utts = texts[max(0, i-10):i]  # 向上窗口
            m = "none"
            if detect_discontinuity(texts[i]):
                m = "discontinuity"
            else:
                coh = detect_cohesion_types(texts[i], prev_utts)
                if coh["lex"]: m = "cohesion_lex"
                elif coh["ref"]: m = "cohesion_ref"
                elif coh["ell"]: m = "cohesion_ell"
                elif detect_topic_closing(prev_utts):
                    m = "topic_closing"
                elif detect_nc_ti(texts[i], prev_utts):
                    m = "noncoherent"
            df.at[gi, "ti_method"] = m

        # 2) 依论文“接纳”原则，落到 NEW1/NEW2/SAME/BACK
        i = 0
        while i < len(idxs):
            gi = idxs[i]
            spk = speakers[i]
            cur = texts[i]
            method = df.at[gi, "ti_method"]

            # 极短应答
            if is_backchannel(cur):
                df.at[gi, "final_label"] = "BACK"
                i += 1
                continue

            if method != "none":
                # 仅当“对方”在 1–2 轮内接纳，才认定新话题
                # 找接下来的对方话轮
                j_candidates = []
                for j in range(i+1, min(i+3, len(idxs))):
                    if speakers[j] != spk:
                        j_candidates.append(j)
                uptake, rel = partner_uptake([texts[j] for j in j_candidates], cur)
                if uptake:
                    df.at[gi, "final_label"] = "NEW1"
                    j_abs = j_candidates[rel]
                    df.at[idxs[j_abs], "final_label"] = "NEW2"
                    i = j_abs + 1
                    continue
                else:
                    # 未被接纳 → 不算新话题，回落 SAME/BACK
                    df.at[gi, "final_label"] = "SAME" if content_len(cur) > 1 else "BACK"
                    i += 1
                    continue
            else:
                # 无 TI 机制 → SAME/BACK
                df.at[gi, "final_label"] = "SAME" if content_len(cur) > 1 else "BACK"
                i += 1

    return df

# —— 用法示例 —— 
df = pd.read_csv("./qwen_mlp_ckpt/val_pred_new1.csv")
out = label_by_paper(df)
out.to_csv("./qwen_mlp_ckpt/val_pred_ti_rules.csv", index=False, encoding="utf-8-sig")
print(out["final_label"].value_counts(), "\n", out["ti_method"].value_counts())

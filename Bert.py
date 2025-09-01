# HSTN — Full Trainer (All Modules v1)
# -------------------------------------------------------------
# Implements your full architecture end-to-end:
# Utterance Encoder (DeBERTa) -> Dialogue Transformer (with RelDist/Speaker/Sim biases)
# -> Joint Heads: CRF + Boundary (CE + bias) + Cues (BCE, weak labels) + Segment (BCE)
# + Distillation on emissions (NEW*/OFF-focused)
# + Two-stage training (freeze encoder -> unfreeze top K) + metrics & confusion matrix
#
# Data: JSONL windows like
#   {"dialogue_id":"n40#1-200", "utterances":[{"uid":"u1","speaker":"INV","content":"..."}, ...],
#    "labels":["NEW1","SAME",...],  (optional) "teacher_logits": [[...5...], ...] }
# Padding on sentence axis is done in collate_fn; do NOT write PAD rows to JSON.
# -------------------------------------------------------------

import os, json, math, random, re
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torchcrf import CRF
from typing import Set

# -----------------------------
# Config
# -----------------------------
LABELS = ["SAME","NEW1","NEW2","BACK","OFF"]
LABEL2ID = {c:i for i,c in enumerate(LABELS)}
ID2LABEL = {i:c for c,i in LABEL2ID.items()}
SAME, NEW1, NEW2, BACK, OFF = [LABEL2ID[x] for x in LABELS]

CUES = ["discontinuity","topic_closing","cohesive_lex","reference_tie","ellipsis","off_domain","novelty"]
CUE2ID = {c:i for i,c in enumerate(CUES)}

@dataclass
class CFG:
    # data
    train_path: str = "./train.jsonl"
    dev_path: str   = "./val.jsonl"
    max_subwords: int = 128
    batch_size: int = 2
    # model
    model_name: str = "microsoft/deberta-v3-large"  # or roberta-large
    d_ctx: int = 512
    ctx_layers: int = 12
    ctx_heads: int = 8
    ffn_mult: int = 8
    dropout: float = 0.1
    # biases
    rel_max_dist: int = 128   # distance buckets
    sim_scale_init: float = 0.2
    speaker_gate_init: float = 0.1
    # losses
    w_main: float = 1.0
    w_boundary: float = 0.5
    w_cues: float = 0.2
    w_segment: float = 0.3
    w_distill: float = 0.5
    # distillation
    distill_T: float = 2.0
    distill_focus: Tuple[int,...] = field(default_factory=lambda: (NEW1, NEW2, OFF))
    # optim / schedule
    lr_enc_frozen: float = 0.0    # stage 1
    lr_enc_unfrozen: float = 2e-6 # stage 2
    lr_ctx: float = 1e-3
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 6
    grad_clip: float = 1.0
    seed: int = 42
    # staging
    freeze_epochs: int = 2
    unfreeze_top_k: int = 6
    # sampler
    sampler_alpha: float = 3.0

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Dataset & Collate
# -----------------------------
class HSTNDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if not obj.get('utterances') or not obj.get('labels'):
                    continue
                if len(obj['utterances']) != len(obj['labels']):
                    continue
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def speakers_to_ids(s: str) -> int:
    s = (s or '').upper()
    if s.startswith('INV'): return 0
    return 1  # PAR/other

# ---------- Weak cues (simple heuristics over text) ----------
TOK_SPLIT = re.compile(r"[\W_]+", re.UNICODE)
FILLERS = {"um","uh","mhm","uh-huh","mm-hmm","er","erm","you know","like"}
CLOSERS = {"anyway","so","well","okay","alright","move on","moving on","next"}
PRONOUNS = {"i","you","he","she","we","they","it","this","that","these","those"}


def jaccard(a:Set[str], b:Set[str]) -> float:
    if not a and not b: return 1.0
    i = len(a & b); u = len(a | b)
    return i / u if u>0 else 0.0



def cues_from_texts(texts: List[str]) -> np.ndarray:
    """Return [U, |CUES|] 0/1 pseudo labels from simple rules."""
    U = len(texts)
    cues = np.zeros((U, len(CUES)), dtype=np.float32)
    prev_sets: List[Set[str]] = []
    vocab_seen: Set[str] = set()

    for i, t in enumerate(texts):
        tokens = [w for w in TOK_SPLIT.split(t.lower()) if w]
        toks_set = set(tokens)
        prev_sets.append(toks_set)

        # fillers / ellipsis
        filler_ratio = sum(1 for w in tokens if w in FILLERS) / max(1,len(tokens))
        ellipsis = (len(tokens) <= 3) or ("..." in t)

        # closers
        closing = any(kw in t.lower() for kw in CLOSERS)

        # reference ties (pronouns)
        ref_tie = any(w in PRONOUNS for w in tokens)

        # discontinuity + novelty (vs prev 3)
        win_prev = prev_sets[max(0, i-3):i]
        if win_prev:
            sims = [jaccard(toks_set, s) for s in win_prev]
            sim_max = max(sims)
        else:
            sim_max = 1.0
        discontinuity = (sim_max < 0.2)
        new_vocab = len([w for w in toks_set if w not in vocab_seen]) / max(1,len(toks_set))
        novelty = (new_vocab > 0.5)

        # off-domain (very heuristic: many unseen tokens & long)
        off_domain = (new_vocab > 0.7 and len(toks_set) > 6)

        # cohesive lex (lexical overlap -> likely SAME)
        cohesive_lex = (sim_max > 0.6)

        vocab_seen |= toks_set

        cues[i, CUE2ID['discontinuity']] = float(discontinuity)
        cues[i, CUE2ID['topic_closing']] = float(closing)
        cues[i, CUE2ID['cohesive_lex']] = float(cohesive_lex)
        cues[i, CUE2ID['reference_tie']] = float(ref_tie)
        cues[i, CUE2ID['ellipsis']] = float(ellipsis or filler_ratio>0.2)
        cues[i, CUE2ID['off_domain']] = float(off_domain)
        cues[i, CUE2ID['novelty']] = float(novelty)
    return cues


def boundary_targets(labels: List[str]) -> np.ndarray:
    """Return [U] targets in {0:Stay,1:NEW1,2:NEW2,3:BACK,4:OFF}."""
    U = len(labels)
    out = np.zeros((U,), dtype=np.int64)
    for i in range(U):
        if i==0 or labels[i]==labels[i-1]:
            out[i] = 0  # Stay
        else:
            out[i] = LABEL2ID.get(labels[i], 0)  # use current class id
    return out


def segment_targets(labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-position start/end targets [U],[U] (0/1). Start at i where label != prev & label != SAME.
       End at i where label == SAME and prev != SAME, or at last position of a non-SAME run.
    """
    U = len(labels)
    start = np.zeros((U,), dtype=np.float32)
    end   = np.zeros((U,), dtype=np.float32)
    for i in range(U):
        cur = labels[i]
        prev = labels[i-1] if i>0 else 'SAME'
        nxt  = labels[i+1] if i<U-1 else 'SAME'
        if cur != prev and cur != 'SAME':
            start[i] = 1.0
        # end of a non-SAME stretch
        if cur != 'SAME' and (nxt == 'SAME' or i==U-1):
            end[i] = 1.0
    return start, end


def collate_fn(batch: List[Dict[str,Any]], tokenizer, max_subwords: int):
    B = len(batch)
    Umax = max(len(x['utterances']) for x in batch)
    L = max_subwords

    all_input_ids, all_attn, all_speakers, all_labels, all_mask_u = [], [], [], [], []
    all_cues, all_bound, all_seg_start, all_seg_end = [], [], [], []
    all_teacher, all_has_t = [], []

    for sample in batch:
        utts = sample['utterances']
        labels = sample['labels']
        U = len(utts)
        texts = [u.get('content') or u.get('text') or '' for u in utts]
        spk_ids = [speakers_to_ids(u.get('speaker','')) for u in utts]
        lab_ids = [LABEL2ID.get(lbl, SAME) for lbl in labels]

        tok = tokenizer(texts, padding='max_length', truncation=True, max_length=L, return_tensors='pt')
        input_ids = tok['input_ids']
        attention_mask = tok['attention_mask']
        speakers = torch.tensor(spk_ids, dtype=torch.long)
        labels_t = torch.tensor(lab_ids, dtype=torch.long)

        # pseudo-labels for cues / boundary / segment
        cues_np = cues_from_texts(texts)
        bound_np = boundary_targets(labels)
        seg_s_np, seg_e_np = segment_targets(labels)

        cues_t = torch.tensor(cues_np, dtype=torch.float)
        bound_t = torch.tensor(bound_np, dtype=torch.long)
        seg_s_t = torch.tensor(seg_s_np, dtype=torch.float)
        seg_e_t = torch.tensor(seg_e_np, dtype=torch.float)

        # teacher logits optional
        if 'teacher_logits' in sample and sample['teacher_logits']:
            tlog = torch.tensor(sample['teacher_logits'], dtype=torch.float)
            has_t = torch.ones((U,), dtype=torch.bool)
        else:
            tlog = torch.zeros((U, len(LABELS)), dtype=torch.float)
            has_t = torch.zeros((U,), dtype=torch.bool)

        # pad along U
        pad_u = Umax - U
        if pad_u > 0:
            pad_ids = torch.full((pad_u, L), tokenizer.pad_token_id, dtype=torch.long)
            pad_attn = torch.zeros((pad_u, L), dtype=torch.long)
            pad_spk = torch.full((pad_u,), 0, dtype=torch.long)
            pad_lab = torch.full((pad_u,), SAME, dtype=torch.long)
            pad_cues = torch.zeros((pad_u, len(CUES)), dtype=torch.float)
            pad_bound = torch.zeros((pad_u,), dtype=torch.long)
            pad_seg = torch.zeros((pad_u,), dtype=torch.float)
            pad_tlog = torch.zeros((pad_u, len(LABELS)), dtype=torch.float)
            pad_has_t = torch.zeros((pad_u,), dtype=torch.bool)
            has_t = torch.cat([has_t, pad_has_t], dim=0)

            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_attn], dim=0)
            speakers = torch.cat([speakers, pad_spk], dim=0)
            labels_t = torch.cat([labels_t, pad_lab], dim=0)
            cues_t = torch.cat([cues_t, pad_cues], dim=0)
            bound_t = torch.cat([bound_t, pad_bound], dim=0)
            seg_s_t = torch.cat([seg_s_t, pad_seg], dim=0)
            seg_e_t = torch.cat([seg_e_t, pad_seg.clone()], dim=0)
            tlog = torch.cat([tlog, pad_tlog], dim=0)

        mask_u = torch.zeros(Umax, dtype=torch.bool); mask_u[:U] = True

        all_input_ids.append(input_ids)
        all_attn.append(attention_mask)
        all_speakers.append(speakers)
        all_labels.append(labels_t)
        all_mask_u.append(mask_u)
        all_cues.append(cues_t)
        all_bound.append(bound_t)
        all_seg_start.append(seg_s_t)
        all_seg_end.append(seg_e_t)
        all_teacher.append(tlog)
        all_has_t.append(has_t)

    batch_out = {
        'input_ids': torch.stack(all_input_ids, dim=0),
        'attention_mask': torch.stack(all_attn, dim=0),
        'speakers': torch.stack(all_speakers, dim=0),
        'labels': torch.stack(all_labels, dim=0),
        'mask_u': torch.stack(all_mask_u, dim=0),
        'cues': torch.stack(all_cues, dim=0),
        'boundary_t': torch.stack(all_bound, dim=0),
        'seg_start_t': torch.stack(all_seg_start, dim=0),
        'seg_end_t': torch.stack(all_seg_end, dim=0),
        'teacher_logits': torch.stack(all_teacher, dim=0),
        'teacher_mask':   torch.stack(all_has_t,  dim=0)
    }
    return batch_out

# -----------------------------
# Model blocks
# -----------------------------
class UtteranceEncoder(nn.Module):
    def __init__(self, model_name: str, out_dim: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.backbone.config.hidden_size, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B,U,L = input_ids.size()
        x = input_ids.view(B*U, L)
        m = attention_mask.view(B*U, L)
        out = self.backbone(input_ids=x, attention_mask=m).last_hidden_state  # [B*U,L,H]
        m_float = m.unsqueeze(-1)
        pooled = (out * m_float).sum(dim=1) / (m_float.sum(dim=1).clamp_min(1.0))
        h = self.ln(self.proj(pooled))
        return h.view(B, U, -1)  # [B,U,d]

class BiasedMHABlock(nn.Module):
    def __init__(self, d_model, nhead, ffn_mult=8, dropout=0.1, rel_max_dist=128):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_mult), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model*ffn_mult, d_model)
        )
        # relative distance bias buckets [0..rel_max_dist]
        self.rel_max_dist = rel_max_dist
        self.rel_bias = nn.Parameter(torch.zeros(nhead, rel_max_dist+1))
        # speaker / sim scales
        self.speaker_gate = nn.Parameter(torch.full((nhead,), 0.1))
        self.sim_scale = nn.Parameter(torch.full((nhead,), 0.2))

    def forward(self, X, mask_u, speakers):
        # X: [B,U,d], mask_u: [B,U] (True=valid), speakers: [B,U]
        B, U, d = X.shape
        H, Dh = self.nhead, self.head_dim

        q = self.q_proj(X).view(B, U, H, Dh).transpose(1, 2)  # [B,H,U,Dh]
        k = self.k_proj(X).view(B, U, H, Dh).transpose(1, 2)
        v = self.v_proj(X).view(B, U, H, Dh).transpose(1, 2)

        # dot-product scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,U,U]

        # ---------- (1) 相对距离 bias ----------
        idxs = torch.arange(U, device=X.device)
        dist = (idxs[None, :] - idxs[:, None]).abs().clamp(max=self.rel_max_dist)  # [U,U]
        rel = self.rel_bias[:, dist]  # [H,U,U]
        scores = scores + rel.unsqueeze(0)  # [B,H,U,U]

        # ---------- (2) 说话人 cross bias ----------
        spk = speakers  # [B,U]
        cross = (spk.unsqueeze(-1) != spk.unsqueeze(-2)).to(scores.dtype)  # [B,U,U]
        scores = scores + self.speaker_gate.view(1, H, 1, 1) * cross.unsqueeze(1)

        # ---------- (3) 语义相似 bias（关键修复：加 eps，且只在有效位置上计算/使用） ----------
        Xn = F.normalize(X, dim=-1, eps=1e-6)  # eps 防止 0 向量归一化为 NaN
        sim = torch.matmul(Xn, Xn.transpose(1, 2))  # [B,U,U]
        valid = mask_u.float()
        sim = sim * (valid.unsqueeze(-1) * valid.unsqueeze(-2))  # 屏蔽 padding 行/列
        scores = scores + self.sim_scale.view(1, H, 1, 1) * sim.unsqueeze(1)

        # ---------- (4) 只 mask keys；不要 mask queries（否则会出现全 -inf 行） ----------
        key_mask = ~mask_u  # [B,U]
        if key_mask.any():
            scores = scores.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,U,Dh]
        out = out.transpose(1, 2).contiguous().view(B, U, d)

        # ---------- (5) 将无效 query 位置清零，避免噪声传播 ----------
        out = out * mask_u.unsqueeze(-1)

        # 残差 + FFN，并在每步后再清零 padding 位置，彻底隔离
        X = self.ln1(X + self.o_proj(out))
        X = X * mask_u.unsqueeze(-1)
        X = self.ln2(X + self.ffn(X))
        X = X * mask_u.unsqueeze(-1)
        return X


class DialogueTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=12, ffn_mult=8, dropout=0.1, rel_max_dist=128):
        super().__init__()
        self.layers = nn.ModuleList([
            BiasedMHABlock(d_model, nhead, ffn_mult, dropout, rel_max_dist) for _ in range(num_layers)
        ])

    def forward(self, H, mask_u, speakers):
        X = H
        for layer in self.layers:
            X = layer(X, mask_u, speakers)
        return X

# -----------------------------
# Heads
# -----------------------------
class CRFTagger(nn.Module):
    def __init__(self, d_ctx=512, n_labels=5, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_ctx, n_labels)
        self.crf = CRF(num_tags=n_labels, batch_first=True)
        # class bias (learnable)
        init_bias = torch.tensor([0.0, 0.15, 0.15, 0.0, 0.15])
        self.class_bias = nn.Parameter(init_bias)
        # transitions prior
        with torch.no_grad():
            T = torch.zeros(n_labels, n_labels)
            T[SAME,SAME] += 0.8
            for y in (NEW1,NEW2):
                T[SAME,y] -= 0.2
                T[y,SAME] += 0.3
            T[NEW1,NEW2] -= 0.5; T[NEW2,NEW1] -= 0.5
            for y in (NEW1,NEW2):
                T[BACK,y] += 0.2
            for c in range(n_labels):
                T[OFF,c] -= 0.3; T[c,OFF] -= 0.3
            self.crf.transitions.copy_(T)

    def emissions(self, Z):
        logits = self.classifier(self.drop(Z))
        return logits + self.class_bias

    def loss(self, emissions, tags, mask_u):
        return -self.crf(emissions, tags, mask=mask_u, reduction='mean')

    def decode(self, emissions, mask_u):
        return self.crf.decode(emissions, mask=mask_u)

class BoundaryHead(nn.Module):
    def __init__(self, d_ctx=512, n_labels=5, scale=0.2, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4*d_ctx, d_ctx), nn.ReLU(), nn.LayerNorm(d_ctx), nn.Dropout(dropout),
            nn.Linear(d_ctx, n_labels)
        )
        self.scale = scale
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, Z, boundary_t=None, mask_u=None):
        Z_prev = torch.roll(Z, shifts=1, dims=1)
        delta = Z - Z_prev
        hadamard = Z * Z_prev
        x = torch.cat([Z_prev, Z, delta, hadamard], dim=-1)
        logits = self.mlp(x)  # [B,U,5]
        bias = torch.zeros_like(logits)
        bias[..., NEW1] = logits[..., NEW1]
        bias[..., NEW2] = logits[..., NEW2]
        bias[..., BACK] = logits[..., BACK]
        bias[..., OFF]  = logits[..., OFF]
        loss = None
        if boundary_t is not None and mask_u is not None:
            B,U,_ = logits.shape
            loss_mat = self.ce(logits.view(B*U, -1), boundary_t.view(-1))
            loss_mat = loss_mat.view(B,U)
            loss = (loss_mat * mask_u.float()).sum() / mask_u.float().sum().clamp_min(1.0)
        return self.scale * bias, loss

class CuesHead(nn.Module):
    def __init__(self, d_ctx=512, n_cues=len(CUES), n_labels=len(LABELS), dropout=0.1, scale=0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_ctx, d_ctx), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ctx, n_cues)
        )
        self.to_bias = nn.Linear(n_cues, n_labels, bias=False)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.scale = scale

    def forward(self, Z, cues_t=None, mask_u=None):
        logits_cues = self.mlp(Z)             # [B,U,C]
        bias = self.to_bias(torch.tanh(logits_cues))  # [B,U,5]
        loss = None
        if cues_t is not None and mask_u is not None:
            loss_mat = self.bce(logits_cues, cues_t)
            loss = (loss_mat.sum(-1) * mask_u.float()).sum() / mask_u.float().sum().clamp_min(1.0)
        return self.scale * bias, loss

class SegmentHead(nn.Module):
    def __init__(self, d_ctx=512, dropout=0.1):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(d_ctx, d_ctx), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ctx, 1))
        self.end   = nn.Sequential(nn.Linear(d_ctx, d_ctx), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ctx, 1))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, Z, start_t=None, end_t=None, mask_u=None):
        s_log = self.start(Z).squeeze(-1)  # [B,U]
        e_log = self.end(Z).squeeze(-1)    # [B,U]
        loss = None
        if start_t is not None and end_t is not None and mask_u is not None:
            ls = self.bce(s_log, start_t)
            le = self.bce(e_log, end_t)
            loss = ((ls+le) * mask_u.float()).sum() / mask_u.float().sum().clamp_min(1.0)
        return s_log, e_log, loss

# -----------------------------
# Full Model
# -----------------------------
class HSTNModel(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.utt = UtteranceEncoder(cfg.model_name, cfg.d_ctx)
        self.ctx = DialogueTransformer(d_model=cfg.d_ctx, nhead=cfg.ctx_heads,
                                       num_layers=cfg.ctx_layers, ffn_mult=cfg.ffn_mult,
                                       dropout=cfg.dropout, rel_max_dist=cfg.rel_max_dist)
        self.boundary = BoundaryHead(cfg.d_ctx, len(LABELS), scale=0.2, dropout=cfg.dropout)
        self.cues = CuesHead(cfg.d_ctx, len(CUES), len(LABELS), dropout=cfg.dropout, scale=0.15)
        self.segment = SegmentHead(cfg.d_ctx, dropout=cfg.dropout)
        self.crf_head = CRFTagger(cfg.d_ctx, len(LABELS), dropout=cfg.dropout)

    def forward(self, batch):
        H = self.utt(batch['input_ids'], batch['attention_mask'])            # [B,U,d]
        Z = self.ctx(H, batch['mask_u'], batch['speakers'])                  # [B,U,d]
        b_bias, loss_boundary = self.boundary(Z, batch['boundary_t'], batch['mask_u'])
        c_bias, loss_cues     = self.cues(Z, batch['cues'], batch['mask_u'])
        s_log, e_log, loss_segment = self.segment(Z, batch['seg_start_t'], batch['seg_end_t'], batch['mask_u'])

        emissions = self.crf_head.emissions(Z) + b_bias + c_bias             # [B,U,5]
        loss_main = self.crf_head.loss(emissions, batch['labels'], batch['mask_u'])

        # distillation (if teacher logits provided)
        loss_distill = emissions.new_tensor(0.0)
        tlog = batch['teacher_logits']  # [B,U,5]
        tmask = batch.get('teacher_mask', None)  # [B,U] bool
        if tmask is not None and tmask.any():
            # 只对有 teacher 的位置算 KL，其余跳过
            T = self.cfg.distill_T
            p = F.log_softmax(emissions / T, dim=-1)
            q = F.softmax(tlog / T, dim=-1)
            # 数值安全检查
            q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
            p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
            kl = F.kl_div(p, q, reduction='none').sum(-1)  # [B,U]
            kl = kl * tmask.float() * batch['mask_u'].float()
            denom = (tmask.float() * batch['mask_u'].float()).sum().clamp_min(1.0)
            loss_distill = (T*T) * (kl.sum() / denom)


        # total loss
        loss = (self.cfg.w_main * loss_main +
                self.cfg.w_boundary * loss_boundary +
                self.cfg.w_cues * loss_cues +
                self.cfg.w_segment * loss_segment +
                self.cfg.w_distill * loss_distill)
        return loss, {
            'loss_main': loss_main.detach(),
            'loss_boundary': loss_boundary.detach(),
            'loss_cues': loss_cues.detach(),
            'loss_segment': loss_segment.detach(),
            'loss_distill': loss_distill.detach(),
            'emissions': emissions.detach(),
        }

    def decode(self, batch):
        with torch.no_grad():
            H = self.utt(batch['input_ids'], batch['attention_mask'])
            Z = self.ctx(H, batch['mask_u'], batch['speakers'])
            b_bias, _ = self.boundary(Z)
            c_bias, _ = self.cues(Z)
            emissions = self.crf_head.emissions(Z) + b_bias + c_bias
            paths = self.crf_head.decode(emissions, batch['mask_u'])
        return paths

# -----------------------------
# Metrics
# -----------------------------

def macro_f1_focus(pred_paths: List[List[int]], gold: torch.Tensor, mask_u: torch.Tensor) -> float:
    preds, gts = flatten_preds_gts(pred_paths, gold, mask_u)
    focus_ids = [NEW1, NEW2, BACK, OFF]
    f1s = []
    for cid in focus_ids:
        tp = ((preds==cid) & (gts==cid)).sum()
        fp = ((preds==cid) & (gts!=cid)).sum()
        fn = ((preds!=cid) & (gts==cid)).sum()
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        f1s.append(float(f1))
    return float(sum(f1s)/len(f1s)) if f1s else 0.0


def macro_f1_all(pred_paths, gold, mask_u) -> float:
    preds, gts = flatten_preds_gts(pred_paths, gold, mask_u)
    f1s = []
    for cid in range(len(LABELS)):
        tp = ((preds==cid) & (gts==cid)).sum()
        fp = ((preds==cid) & (gts!=cid)).sum()
        fn = ((preds!=cid) & (gts==cid)).sum()
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        f1s.append(float(f1))
    return float(sum(f1s)/len(f1s)) if f1s else 0.0


def token_accuracy(pred_paths, gold, mask_u) -> float:
    preds, gts = flatten_preds_gts(pred_paths, gold, mask_u)
    return float((preds==gts).sum() / max(1,len(preds)))


def flatten_preds_gts(pred_paths, gold, mask_u):
    gold_np = gold.cpu().numpy()
    mask_np = mask_u.cpu().numpy()
    preds, gts = [], []
    for b in range(len(pred_paths)):
        U = int(mask_np[b].sum())
        preds.extend(pred_paths[b][:U])
        gts.extend(gold_np[b][:U].tolist())
    return np.array(preds), np.array(gts)


def confusion_report(pred_paths, gold, mask_u, labels=LABELS):
    preds, gts = flatten_preds_gts(pred_paths, gold, mask_u)
    K = len(labels)
    cm = np.zeros((K,K), dtype=int)
    for p,g in zip(preds,gts):
        cm[g,p] += 1
    lines = []
    for cid, name in enumerate(labels):
        tp = cm[cid,cid]
        fp = cm[:,cid].sum() - tp
        fn = cm[cid,:].sum() - tp
        supp = cm[cid,:].sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        lines.append(f"{name:5s} | P={prec:5.2f} R={rec:5.2f} F1={f1:5.2f} | support={supp}")
    return cm, "\n".join(lines)

# -----------------------------
# Train / Eval
# -----------------------------
def _assert_finite(t, name):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} has NaN/Inf")

# 在关键节点后加几次（调试期用）


def build_optimizer(model: HSTNModel, cfg: CFG, stage: int):
    params = []
    if stage == 1:
        # encoder frozen: only heads+ctx
        for p in model.utt.backbone.parameters(): p.requires_grad=False
        params = [
            {"params": list(model.utt.proj.parameters()) + list(model.utt.ln.parameters()), "lr": cfg.lr_head},
            {"params": model.ctx.parameters(), "lr": cfg.lr_ctx},
            {"params": model.boundary.parameters(), "lr": cfg.lr_head},
            {"params": model.cues.parameters(), "lr": cfg.lr_head},
            {"params": model.segment.parameters(), "lr": cfg.lr_head},
            {"params": model.crf_head.parameters(), "lr": cfg.lr_head},
        ]
    else:
        # unfreeze top K encoder layers with tiny lr
        enc = model.utt.backbone
        nL = enc.config.num_hidden_layers
        for name, p in enc.named_parameters():
            on = any(f"layer.{i}." in name for i in range(nL-cfg.unfreeze_top_k, nL))
            p.requires_grad = on
        enc_params = [p for p in enc.parameters() if p.requires_grad]
        params = [
            {"params": enc_params, "lr": cfg.lr_enc_unfrozen},
            {"params": list(model.utt.proj.parameters()) + list(model.utt.ln.parameters()), "lr": cfg.lr_head},
            {"params": model.ctx.parameters(), "lr": cfg.lr_ctx},
            {"params": model.boundary.parameters(), "lr": cfg.lr_head},
            {"params": model.cues.parameters(), "lr": cfg.lr_head},
            {"params": model.segment.parameters(), "lr": cfg.lr_head},
            {"params": model.crf_head.parameters(), "lr": cfg.lr_head},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    return optimizer


def make_loaders(cfg: CFG, tokenizer):
    train_ds = HSTNDataset(cfg.train_path)
    dev_ds   = HSTNDataset(cfg.dev_path)

    # Weighted sampler to boost minority-class windows
    def window_weight(item):
        from collections import Counter
        cnt = Counter(item['labels'])
        minority = cnt.get('NEW1',0)+cnt.get('NEW2',0)+cnt.get('BACK',0)+cnt.get('OFF',0)
        U = len(item['labels']) if item.get('labels') else 1
        return 1.0 + cfg.sampler_alpha * (minority/max(1,U))

    weights = [window_weight(it) for it in train_ds.items]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                          collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_subwords))
    dev_dl   = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_subwords))
    return train_dl, dev_dl


def train_full(cfg: CFG):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dl, dev_dl = make_loaders(cfg, tokenizer)
    model = HSTNModel(cfg).to(device)

    total_steps = cfg.epochs * max(1, len(train_dl))
    warmup = int(total_steps * cfg.warmup_ratio)

    best_f1 = -1.0
    for epoch in range(1, cfg.epochs+1):
        stage = 1 if epoch <= cfg.freeze_epochs else 2
        optimizer = build_optimizer(model, cfg, stage)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

        # ---- train ----
        model.train()
        running = {k:0.0 for k in ['total','main','boundary','cues','segment','distill']}
        for step, batch in enumerate(train_dl, 1):
            for k in batch: batch[k] = batch[k].to(device)
            loss, logs = model(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            running['total'] += float(loss.item())
            running['main'] += float(logs['loss_main'])
            running['boundary'] += float(logs['loss_boundary'])
            running['cues'] += float(logs['loss_cues'])
            running['segment'] += float(logs['loss_segment'])
            running['distill'] += float(logs['loss_distill'])
            if step % 20 == 0:
                n = 20
                print(f"epoch{epoch} step{step}/{len(train_dl)} loss={running['total']/n:.3f} "+
                      f"main={running['main']/n:.3f} bnd={running['boundary']/n:.3f} cues={running['cues']/n:.3f} "+
                      f"seg={running['segment']/n:.3f} dis={running['distill']/n:.3f}")
                for k in running: running[k]=0.0

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            f1_focus_list, acc_list, f1_all_list = [], [], []
            all_preds, all_gts = [], []
            for batch in dev_dl:
                for k in batch: batch[k] = batch[k].to(device)
                paths = model.decode(batch)
                f1_focus_list.append(macro_f1_focus(paths, batch['labels'], batch['mask_u']))
                acc_list.append(token_accuracy(paths, batch['labels'], batch['mask_u']))
                f1_all_list.append(macro_f1_all(paths, batch['labels'], batch['mask_u']))
                preds, gts = flatten_preds_gts(paths, batch['labels'], batch['mask_u'])
                all_preds.append(preds); all_gts.append(gts)
            # aggregate
            if all_preds:
                all_preds = np.concatenate(all_preds); all_gts = np.concatenate(all_gts)
                K = len(LABELS); cm = np.zeros((K,K), dtype=int)
                for p,g in zip(all_preds, all_gts): cm[g,p]+=1
                print("Confusion matrix (rows=gold, cols=pred):\n", cm)
                for cid, name in enumerate(LABELS):
                    tp = cm[cid,cid]; fp = cm[:,cid].sum()-tp; fn = cm[cid,:].sum()-tp
                    supp = cm[cid,:].sum()
                    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
                    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
                    print(f"{name:5s} | P={prec:5.2f} R={rec:5.2f} F1={f1:5.2f} | support={supp}")
            mean_f1_focus = sum(f1_focus_list)/len(f1_focus_list) if f1_focus_list else 0.0
            mean_acc = sum(acc_list)/len(acc_list) if acc_list else 0.0
            mean_f1_all = sum(f1_all_list)/len(f1_all_list) if f1_all_list else 0.0
        print(f"[DEV] epoch {epoch} acc(all5)={mean_acc:.4f}  macroF1(N1/N2/BACK/OFF)={mean_f1_focus:.4f}  macroF1(all5)={mean_f1_all:.4f}")

        # save best
        # os.makedirs('./checkpoints', exist_ok=True)
        # torch.save({'cfg': asdict(cfg), 'state_dict': model.state_dict()}, f'./checkpoints/hstn_epoch{epoch}.pt')
        # if mean_f1_focus > best_f1:
        #     best_f1 = mean_f1_focus
        #     torch.save({'cfg': asdict(cfg), 'state_dict': model.state_dict()}, './checkpoints/hstn_best.pt')
        #     print(f"Saved BEST checkpoint (macroF1_focus={best_f1:.4f})")


if __name__ == '__main__':
    cfg = CFG()
    print("Using device:", 'cuda' if torch.cuda.is_available() else 'cpu')
    train_full(cfg)

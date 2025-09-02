import os, json, re
import xml.etree.ElementTree as ET

# =========================
# 可配置项
# =========================
MERGE_RUNS     = True   # 是否合并“同说话人 + 同 label”的连续句
STRIP_SPACES   = True   # 是否把多空白折叠为单空格
UID_JOIN_MODE  = "list" # 合并后 uid 拼法: "list" => "u1,u2,u3"; "range" => "u1-u5"（仅当纯数字且连续）

# =========================
# 1) XML -> 原始记录列表
# =========================
def parse_xml_to_list(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        namespace_uri = root.tag.split("}")[0].strip("{")
        ns = {"ns": namespace_uri}
        print(f"[XML] {os.path.basename(xml_file)}  ns={namespace_uri}")

        out = []
        u_elems = root.findall(".//ns:u", ns)
        for elem in u_elems:
            uid = elem.attrib.get("uID", "Unknown")
            speaker = elem.attrib.get("who", "Unknown")
            words = [w.text for w in elem.findall("ns:w", ns) if w.text]
            content = " ".join(words) if words else ""
            if STRIP_SPACES and content:
                content = re.sub(r"\s+", " ", content).strip()

            a = elem.find("ns:a", ns)
            speech_act = a.text.strip() if (a is not None and a.text) else "Unknown"

            out.append({"uID": uid, "speaker": speaker, "content": content, "speech_act": speech_act})
        return out
    except Exception as e:
        print(f"[ERROR] {xml_file}: {e}")
        return []

# =========================
# 2) 映射 speech_act -> 5类
# =========================
SPEECH_ACT_MAP = {
    "$INEW":"$NEW1", "$IPNEW":"$NEW2",
    "$IPSAME":"$SAME", "$PISAME":"$SAME",
    "$IOFF":"$OFF", "$POFF":"$OFF",
    "$PNEW":"$NEW1", "$PINEW":"$NEW2",
    "$BACK":"$BACK",
}
def filter_and_map(records):
    out = []
    for r in records:
        sa = r.get("speech_act")
        if sa in SPEECH_ACT_MAP:
            out.append({
                "uID": r["uID"],
                "speaker": r["speaker"],
                "content": r["content"],
                "label": SPEECH_ACT_MAP[sa].replace("$","")  # "$NEW1" -> "NEW1"
            })
    return out

# =========================
# 3) 可选：合并相邻“同说话人 + 同label”
# =========================
def _join_uid(uids):
    if UID_JOIN_MODE == "range":
        try:
            nums = [int(re.findall(r"\d+", str(u))[0]) for u in uids]
            if nums and nums == list(range(nums[0], nums[-1]+1)):
                return f"{uids[0]}-{uids[-1]}"
        except Exception:
            pass
    return ",".join(map(str, uids))

def merge_runs_same_speaker_label(records):
    merged, buf = [], None
    for r in records:
        spk = (r["speaker"] or "").strip()
        lbl = str(r["label"]).strip().upper()
        txt = (r["content"] or "").strip()
        if buf is None:
            buf = {"uIDs":[r["uID"]], "speaker": spk, "content": txt, "label": lbl}
        else:
            if spk == buf["speaker"] and lbl == buf["label"]:
                buf["uIDs"].append(r["uID"])
                if txt:
                    buf["content"] = (buf["content"] + " " + txt).strip() if buf["content"] else txt
            else:
                merged.append({"uID": _join_uid(buf["uIDs"]), "speaker": buf["speaker"],
                               "content": buf["content"], "label": buf["label"]})
                buf = {"uIDs":[r["uID"]], "speaker": spk, "content": txt, "label": lbl}
    if buf is not None:
        merged.append({"uID": _join_uid(buf["uIDs"]), "speaker": buf["speaker"],
                       "content": buf["content"], "label": buf["label"]})
    return merged

# =========================
# 4) 计算 tran（上升沿：prev!=NEW1 且 curr==NEW1）
# =========================
def compute_tran_labels(recs):
    tran, prev_is_new1 = [], False
    for r in recs:
        lbl = str(r.get("label","")).strip().upper().replace("$","").replace("_","")
        is_new1 = (lbl == "NEW1")
        tran.append(1 if (is_new1 and not prev_is_new1) else 0)
        prev_is_new1 = is_new1
    return tran

# =========================
# 5) 说话人编号（对话内 0..N-1）
# =========================
def build_speaker_id_map(records):
    spk2id = {}
    for r in records:
        spk = (r.get("speaker") or "").strip()
        if spk not in spk2id:
            spk2id[spk] = len(spk2id)
    return spk2id

# =========================
# 6) 组装对话对象（不切窗）
# =========================
def to_dialogue(dialogue_id, recs, tran_flags):
    spk2id = build_speaker_id_map(recs)
    utts, labels, trans = [], [], []
    for r, t in zip(recs, tran_flags):
        spk_raw = (r["speaker"] or "").strip()
        utts.append({
            "uid": r["uID"],
            "speaker": spk_raw,                 # 原始字符串
            "speaker_id": int(spk2id[spk_raw]), # 数字ID：0..N-1
            "content": r["content"]
        })
        labels.append(r["label"])
        trans.append(int(t))
    return {
        "dialogue_id": dialogue_id,
        "utterances": utts,
        "labels": labels,
        "tran": trans,
        "num_speakers": len(spk2id)
    }

# =========================
# 7) 目录 -> JSONL（一段一行）
# =========================
def xml_dir_to_jsonl(xml_dir, out_jsonl, prefix_filter=None):
    paths = []
    for name in os.listdir(xml_dir):
        if not name.lower().endswith(".xml"):
            continue
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        paths.append(os.path.join(xml_dir, name))
    paths.sort()

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    n_ok = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            recs = parse_xml_to_list(p)
            recs = filter_and_map(recs)
            if not recs:
                continue
            if MERGE_RUNS:
                recs = merge_runs_same_speaker_label(recs)
            tran = compute_tran_labels(recs)
            dia = to_dialogue(base, recs, tran)
            f.write(json.dumps(dia, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[DONE] wrote {n_ok} dialogues -> {out_jsonl}")

# =========================
# 示例调用
# =========================
if __name__ == "__main__":
    coded_n_folder_path  = "./RawData/Coded_N-xml"
    coded_tb_folder_path = "./RawData/Coded_TB-xml"
    val_folder_path      = "./RawData/Val-xml"

    os.makedirs("./HSTN_JSONL", exist_ok=True)
    xml_dir_to_jsonl(coded_n_folder_path,  "./HSTN_JSONL/coded_n.jsonl")
    xml_dir_to_jsonl(coded_tb_folder_path, "./HSTN_JSONL/coded_tb.jsonl")
    xml_dir_to_jsonl(val_folder_path,      "./HSTN_JSONL/val.jsonl")

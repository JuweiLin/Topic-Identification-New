import os, json, glob
import xml.etree.ElementTree as ET

############################
# 1) 基础：XML → 句子列表
############################
def parse_xml_to_list(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        namespace_uri = root.tag.split("}")[0].strip("{")
        namespace = {"ns": namespace_uri}
        print("Detected namespace:", namespace_uri) 

        result = []
        u_elements = root.findall(".//ns:u", namespace)
        print("Found <u> elements:", len(u_elements))

        for elem in u_elements:
            uid = elem.attrib.get("uID", "Unknown")
            speaker = elem.attrib.get("who", "Unknown")
            content = " ".join([w.text for w in elem.findall("ns:w", namespace) if w.text])
            speech_act_elem = elem.find("ns:a", namespace)
            speech_act = speech_act_elem.text.strip() if (speech_act_elem is not None and speech_act_elem.text is not None) else "Unknown"

            result.append({
                "uID": uid,
                "speaker": speaker,
                "content": content,
                "speech_act": speech_act
            })
        print("XML to list conversion completed. Total records:", len(result))
        return result
    except ET.ParseError as e:
        print(f"XML ParseError: {e}")
    except FileNotFoundError:
        print(f"File not found: {xml_file}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return []

###########################################
# 2) 将 speech_act → {SAME,NEW1,NEW2,BACK,OFF}
###########################################
SPEECH_ACT_MAP = {
    "$INEW": "$NEW1", "$IPNEW": "$NEW2",
    "$IPSAME": "$SAME", "$PISAME": "$SAME",
    "$IOFF": "$OFF", "$POFF": "$OFF",
    "$PNEW": "$NEW1", "$PINEW": "$NEW2",
    "$BACK": "$BACK",
}

def filter_and_map(records):
    """仅保留能映射到五类标签的句子，并附加 label。"""
    out = []
    for r in records:
        sa = r.get("speech_act")
        if sa in SPEECH_ACT_MAP:
            r2 = {
                "uID": r["uID"],
                "speaker": r["speaker"],
                "content": r["content"],
                "label": SPEECH_ACT_MAP[sa].replace("$", "")  # "$NEW1" -> "NEW1"
            }
            out.append(r2)
    return out

##########################################
# 3) 统一说话人名：INV / PAR（可按你语料调整）
##########################################
def normalize_speaker(s):
    s = (s or "").upper()
    if s.startswith("INV"):
        return "INV"
    if s.startswith("PAR"):
        return "PAR"
    # 其它全部归一到PAR（或自定义）
    return "PAR"

############################################
# 4) 生成 HSTN 对话对象（不加任何 PAD）
############################################
def to_hstn_dialogue(dialogue_id, recs):
    utts = []
    labels = []
    for r in recs:
        utts.append({
            "uid": r["uID"],
            "speaker": normalize_speaker(r["speaker"]),
            "content": r["content"]
        })
        labels.append(r["label"])
    return {
        "dialogue_id": dialogue_id,
        "utterances": utts,
        "labels": labels
    }

########################################
# 5) 对长对话做窗口：win=256, stride=128
########################################
def window_dialogue(dialogue, win=256, stride=64):
    U = len(dialogue["utterances"])
    if U == 0:
        return []

    # 若本来就 <=win，直接返回单窗口
    if U <= win:
        return [{
            "dialogue_id": f'{dialogue["dialogue_id"]}#1-{U}',
            "utterances": dialogue["utterances"],
            "labels": dialogue["labels"]
        }]

    # 滑窗
    out = []
    start = 1
    # 用 0-based 索引切片更方便
    i = 0
    while i < U:
        j = min(i + win, U)
        utts = dialogue["utterances"][i:j]
        labs = dialogue["labels"][i:j]
        # 对话内的 uid 范围：用原 uid 显示更直观；若 uid 不是纯数字可显示序号
        left_uid = utts[0]["uid"]
        right_uid = utts[-1]["uid"]
        did = f'{dialogue["dialogue_id"]}#{left_uid}-{right_uid}'
        out.append({"dialogue_id": did, "utterances": utts, "labels": labs})
        if j == U:  # 收尾
            break
        i += stride
        # 若最后一段太短且没覆盖到末尾，强行补一个“尾窗”
        if U - i < win // 2 and j < U:
            i = max(U - win, 0)
    return out

##############################
# 6) 写 JSONL
##############################
def write_jsonl(dialogues, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

#############################################
# 7) 整体入口：批量把 XML → JSONL（带窗口）
#############################################
def xml_dir_to_jsonl(xml_dir, out_jsonl, prefix_filter=None, win=256, stride=64):
    paths = []
    for name in os.listdir(xml_dir):
        if not name.lower().endswith(".xml"):
            continue
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        paths.append(os.path.join(xml_dir, name))
    paths.sort()

    all_windows = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]  # e.g., "n01"
        records = parse_xml_to_list(p)
        records = filter_and_map(records)
        if not records:
            continue
        dia = to_hstn_dialogue(base, records)
        wins = window_dialogue(dia, win=win, stride=stride)
        all_windows.extend(wins)

    write_jsonl(all_windows, out_jsonl)
    print(f"Wrote {len(all_windows)} windows to {out_jsonl}")

#############################################
# 示例调用
#############################################
if __name__ == "__main__":
    # 你的两类目录
    coded_n_folder_path = "./RawData/Coded_N-xml"
    coded_tb_folder_path = "./RawData/Coded_TB-xml"
    val_folder_path = "./RawData/Val-xml"

    os.makedirs("./HSTN_JSONL", exist_ok=True)
    xml_dir_to_jsonl(coded_n_folder_path, "./HSTN_JSONL/coded_n.jsonl", prefix_filter="n", win=256, stride=64)
    xml_dir_to_jsonl(coded_tb_folder_path, "./HSTN_JSONL/coded_tb.jsonl", prefix_filter="tb", win=256, stride=64)
    xml_dir_to_jsonl(val_folder_path, "./HSTN_JSONL/val.jsonl", prefix_filter="", win=256, stride=64)


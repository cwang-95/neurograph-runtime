#!/usr/bin/env python3
"""
merge_talk_vision.py — 合并「讲者语音转写 + PPT 视觉描述」为 talk 级双层灌库文档。

输入：
  - lecture_study_output/**/slide_speech.jsonl  （逐页语音，字段 slide_number/speech）
  - ppt_vision/output/{session}__{talk}.jsonl     （逐页视觉，字段 index/page_desc）

输出：
  - merged_talks/{talk}.md  —— 每个 talk 一份干净的「语音 + 视觉」双层文档，
    每页一个结构块，供 cognee add 灌入知识图谱。

对齐键：slide_speech.slide_number == VL.index（1-based），
另有 VL.slide 文件名含时间戳可交叉验证。
"""

import json
import os
import re
import sys
from pathlib import Path

LECTURE_ROOT = Path(os.path.expanduser(
    "~/Documents/文稿_移动端mac/UIH/2026AAPM/lecture_study_output"
))
VISION_OUT = Path(os.path.expanduser(
    "~/.openclaw/workspace/projects/neurograph/ppt_vision/output"
))
OUT_DIR = Path(os.path.expanduser(
    "~/.openclaw/workspace/projects/neurograph/merged_talks"
))


def find_talk_dirs(root: Path):
    """返回所有含 slides 目录的 talk 目录（session/talk 两级）。"""
    talks = []
    for slides_dir in root.rglob("slides"):
        talk_dir = slides_dir.parent
        session = talk_dir.parent.name
        talk = talk_dir.name
        talks.append({"session": session, "talk": talk, "dir": talk_dir})
    return talks


def load_slide_speech(talk: dict):
    """读 slide_speech.jsonl，返回 {slide_number: speech_text}。"""
    path = talk["dir"] / "slide_speech.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d.get("slide_number", 0))] = d.get("speech", "").strip()
    return out


def load_vision(talk: dict):
    """读对应 VL output 文件，返回 {index: page_desc}。"""
    # vision 文件名 = {session}__{talk}.jsonl
    fname = f"{talk['session']}__{talk['talk']}.jsonl"
    path = VISION_OUT / fname
    if not path.exists():
        # 尝试只按 talk 名匹配（session 名可能差异）
        matches = list(VISION_OUT.glob(f"*__{talk['talk']}.jsonl"))
        if not matches:
            return {}
        path = matches[0]
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d.get("index", 0))] = d.get("page_desc", "").strip()
    return out


def load_ocr(talk: dict):
    """读 ocr.jsonl，返回 {slide_number: text}，作视觉补充的兜底。"""
    path = talk["dir"] / "ocr.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d.get("slide_number", 0))] = (d.get("text") or "").strip()
    return out


def clean_title(s):
    """把 talk 英文名转成可读标题。"""
    s = s.replace("-", " ").strip()
    return " ".join(w.capitalize() for w in s.split())


def build_md(talk: dict, speech: dict, vision: dict, ocr: dict):
    """合成一份 talk 级 markdown。"""
    all_pages = sorted(set(list(speech.keys()) + list(vision.keys()) + list(ocr.keys())))
    if not all_pages:
        return None

    title = clean_title(talk["talk"])
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> Session: {talk['session']}")
    lines.append(f"> 共 {len(all_pages)} 页（语音转写 + PPT 视觉提取合并）")
    lines.append("")

    for page in all_pages:
        lines.append(f"## Slide {page}")
        lines.append("")
        # 语音层
        sp = speech.get(page, "")
        if sp:
            lines.append("### 讲者语音")
            lines.append("")
            lines.append(sp)
            lines.append("")
        # 视觉层（VL 优先，OCR 兜底）
        vis = vision.get(page, "")
        if vis:
            lines.append("### PPT 视觉提取")
            lines.append("")
            lines.append(vis)
            lines.append("")
        elif ocr.get(page, ""):
            lines.append("### PPT OCR 文字")
            lines.append("")
            lines.append(ocr[page])
            lines.append("")

    return "\n".join(lines)


def main():
    talks = find_talk_dirs(LECTURE_ROOT)
    if not talks:
        print("❌ 未找到任何 talk 目录", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = []

    for talk in sorted(talks, key=lambda t: (t["session"], t["talk"])):
        speech = load_slide_speech(talk)
        vision = load_vision(talk)
        ocr = load_ocr(talk)
        md = build_md(talk, speech, vision, ocr)
        if md is None:
            print(f"⚠️  {talk['talk'][:50]} 无任何页数据，跳过")
            continue

        out_name = talk["talk"] + ".md"
        out_path = OUT_DIR / out_name
        out_path.write_text(md)

        n_speech = len(speech)
        n_vision = len(vision)
        n_pages = len(set(list(speech.keys()) + list(vision.keys())))
        stats.append((talk["talk"], n_pages, n_speech, n_vision, out_path.stat().st_size))
        print(f"✅ {talk['talk'][:55]:<55}  页{n_pages:>3}  语音{n_speech:>3}  视觉{n_vision:>3}  {out_path.stat().st_size}字节")

    print("\n" + "=" * 70)
    print(f"合计 {len(stats)} 个 talk 已合并，输出目录：{OUT_DIR}")
    total_pages = sum(s[1] for s in stats)
    total_bytes = sum(s[4] for s in stats)
    print(f"总页数 {total_pages}，总字节 {total_bytes}")


if __name__ == "__main__":
    main()

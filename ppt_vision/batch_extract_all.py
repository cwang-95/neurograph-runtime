#!/usr/bin/env python3
"""
批量遍历 12 个 talk 的 slides 目录，逐页视觉提取，输出到统一目录。
每个 talk 一个 jsonl 文件，命名用 talk 目录名（含父目录前缀避免重名冲突）。

用法：
  python3 batch_extract_all.py [--root <lecture_study_output路径>] [--outdir <输出目录>]
"""
import json, os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppt_vision_extract import call_vl

ROOT = os.path.expanduser(
    "~/Documents/文稿_移动端mac/UIH/2026AAPM/lecture_study_output"
)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def find_slides_dirs(root):
    dirs = []
    for d in glob.glob(os.path.join(root, "*", "*", "slides")):
        jpgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if jpgs:
            dirs.append((d, jpgs))
    return dirs


def main():
    root = ROOT
    outdir = OUTDIR
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]
    if "--outdir" in sys.argv:
        outdir = sys.argv[sys.argv.index("--outdir") + 1]

    os.makedirs(outdir, exist_ok=True)

    dirs = find_slides_dirs(root)
    print(f"共发现 {len(dirs)} 个 talk 的 slides 目录", file=sys.stderr)

    for d, jpgs in dirs:
        # talk 唯一命名 = 父目录(主题) + talk目录名
        rel = os.path.relpath(d, root).replace("/slides", "").replace("/", "__")
        out_path = os.path.join(outdir, f"{rel}.jsonl")
        if os.path.exists(out_path):
            # 断点续传：跳过已完成的（按行数判断）
            done = sum(1 for _ in open(out_path, encoding="utf-8")) if os.path.getsize(out_path) > 0 else 0
            if done >= len(jpgs):
                print(f"[跳过] {rel}: 已完成 {done}/{len(jpgs)}", file=sys.stderr)
                continue
            print(f"[续传] {rel}: {done}/{len(jpgs)}", file=sys.stderr)
            jpgs = jpgs[done:]
        else:
            print(f"[开始] {rel}: 0/{len(jpgs)}", file=sys.stderr)

        with open(out_path, "a", encoding="utf-8") as out:
            for idx, jpg in enumerate(jpgs, 1):
                fn = os.path.basename(jpg)
                desc = call_vl(jpg)
                if desc is None:
                    desc = "提取失败"
                rec = {"talk": rel, "slide": fn, "index": idx, "page_desc": desc}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if desc != "提取失败":
                    print(f"  [{rel}] {fn} OK", file=sys.stderr, flush=True)
                else:
                    print(f"  [{rel}] {fn} FAIL", file=sys.stderr, flush=True)
            print(f"[完成] {rel}: {len(jpgs)} 页", file=sys.stderr, flush=True)

    print("\n全部完成", file=sys.stderr)


if __name__ == "__main__":
    main()

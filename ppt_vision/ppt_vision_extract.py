#!/usr/bin/env python3
"""
PPT 视觉理解提取脚本（调云端千问 VL：qwen3-vl-plus）

用法：
  python3 ppt_vision_extract.py <slides目录> <输出jsonl> [--limit N] [--start N]

产出每页一行 JSON：
  {"talk": ..., "slide": "001_...jpg", "index": 1, "page_desc": "..."}

page_desc 是结构化视觉理解描述，包含：
- 页面标题
- 核心公式（及其含义）
- 流程图/网络结构的每一步
- 所有数字/指标/百分比
- 坐标轴标签与图表含义
- 关键专业术语及定义
"""
import json, os, sys, base64, time
import urllib.request

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
MODEL = "qwen3-vl-plus"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

PROMPT = """你是医学物理/放射肿瘤学领域的专业文献分析助手。请对这张 AAPM 会议 PPT 幻灯片做**严格依据图像内容**的逐项提取（不要编造图里没有的信息，看不清就明确写"看不清/不存在"）。

请按以下结构输出（中文为主，专业术语保留英文原文）：

1) **页面标题**：主标题/副标题原文

2) **核心公式**：如有公式，写出符号含义与物理意义；没有则写"无"

3) **流程图/网络架构的每一步**：按流程顺序逐条列出（如果这页是流程图/框图/网络结构图）

4) **所有数字/指标/百分比**：列出图上每个可辨识的数值，并注明它对应的含义（如误差、剂量、时间、通过率等）

5) **坐标轴标签和图表含义**：如有图表，说明横轴/纵轴标签、曲线/颜色含义、数据趋势

6) **关键专业术语及定义**：列出图上的专业术语原文（PPT 可能没给定义，此时只列术语）

请确保数值精确转录（小数点、正负号、百分号、± 号都要准确）。"""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vl(image_path, retries=3):
    b64 = encode_image(image_path)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    data = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=data, method="POST")
            req.add_header("Authorization", f"Bearer {API_KEY}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=120) as resp:
                r = json.loads(resp.read().decode())
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [重试 {attempt+1}/{retries}] {image_path}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return None


def main():
    slides_dir = sys.argv[1]
    out_path = sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0

    files = sorted([f for f in os.listdir(slides_dir) if f.lower().endswith(".jpg")])
    if limit is not None:
        files = files[start : start + limit]
    else:
        files = files[start:]

    talk = os.path.basename(os.path.dirname(slides_dir.rstrip("/")))
    print(f"目录: {slides_dir}", file=sys.stderr)
    print(f"talk: {talk}", file=sys.stderr)
    print(f"共 {len(files)} 页", file=sys.stderr)

    with open(out_path, "a", encoding="utf-8") as out:
        for i, fn in enumerate(files, 1):
            path = os.path.join(slides_dir, fn)
            print(f"[{i}/{len(files)}] {fn} ...", file=sys.stderr, end=" ")
            desc = call_vl(path)
            if desc is None:
                desc = "提取失败"
            rec = {"talk": talk, "slide": fn, "index": i, "page_desc": desc}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print("OK" if desc != "提取失败" else "FAIL", file=sys.stderr)

    print(f"\n完成，输出到 {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案① cognee 底座 — 批量灌 wiki 全量文献 + 建认知图谱
用法: ~/.cognee-venv/bin/python -u feed_wiki.py <dataset_name> [--papers N] [--no-cognify]
注意: 必须在 workspace 内运行(沙箱), 日志写到 data/logs/
"""
import os, sys, glob, asyncio, time, argparse

# litellm 忽略不支持的参数(防 dimensions 坑)
import litellm
litellm.drop_params = True
import logging
logging.getLogger("cognee").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.ERROR)

# ---------- 配置 ----------
DS_KEY = os.environ.get("DASHSCOPE_API_KEY") or ""
DS_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DS_BASE_URL = "https://api.deepseek.com/v1"   # DeepSeek LLM 端点
def _get_deepseek_key():
    try:
        import json
        cfg = json.load(open(os.path.expanduser("~/.openclaw/openclaw.json")))
        p = cfg.get("models", {}).get("providers", {}).get("custom-api-deepseek-com", {})
        return p.get("apiKey", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY", "")
DK_KEY = _get_deepseek_key()
# LLM 用 DeepSeek-V4-Flash(快且省, 抽实体/关系够用)
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_MODEL", "deepseek/deepseek-v4-flash")
os.environ.setdefault("LLM_API_KEY", DK_KEY)
os.environ.setdefault("LLM_ENDPOINT", DS_BASE_URL)
# Embedding 用本地 MLX(Qwen3-Embedding-0.6B, 1024维, 免费零延迟, 不再走 DashScope)
os.environ.setdefault("EMBEDDING_PROVIDER", "openai_compatible")
os.environ.setdefault("EMBEDDING_MODEL", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
os.environ.setdefault("EMBEDDING_API_KEY", "local-qwen")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://127.0.0.1:8000/v1")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "1024")
os.environ.setdefault("DB_PROVIDER", "sqlite")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "4")
os.environ.setdefault("EMBEDDING_MAX_CONCURRENT_DATA_POINTS", "1")
# deepseek-v4-flash 默认开 thinking mode, 但 cognee 结构化输出(强制 tool_choice)不支持 thinking
# 用 llm_args 透传 thinking=disabled 关掉(参考 litellm deepseek transformation.py)
os.environ.setdefault("LLM_ARGS", '{"thinking": {"type": "disabled"}}')

HERE = os.path.dirname(os.path.abspath(__file__))
WIKI_DIR = os.path.expanduser("~/wiki/raw/papers")
# cognee 数据库放项目 data/cognee(避免散落)
os.environ.setdefault("COGNEE_DATA_ROOT", os.path.join(HERE, "..", "data", "cognee"))

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="数据集名, 如 wiki_full")
    ap.add_argument("--papers", type=int, default=0, help="只灌前 N 篇(0=全部)")
    ap.add_argument("--no-cognify", action="store_true", help="只 add 不建图谱")
    ap.add_argument("--start", type=int, default=0, help="从第 start 篇开始(断点续灌)")
    ap.add_argument("--dir", default=None, help="灌入目录(默认 ~/wiki/raw/papers), 如灌 GitHub 项目用 ai-agent-learning 下某目录")
    args = ap.parse_args()

    import cognee
    log(f"cognee {cognee.__version__}")

    src_dir = args.dir or WIKI_DIR
    papers = sorted(glob.glob(os.path.join(src_dir, "*.md")))
    log(f"目录共 {len(papers)} 篇 ({src_dir})")
    if args.papers > 0:
        papers = papers[:args.papers]
    if args.start > 0:
        papers = papers[args.start - 1:]
    log(f"本次灌 {len(papers)} 篇(dataset={args.dataset})")

    # 1. add
    t0 = time.time()
    log("[1/2] cognee.add() 批量灌入...")
    await cognee.add(papers, dataset_name=args.dataset)
    log(f"  ✓ add 完成  耗时 {time.time()-t0:.0f}s")

    if args.no_cognify:
        log("跳过 cognify(--no-cognify)")
        return

    # 2. cognify
    t1 = time.time()
    log("[2/2] cognee.cognify() 建认知图谱(LLM抽实体/关系, 慢)...")
    await cognee.cognify(datasets=args.dataset)
    log(f"  ✓ cognify 完成  耗时 {time.time()-t1:.0f}s ({len(papers)}篇)")

    log(f"=== 全部完成  总耗时 {time.time()-t0:.0f}s ===")

if __name__ == "__main__":
    asyncio.run(main())

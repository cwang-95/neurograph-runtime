#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案① 检索入口 — cognee 知识域(文献) + ZenBrain 类脑加权
用法(日常): ~/.cognee-venv/bin/python -u query.py "自适应放疗 在线自适应" [--top 5]
          ~/.cognee-venv/bin/python -u query.py "危及器官 自动勾画" --domain knowledge
          ~/.cognee-venv/bin/python -u query.py "我今天要做什么" --domain memory
说明:
  - 默认查知识域(文献, cognee): 专业底色
  - --domain memory: 走 memory_rag(记忆域, 私有经历), 不串味
"""
import os, sys, asyncio, argparse, json
import litellm; litellm.drop_params = True

def _get_deepseek_key():
    """从 openclaw.json 读 DeepSeek key(不硬编码)"""
    try:
        import json
        cfg = json.load(open(os.path.expanduser("~/.openclaw/openclaw.json")))
        p = cfg.get("models", {}).get("providers", {}).get("custom-api-deepseek-com", {})
        return p.get("apiKey", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY", "")

DS_KEY = os.environ.get("DASHSCOPE_API_KEY") or ""
DS_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DS_BASE_URL = "https://api.deepseek.com/v1"   # DeepSeek LLM 端点(OpenAI 兼容)
DK_KEY = _get_deepseek_key()

# LLM 用 DeepSeek(付费, 额度足); Embedding 用本地 MLX(与灌库一致)
for k, v in {
    "LLM_PROVIDER": "openai", "LLM_MODEL": "deepseek/deepseek-v4-flash",
    "LLM_API_KEY": DK_KEY, "LLM_ENDPOINT": DS_BASE_URL,
    "EMBEDDING_PROVIDER": "openai_compatible",
    "EMBEDDING_MODEL": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "EMBEDDING_API_KEY": "local-qwen", "EMBEDDING_ENDPOINT": "http://127.0.0.1:8000/v1",
    "EMBEDDING_DIMENSIONS": "1024", "ENABLE_BACKEND_ACCESS_CONTROL": "false",
}.items(): os.environ.setdefault(k, v)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zenbrain_llm import boost

DATASET = "wiki_full"   # cognee 默认查询数据集(文献)
DATASET_MULTI = "https://wikifull"  # 占位, 实际用 --datasets
MEM_SEARCH = os.path.expanduser("~/.openclaw/workspace/memory_rag/scripts/search_memory.py")

def _clean_title(raw):
    """从提取的原始文本里清洗出干净的标题"""
    if not raw:
        return ""
    t = raw.strip()
    # 1. 剥掉 --- frontmatter 块(起始 --- 到下一个 --- / ...)
    if t.startswith("---"):
        # 找 frontmatter 里 title: 字段
        import re
        m = re.search(r'^title:\s*["\']?([^"\'\n]+)["\']?\s*$', t, re.M)
        if m:
            return m.group(1).strip()[:110]
        # 否则去掉整个 frontmatter 块
        lines = t.split("\n")
        out = []
        in_fm = False
        for ln in lines:
            if ln.strip().startswith("---") and not in_fm:
                in_fm = True
                continue
            if in_fm:
                if ln.strip().startswith("---"):
                    in_fm = False
                continue
            out.append(ln)
        t = "\n".join(out).strip()
    # 2. 取第一个 # 标题行
    for ln in t.split("\n"):
        if ln.strip().startswith("#"):
            return ln.strip().lstrip("# ").strip()[:110]
    # 3. 取第一段非空文本
    for ln in t.split("\n"):
        if ln.strip():
            return ln.strip()[:110]
    return t[:110]


def extract_source(x):
    """从 cognee 检索结果里挖出来源信息(journal/DOI/authors/date), 用于标注 📚"""
    import re
    inner = getattr(x, "search_result", None) or x
    raw = None
    try:
        if isinstance(inner, dict):
            raw = inner.get("text") or inner.get("content") or ""
        elif isinstance(inner, str):
            raw = inner
        elif isinstance(inner, list):
            for it in inner:
                t = extract_source(it)
                if t: return t
    except Exception:
        pass
    if not raw or not isinstance(raw, str):
        return ""
    fields = {}
    # frontmatter 里的键值(支持 title/journal/authors/date/DOI 等)
    for key in ("journal", "doi", "authors", "author", "date", "year", "source", "url"):
        m = re.search(rf'^{key}:\s*["\' ]*(.+?)["\' ]*$', raw, re.M | re.I)
        if m:
            v = m.group(1).strip().rstrip(',')
            if v and len(v) < 120:
                fields[key.lower()] = v
    # 也试一下 markdown 链接行 "- **来源**: xx" 或 "- 来源: xx"
    for key in ("journal", "doi", "authors", "date", "year", "url", "source"):
        if key.lower() in fields:
            continue
        m = re.search(rf'^[-*]\s*\*?\*?{key}\*?\*?\s*[:：]\s*(.+)$', raw, re.M | re.I)
        if m:
            v = m.group(1).strip().rstrip(',')
            if v and len(v) < 120:
                fields[key.lower()] = v
    parts = []
    for k in ("journal", "doi", "authors", "date", "year"):
        if k in fields:
            parts.append(f"{fields[k]}")
    return " | ".join(parts)[:180]


def extract_text(x):
    """从 cognee 检索结果里挖可读文本(标题等), 清洗 frontmatter/markdown 噪音"""
    inner = getattr(x, "search_result", None) or x
    try:
        if isinstance(inner, dict):
            for k in ("title", "text", "part", "content", "description", "name"):
                v = inner.get(k)
                if isinstance(v, str) and v.strip():
                    return _clean_title(v)
            for v in inner.values():
                if isinstance(v, (dict, list)):
                    t = extract_text(v)
                    if t: return t
        elif isinstance(inner, list):
            for it in inner[:10]:
                t = extract_text(it)
                if t: return t
    except Exception:
        pass
    return _clean_title(str(inner))[:110]

def _unwrap_graph_answer(r):
    """GRAPH_COMPLETION 返回结构统一提取为纯文本答案。
    实测稳定形态: list[str] (len=1, 元素即 LLM 综合答案)。
    兼容 dict 形态(带 search_result 字段)。"""
    answers = []
    if isinstance(r, list):
        for x in r:
            if isinstance(x, str) and x.strip():
                answers.append(x.strip())
            elif isinstance(x, dict):
                sr = x.get("search_result")
                if isinstance(sr, str) and sr.strip():
                    answers.append(sr.strip())
                elif isinstance(sr, list):
                    for it in sr:
                        if isinstance(it, str) and it.strip():
                            answers.append(it.strip())
    elif isinstance(r, dict):
        for k, v in r.items():
            if k == "search_result":
                if isinstance(v, str) and v.strip():
                    answers.append(v.strip())
                elif isinstance(v, list):
                    answers += [it for it in v if isinstance(it, str) and it.strip()]
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict) and isinstance(it.get("search_result"), str):
                        answers.append(it["search_result"].strip())
    return "\n\n".join(answers)


async def search_knowledge(q, top=5, datasets=None, mode="graph"):
    import cognee
    # 支持逗号分隔的多数据集, cognee 需要列表
    if datasets is None:
        datasets = [DATASET]
    elif isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    if mode == "chunks":
        # 纯向量检索(备选, 周末 ZenBrain 接回时用)
        r = await cognee.search(query_text=q, query_type=cognee.SearchType.CHUNKS, datasets=datasets)
        results = [{"id": str(getattr(x, "dataset_id", "") or ""),
                    "title": extract_text(x),
                    "source": extract_source(x)} for x in r[:top]]
        return results, r
    # 默认: GRAPH 图谱增强检索(向量召回 + 图邻域扩展 + LLM 综合)
    r = await cognee.search(query_text=q, query_type=cognee.SearchType.GRAPH_COMPLETION, datasets=datasets)
    answer = _unwrap_graph_answer(r)
    return answer, r

def search_memory(q, top=5):
    """记忆域: 走 memory_rag(search_memory.py), 私有经历, 不与知识串"""
    if not os.path.exists(MEM_SEARCH):
        return "memory_rag 未找到", []
    r = subprocess_run(MEM_SEARCH, q, top)
    return r

def subprocess_run(*cmd):
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.stdout[-3000:] or r.stderr[-1000:]
    except Exception as e:
        return f"err: {e}"

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--domain", choices=["knowledge", "memory"], default="knowledge")
    ap.add_argument("--datasets", default=None, help="逗号分隔数据集, 如 wiki_full,github_projects")
    ap.add_argument("--chunks", action="store_true", help="用纯向量 CHUNKS 检索(备选, 不图谱增强)")
    args = ap.parse_args()

    if args.domain == "memory":
        print(f"[记忆域] 经 memory_rag 检索: {args.query}")
        print(search_memory(args.query, args.top))
        return

    mode = "chunks" if args.chunks else "graph"
    tag = "GRAPH 图谱增强" if mode == "graph" else "CHUNKS 纯向量"
    print(f"[知识域] cognee {tag}检索: {args.query} | datasets={args.datasets or DATASET}")
    if mode == "graph":
        answer, raw = await search_knowledge(args.query, args.top, args.datasets, mode="graph")
        print(f"\n{answer}" if answer else "\n(图谱检索未召回相关内容)")
        return
    # chunks 分支: 类脑加权排序(周末 ZenBrain 接回后复用)
    results, raw = await search_knowledge(args.query, args.top, args.datasets, mode="chunks")
    boosted = boost(results) if results else []
    print(f"\n按类脑权重排序(top {len(boosted)}):")
    for i, r in enumerate(boosted, 1):
        title = r.get("title", "")[:110]
        score = r.get("_classed_score", 0)
        retr = r.get("_retrievability_pct", 0)
        src = r.get("source", "") or ""
        src_tag = f"  📚 {src}" if src else ""
        print(f"  {i}. [{score}分/可回忆{retr}%] {title}{src_tag}")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案① 检索入口 — cognee 知识域(文献) + ZenBrain 类脑加权
用法(日常): ~/.cognee-venv/bin/python -u query.py "自适应放疗 在线自适应" [--top 5]
          ~/.cognee-venv/bin/python -u query.py "问题" --mode graph-evidence --format json
          ~/.cognee-venv/bin/python -u query.py "危及器官 自动勾画" --domain knowledge
          ~/.cognee-venv/bin/python -u query.py "我今天要做什么" --domain memory
说明:
  - graph-evidence: 返回图邻域证据，不生成答案（默认，适合 Codex/OpenClaw）
  - answer: DeepSeek 基于图上下文生成答案（独立问答备用）
  - evidence: 返回原始向量片段，不生成答案（精确事实与溯源）
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
    "LLM_ARGS": '{"thinking": {"type": "disabled"}}',
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

def extract_content(x):
    """保留召回片段正文，供上层模型直接基于证据回答。"""
    inner = getattr(x, "search_result", None) or x
    if isinstance(inner, str):
        return inner.strip()
    if isinstance(inner, dict):
        for key in ("text", "content", "description", "part"):
            value = inner.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in inner.values():
            if isinstance(value, (dict, list)):
                content = extract_content(value)
                if content:
                    return content
    if isinstance(inner, list):
        parts = [extract_content(item) for item in inner]
        return "\n\n".join(part for part in parts if part)
    return str(inner).strip()

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


async def search_knowledge(q, top=5, datasets=None, mode="graph-evidence"):
    import cognee
    # 支持逗号分隔的多数据集, cognee 需要列表
    if datasets is None:
        datasets = [DATASET]
    elif isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    if mode == "evidence":
        r = await cognee.search(
            query_text=q, query_type=cognee.SearchType.CHUNKS,
            datasets=datasets, top_k=max(top, 15),
        )
        results = [{"id": str(getattr(x, "dataset_id", "") or ""),
                    "title": extract_text(x), "source": extract_source(x),
                    "content": extract_content(x)} for x in r[:top]]
        return results, r
    if mode == "graph-evidence":
        # 与 GRAPH_COMPLETION 使用同一图检索器，但 only_context 跳过 LLM 回答。
        r = await cognee.search(
            query_text=q, query_type=cognee.SearchType.GRAPH_COMPLETION,
            datasets=datasets, top_k=top, only_context=True,
            neighborhood_depth=2, neighborhood_seed_top_k=min(top, 10),
        )
        return [str(item) for item in r if str(item).strip()], r
    # answer: 图上下文 + DeepSeek 综合答案。
    r = await cognee.search(
        query_text=q, query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=datasets, top_k=top,
    )
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
    ap.add_argument("--mode", choices=["evidence", "graph-evidence", "answer"],
                    default="graph-evidence", help="检索模式（默认 graph-evidence）")
    ap.add_argument("--format", choices=["text", "json"], default="text",
                    help="输出格式（默认 text）")
    ap.add_argument("--evidence", action="store_true", help="等价于 --mode evidence")
    ap.add_argument("--graph-evidence", action="store_true", help="等价于 --mode graph-evidence")
    ap.add_argument("--answer", action="store_true", help="等价于 --mode answer")
    ap.add_argument("--chunks", action="store_true", help="兼容旧参数，等价于 --mode evidence")
    args = ap.parse_args()

    if args.domain == "memory":
        print(f"[记忆域] 经 memory_rag 检索: {args.query}")
        print(search_memory(args.query, args.top))
        return

    aliases = [args.evidence or args.chunks, args.graph_evidence, args.answer]
    if sum(bool(value) for value in aliases) > 1:
        ap.error("--evidence/--chunks、--graph-evidence、--answer 只能选一个")
    mode = ("evidence" if aliases[0] else
            "graph-evidence" if aliases[1] else
            "answer" if aliases[2] else args.mode)
    datasets = args.datasets or DATASET

    if mode == "answer":
        answer, raw = await search_knowledge(args.query, args.top, datasets, mode=mode)
        if args.format == "json":
            print(json.dumps({"mode": mode, "query": args.query, "datasets": datasets,
                              "answer": answer}, ensure_ascii=False, indent=2))
        else:
            print(f"[知识域] GRAPH 图增强答案: {args.query} | datasets={datasets}")
            print(f"\n{answer}" if answer else "\n(图谱检索未召回相关内容)")
        return

    results, raw = await search_knowledge(args.query, args.top, datasets, mode=mode)
    if mode == "evidence":
        boosted = boost(results) if results else []
        payload = {"mode": mode, "query": args.query, "datasets": datasets,
                   "evidence": boosted}
        if args.format == "json":
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(f"[知识域] CHUNKS 原始证据: {args.query} | datasets={datasets}")
        for i, item in enumerate(boosted, 1):
            src = f"\n    来源: {item['source']}" if item.get("source") else ""
            print(f"\n  [{i}] {item.get('title', '')}{src}\n"
                  f"    {item.get('content', '').replace(chr(10), chr(10) + '    ')}")
        return

    payload = {"mode": mode, "query": args.query, "datasets": datasets,
               "graph_context": results}
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"[知识域] GRAPH 图关联证据（无LLM回答）: {args.query} | datasets={datasets}")
    print("\n\n".join(results) if results else "\n(图谱未召回关联证据)")

if __name__ == "__main__":
    asyncio.run(main())

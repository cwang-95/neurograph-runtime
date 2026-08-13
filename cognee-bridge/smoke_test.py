#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试 — 记忆图谱2.0 方案① 全链路验收
测: cognee 检索(wiki_full 223篇) + ZenBrain 加权, 多个专业查询
"""
import os, asyncio, sys, json
import litellm; litellm.drop_params = True
DS_KEY = os.environ.get("DASHSCOPE_API_KEY") or ""
DS_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
for k, v in {
    "LLM_PROVIDER": "openai", "LLM_MODEL": "openai/qwen-plus",
    "LLM_API_KEY": DS_KEY, "LLM_ENDPOINT": DS_BASE,
    "EMBEDDING_PROVIDER": "openai_compatible", "EMBEDDING_MODEL": "text-embedding-v4",
    "EMBEDDING_API_KEY": DS_KEY, "EMBEDDING_ENDPOINT": DS_BASE,
    "EMBEDDING_DIMENSIONS": "1024", "ENABLE_BACKEND_ACCESS_CONTROL": "false",
}.items(): os.environ.setdefault(k, v)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zenbrain_llm import boost

QUERIES = [
    "自适应放疗",
    "深度学习 危及器官 自动勾画",
    "CBCT 图像配准",
    "宫颈癌 放疗",
    "剂量预测",
]

def extract(x):
    inner = getattr(x, "search_result", None) or x
    try:
        if isinstance(inner, dict):
            for k in ("title","text","part","content","description","name"):
                v = inner.get(k)
                if isinstance(v,str) and v.strip(): return v[:80]
            for v in inner.values():
                if isinstance(v,(dict,list)):
                    t = extract(v)
                    if t: return t
        elif isinstance(inner,list):
            for it in inner[:10]:
                t = extract(it)
                if t: return t
    except Exception: pass
    return str(inner)[:60]

async def main():
    import cognee
    print(f"cognee {cognee.__version__} | 数据集 wiki_full | 冒烟测试\n")
    all_results = {}
    for q in QUERIES:
        print(f"--- 查询: {q} ---")
        try:
            r = await cognee.search(query_text=q, query_type=cognee.SearchType.CHUNKS, datasets="wiki_full")
            results = [{"id": str(getattr(x,"dataset_id","") or ""), "title": extract(x)} for x in r[:4]]
            boosted = boost(results) if results else []
            print(f"  召回 {len(r)} 条, ZenBrain排序:")
            for i, x in enumerate(boosted[:4], 1):
                print(f"    {i}. [{x.get('_classed_score','?')}分] {x.get('title','')[:70]}")
            all_results[q] = [x.get("title") for x in boosted[:4]]
        except Exception as ex:
            print(f"   ✗ 失败: {repr(ex)[:150]}")
        print()
    print("=== 冒烟测试完成 ===")
    with open(os.path.join(HERE,"..","data","smoke_test.json"),"w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())

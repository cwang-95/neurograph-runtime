#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案① ZenBrain 外挂 — cognee 检索结果 + 类脑加权排序
思路: cognee 查出候选文献 → 每条挂 FSRS 调度(初始 normal_decay)
      → 用 ZenBrain(经 node zenbrain_scheduler.js)算"可回忆度/新鲜度"
      → 按类脑加权排序: 常查/新近的靠前, 久不碰的降权(但仍保留)
用法: ~/.cognee-venv/bin/python -u zenbrain_llm.py  (自测)
      (或 import ZenBrainBoost 供 query.py 调用)
"""
import os, json, subprocess, time, glob

NODE = "/usr/local/bin/node"
SCHED = os.path.expanduser("~/.openclaw/workspace/memory_rag/server/zenbrain_scheduler.js")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zenbrain_state.json")

def _node(args):
    """调 node zenbrain_scheduler.js, 返回 dict"""
    r = subprocess.run([NODE, SCHED] + args, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return {"error": r.stderr[-200:]}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": r.stdout[-200:]}

def new_scheduler():
    """给一条新知识建 FSRS 调度"""
    r = _node(["new"])
    return r.get("schedulers", {})

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def boost(results):
    """
    cognee 检索候选 results(list of dict, 每条含 id/title) →
    用 FSRS 算每条可回忆度, 返回带 classed_boost 的排序结果
    规则: 可回忆度高(常看/新近)→ 保持靠前; 低(久不碰)→ 降到后面但仍返回
    """
    state = load_state()
    enriched = []
    for r in results:
        rid = str(r.get("id") or r.get("title") or "")
        if rid not in state:
            # 新知识: 初始化 FSRS + 记录首次加入
            state[rid] = {"schedulers": new_scheduler(), "firstSeenAt": time.time(), "hitCount": 0}
        # 这次被召回 → 强化(Hebbian + FSRS recall), 记命中
        state[rid]["hitCount"] = state[rid].get("hitCount", 0) + 1
        mem = {"schedulers": state[rid]["schedulers"]}
        rec = _node(["recall", json.dumps(mem), "1", "4"])
        if "stability" in rec:
            state[rid]["schedulers"] = rec  # 更新后调度
            state[rid]["lastRecalledAt"] = time.time()
        # 算衰减后的可回忆度(未来7天)
        decay = _node(["decay", json.dumps(mem), "7"])
        retriev = float(decay.get("retrievabilityPct", 100))
        # 类脑评分: 可回忆度 + 命中强化(封顶)
        classed = retriev + min(state[rid]["hitCount"] * 2, 30)
        enriched.append({**r, "_retrievability_pct": retriev, "_classed_score": round(classed, 1)})
    save_state(state)
    # 按类脑分数降序
    enriched.sort(key=lambda x: x["_classed_score"], reverse=True)
    return enriched

def selfcheck():
    r = _node(["selfcheck"])
    print("ZenBrain selfcheck:", r)
    # 小测
    fake = [{"id": "test-1", "title": "自适应放疗"}, {"id": "test-2", "title": "旧知识"}]
    boosted = boost(fake)
    print("boost 排序:", [(x["title"], x["_classed_score"]) for x in boosted])
    return r.get("ok", False)

if __name__ == "__main__":
    print("ZenBrain 外挂自检...")
    selfcheck()

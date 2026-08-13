#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案① 图谱可视化 — 读 cognee 认知图谱, 画实体关系网 PNG
用 get_triplets_batch 分页读全图, 筛出有名字的知识实体(Entity/EntityType),
画出概念之间的关联(真正的知识图谱, 不是文档结构)。
用法: ~/.cognee-venv/bin/python -u visualize.py [--top 40] [--out xxx.png]
matplotlib 装 /tmp/viz_pkgs, 脚本自动加入 sys.path
"""
import os, sys, asyncio, argparse, json
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "ladybug")
for p in ("/tmp/viz_pkgs",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data")

def find_zh_font():
    for c in ("/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc",
              "/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/Supplemental/Songti.ttc"):
        if os.path.exists(c): return c
    return None

async def load_triplets():
    """分页读全部三元组"""
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
    e = await get_graph_engine()
    triplets, off = [], 0
    while True:
        batch = await e.get_triplets_batch(off, 1000)
        if not batch: break
        triplets.extend(batch)
        off += len(batch)
        if off > 30000: break  # 安全上限
    return triplets

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40, help="画最核心N个实体")
    ap.add_argument("--min-freq", type=int, default=3, help="实体至少出现次数才画")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    zh = find_zh_font()
    if zh:
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(zh)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=zh).get_name()
            plt.rcParams["axes.unicode_minus"] = False
        except Exception: pass

    triplets = await load_triplets()
    print(f"读到 {len(triplets)} 条三元组")

    # 收集实体出现频次 + 实体间连接(共现于一条 triplet)
    ent_freq = {}      # name -> count
    ent_type = {}      # name -> main type(Entity/EntityType)
    ent_edge = {}      # (a,b) -> count
    for t in triplets:
        s, en = t.get("start_node", {}), t.get("end_node", {})
        for node in (s, en):
            nm = node.get("name", "")
            if nm and node.get("type") in ("Entity", "EntityType"):
                ent_freq[nm] = ent_freq.get(nm, 0) + 1
                if nm not in ent_type: ent_type[nm] = node.get("type")
    print(f"命名实体(去重): {len(ent_freq)} 种")

    # 实体间的邻接(同一 triplet 两个实体相连)
    for t in triplets:
        s, en = t.get("start_node", {}), t.get("end_node", {})
        a = s.get("name", "") if s.get("type") in ("Entity", "EntityType") else ""
        b = en.get("name", "") if en.get("type") in ("Entity", "EntityType") else ""
        if a and b and a != b:
            key = tuple(sorted([a, b]))
            ent_edge[key] = ent_edge.get(key, 0) + 1

    # 过滤通用类型标签/关系噪声词(非内容实体)
    NOISE = {"concept","person","organization","technology","technique","metric","date",
             "publication","entity","entitytype","category","method","system","type",
             "name","relation","relationship","attribute","property","event","time",
             "location","number","text","url","id","status","value","group","class",
             "assessment","measurement","process","task","tool","model","paper","case"}
    ent_freq = {nm:c for nm,c in ent_freq.items() if nm.lower() not in NOISE}
    # 选出现频次靠前的实体(过滤掉过稀的)
    freq_items = sorted(ent_freq.items(), key=lambda x: -x[1])
    chosen = [(nm, c) for nm, c in freq_items if c >= args.min_freq][:args.top]
    if not chosen:
        print("!! 没有满足 min_freq 的实体, 降低阈值重试")
        chosen = freq_items[:args.top]
    chosen_names = {nm for nm, _ in chosen}
    print(f"选中前{len(chosen)}个高频实体: {[(nm,c) for nm,c in chosen[:15]]}")

    G = nx.Graph()
    G.add_nodes_from([nm for nm, _ in chosen])
    for (a, b), c in ent_edge.items():
        if a in chosen_names and b in chosen_names:
            G.add_edge(a, b, weight=c)
    print(f"构图: {G.number_of_nodes()}节点 {G.number_of_edges()}边")

    # 画图
    pos = nx.spring_layout(G, k=1.4, iterations=60, seed=42)
    plt.figure(figsize=(20, 15), dpi=130)
    deg = dict(G.degree())
    maxdeg = max(deg.values()) if deg else 1
    node_sizes = [300 + 1200 * deg.get(n, 0) / maxdeg for n in G.nodes()]
    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    wmin, wmax = (min(weights), max(weights)) if weights else (1, 1)
    edge_w = [0.6 + 2.5 * (w - wmin) / max(wmax - wmin, 1) for w in weights]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue",
                           edgecolors="steelblue", alpha=0.92)
    nx.draw_networkx_edges(G, pos, alpha=0.35, edge_color="gray", width=edge_w)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family=plt.rcParams.get("font.family"))
    out = args.out or os.path.join(OUT_DIR, "graph_wiki_full.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.title(f"cognee 认知图谱·高频实体关系网 (总实体{len(ent_freq)}种 / 展示{len(chosen)}个)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"✓ 图已保存: {out}")
    with open(out.replace(".png", ".json"), "w") as f:
        json.dump({"total_entities": len(ent_freq), "shown": len(chosen),
                   "top_entities": [{"name": nm, "freq": c} for nm, c in chosen[:30]]},
                  f, ensure_ascii=False, indent=2)
    print("✓ JSON 已保存")

if __name__ == "__main__":
    asyncio.run(main())

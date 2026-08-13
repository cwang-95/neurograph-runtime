#!/bin/bash
# ============================================================
# 记忆图谱 2.0 (方案① cognee) — 一键复现脚本
# 用法: bash setup_repro.sh
# 作用: 从零搭建 venv + 配置 env + 回灌文献(可选) + 验证检索
# 前提: 已有 cognee-bridge/ 脚本 + DeepSeek/DashScope key
# 说明: 详细每一步的"为什么"见 TECH.md
# ============================================================
set -e

echo "=== [1/6] 创建 venv + 安装依赖(版本锁定见 TECH.md 1.1) ==="
if [ ! -d ~/.cognee-venv ]; then
  python3 -m venv ~/.cognee-venv
fi
~/.cognee-venv/bin/pip install --quiet "cognee==1.4.2" litellm networkx
# matplotlib 隔离安装(不污染 venv)
if [ ! -d /tmp/viz_pkgs ]; then
  ~/.cognee-venv/bin/pip install --quiet --target /tmp/viz_pkgs matplotlib
fi
echo "  ✓ venv 就绪: ~/.cognee-venv"
echo "  ✓ matplotlib: /tmp/viz_pkgs"

echo ""
echo "=== [2/6] 配置环境变量(证书 + NO_PROXY) ==="
# TUN 模式已全局接管（2026-08-14 起），无需显式 1097 代理
export SSL_CERT_FILE=/etc/ssl/cert.pem CURL_CA_BUNDLE=/etc/ssl/cert.pem
export NO_PROXY="us.i.posthog.com,api.posthog.com,127.0.0.1,localhost"
echo "  ✓ 证书/NO_PROXY 已设(TUN 全局接管，无显式代理)"

echo ""
echo "=== [3/6] DashScope embedding key ==="
if [ -z "$DASHSCOPE_API_KEY" ]; then
  export DASHSCOPE_API_KEY=$(python3 -c "import json;print(json.load(open('$HOME/.openclaw/openclaw.json'))['env']['DASHSCOPE_API_KEY'])" 2>/dev/null) || true
fi
if [ -z "$DASHSCOPE_API_KEY" ]; then
  echo "  ⚠️ 未找到 DASHSCOPE_API_KEY, 请手动 export"
  exit 1
fi
echo "  ✓ DashScope key 就绪"

echo ""
echo "=== [4/6] 回灌文献到 wiki_full(可加 --no-cognify 只灌不建图) ==="
cd ~/.openclaw/workspace/projects/neurograph/cognee-bridge
if [ -d ~/wiki/raw/papers ] && [ "$(ls ~/wiki/raw/papers/*.md 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "  灌入 $(ls ~/wiki/raw/papers/*.md | wc -l) 篇文献..."
  ~/.cognee-venv/bin/python -u feed_wiki.py wiki_full --no-cognify
  echo "  ✓ 文献已灌(本次 skip cognify, 建图命令: feed_wiki.py wiki_full)"
else
  echo "  ⚠️ 无本地文献目录 ~/wiki/raw/papers, 跳过回灌(仅当需重建时执行)"
fi

echo ""
echo "=== [5/6] 验证检索(文献+项目跨数据集) ==="
DSKEY="$DASHSCOPE_API_KEY" ~/.cognee-venv/bin/python -u query.py "自适应放疗" --top 3 --datasets wiki_full,github_projects 2>/dev/null | head -8 || echo "  ⚠️ 检索失败, 看 TECH.md 踩坑1-7"

echo ""
echo "=== [6/6] 完成 ==="
echo "  日常检索: bash scripts/kb_search \"查询\" [topN] [--all|--github|--memory]"
echo "  技术细节: 见 projects/neurograph/TECH.md"
echo "  复现完成 ✅"

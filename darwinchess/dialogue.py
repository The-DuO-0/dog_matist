from __future__ import annotations

from dataclasses import asdict
from typing import Any
import json
import re
import urllib.error
import urllib.request

import chess

from .runtime import DarwinRuntime
from .search import SearchResult


def explain_search(board: chess.Board, result: SearchResult) -> str:
    if result.move is None:
        return "这个局面已经结束，没有合法着法。"
    move = result.move
    san = board.san(move)
    reasons: list[str] = []
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        reasons.append("这是吃子" + (f"，目标是 {captured.symbol()}" if captured else ""))
    if board.gives_check(move):
        reasons.append("它同时将军")
    if board.is_castling(move):
        reasons.append("它完成王车易位，改善王安全并连接车")
    if move.promotion:
        reasons.append("它完成兵的升变")
    if not reasons:
        reasons.append("搜索认为它在当前深度下保留了最好的综合局面评价")

    cand = []
    for c in result.candidates[:4]:
        try:
            csan = board.san(c.move)
        except Exception:
            csan = c.move.uci()
        cand.append(f"{csan} {c.score_cp:+.0f}cp")
    alternatives = "；候选：" + ", ".join(cand) if cand else ""
    pv = ""
    if result.pv:
        b = board.copy(stack=False)
        sans: list[str] = []
        for m in result.pv[:6]:
            if m not in b.legal_moves:
                break
            sans.append(b.san(m))
            b.push(m)
        if sans:
            pv = "；主变化：" + " ".join(sans)
    return (
        f"我会走 {san}（{move.uci()}）。当前搜索评价约 {result.score_cp:+.0f}cp，"
        + "；".join(reasons)
        + f"。搜索深度 {result.depth}，访问 {result.nodes} 个节点{alternatives}{pv}。"
    )


class DialogueAgent:
    def __init__(self, runtime: DarwinRuntime):
        self.runtime = runtime

    def _ollama(self, query: str, context: str) -> str | None:
        cfg = self.runtime.config.get("dialogue", {})
        model = cfg.get("ollama_model")
        if not model:
            return None
        url = str(cfg.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/") + "/api/generate"
        prompt = (
            "你是 DarwinChess，一个会持续学习的国际象棋 agent。基于以下真实状态回答，"
            "不得虚构训练进度、棋力或记忆。\n\n"
            f"STATE:\n{context}\n\nUSER:\n{query}\n"
        )
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return str(payload.get("response", "")).strip() or None
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None

    def respond(self, query: str) -> str:
        """Public alias for agent-style integrations."""
        return self.answer(query)

    def answer(self, query: str) -> str:
        q = query.strip()
        status = self.runtime.status()
        ql = q.lower()

        if any(k in ql for k in ["status", "状态", "多强", "generation", "第几代", "进度"]):
            return (
                f"我现在的 champion 是 generation {status['champion_generation']}，"
                f"已经保存 {status['games']} 盘棋和 {status['replay_examples']} 个训练局面。"
                f"当前运行模式是 {status['mode']}，计算设备是 {status['device']}。"
            )

        if any(k in ql for k in ["学到", "learn", "最近", "记得", "memory", "反思"]):
            insights = status.get("recent_insights", [])
            if not insights:
                return "我还没有足够的长期记录生成反思。先让我完成一些 self-play，我会把统计和继位/淘汰事件写进长期记忆。"
            return "我最近留下的长期记忆包括：\n- " + "\n- ".join(insights[:6])

        fen_match = re.search(r"(?:fen\s*[:=]?\s*)(.+)$", q, flags=re.IGNORECASE)
        if fen_match:
            fen = fen_match.group(1).strip()
            try:
                board = chess.Board(fen)
                result = self.runtime.analyze(fen, top_n=5)
                return explain_search(board, result)
            except ValueError:
                return "这个 FEN 我没法解析。请把完整的 6 段 FEN 发给我。"

        context = json.dumps(status, ensure_ascii=False, indent=2)
        llm_answer = self._ollama(q, context)
        if llm_answer:
            return llm_answer

        return (
            "我现在的对话层是 chess-native：我能直接告诉你自己的训练状态、最近学到的东西、"
            "分析 FEN、解释为什么选某步，而且这些回答会读取真实数据库而不是编故事。"
            "如果以后在配置里接一个本地 Ollama 模型，我还能把同一份真实状态变成更自由的自然语言交流。"
        )

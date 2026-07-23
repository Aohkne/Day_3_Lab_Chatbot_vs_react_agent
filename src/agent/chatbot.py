"""
Chatbot Baseline — Gọi LLM 1 lần duy nhất, KHÔNG có tool.
Baseline so sánh với ReAct Agent trong lĩnh vực nghiên cứu học thuật.
Không có tool → có thể hallucinate thông tin bài báo cụ thể.
Provider (OpenAI/Gemini/Local) được chọn qua get_provider() — xem provider_factory.py.
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from dotenv import load_dotenv
from src.core.provider_factory import get_provider
from src.telemetry.logger import logger

load_dotenv()

SYSTEM_PROMPT = """Ban la tro ly nghien cuu hoc thuat thong minh, ho tro sinh vien va nha nghien cuu tim hieu bai bao khoa hoc.
Ban co kien thuc ve cac linh vuc: NLP, Computer Vision, Reinforcement Learning, Education Technology.
Hay tra loi ngan gon, hoc thuat, bang tieng Viet.
LUU Y: Ban khong co cong cu tim kiem, vi vay cau tra loi dua tren kien thuc tong quat cua ban."""


def run_chatbot(user_input: str, system_prompt: Optional[str] = None, provider: Optional[str] = None) -> dict:
    """
    Chạy chatbot baseline: 1 system prompt + 1 lần gọi LLM duy nhất.
    system_prompt: nếu có, ghi đè SYSTEM_PROMPT mặc định.
    provider: "openai" | "google" | "local". Nếu None, đọc từ env DEFAULT_PROVIDER
              (mặc định "local"). Cho phép swap provider mà không cần sửa code,
              chỉ cần đổi provider ở đây hoặc trong .env.
    Returns: dict chứa content, usage, latency_ms
    """
    logger.log_event("CHATBOT_START", {"input": user_input, "provider": provider or os.getenv("DEFAULT_PROVIDER", "local")})

    effective_prompt = system_prompt if system_prompt else SYSTEM_PROMPT

    llm = get_provider(provider_name=provider)
    result = llm.generate(user_input, system_prompt=effective_prompt)

    logger.log_event("CHATBOT_END", {
        "output": result["content"][:200],
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    })

    return result


# Run directly
if __name__ == "__main__":
    print("=" * 60)
    print("CHATBOT BASELINE - Tro Ly Nghien Cuu Hoc Thuat")
    print("=" * 60)

    test_queries = [
        "Attention mechanism trong Transformer la gi?",
        "BERT va GPT khac nhau nhu the nao?",
    ]

    for query in test_queries:
        print(f"\n[User]: {query}")
        result = run_chatbot(query)
        print(f"[Chatbot]: {result['content']}")
        print(f"  >> {result['latency_ms']:.0f}ms | {result['usage']['total_tokens']} tokens")
        print("-" * 60)

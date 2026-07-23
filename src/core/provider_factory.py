"""
Provider Factory — chọn LLMProvider (OpenAI / Gemini / Local) dựa vào biến môi trường.
Cho phép đổi provider chỉ bằng cách sửa .env (DEFAULT_PROVIDER=openai|google|local),
không cần sửa code ở chatbot.py / agent.py / main.py.
"""
import os
from typing import Optional

from src.core.llm_provider import LLMProvider

VALID_PROVIDERS = ("openai", "google", "local")


def get_provider(provider_name: Optional[str] = None, model_name: Optional[str] = None) -> LLMProvider:
    """
    Khoi tao LLMProvider theo cau hinh.

    Args:
        provider_name: "openai" | "google" | "local". Neu None, doc tu env DEFAULT_PROVIDER
                       (mac dinh "local" neu khong set).
        model_name: ten model cu the. Neu None, doc tu env DEFAULT_MODEL hoac dung
                    default cua tung provider.

    Returns:
        Mot instance LLMProvider (OpenAIProvider / GeminiProvider / LocalProvider)
        deu implement cung interface generate()/stream(), nen agent/chatbot dung
        chung mot code path bat ke provider nao duoc chon.
    """
    name = (provider_name or os.getenv("DEFAULT_PROVIDER") or "local").strip().lower()

    if name == "openai":
        from src.core.openai_provider import OpenAIProvider
        model = model_name or os.getenv("DEFAULT_MODEL") or "gpt-4o"
        return OpenAIProvider(model_name=model, api_key=os.getenv("OPENAI_API_KEY"))

    if name in ("google", "gemini"):
        from src.core.gemini_provider import GeminiProvider
        model = model_name or os.getenv("DEFAULT_MODEL") or "gemini-1.5-flash"
        return GeminiProvider(model_name=model, api_key=os.getenv("GEMINI_API_KEY"))

    if name == "local":
        from src.core.local_provider import LocalProvider
        model_path = os.getenv("LOCAL_MODEL_PATH") or "./models/Phi-3-mini-4k-instruct-q4.gguf"
        return LocalProvider(model_path=os.path.normpath(model_path), n_ctx=4096)

    raise ValueError(
        f"Provider '{name}' khong hop le. Cac gia tri hop le: {', '.join(VALID_PROVIDERS)}"
    )

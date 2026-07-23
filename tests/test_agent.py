"""
Tests cho vong lap ReAct (agent.py, agent_v1.py) dung mot LLMProvider gia lap
(ScriptedLLM) tra ve chuoi response da lap trinh san — khong can tai model that,
chay duoc o moi moi truong.
"""
import os
import sys
from typing import Any, Dict, Generator, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import ReActAgent
from src.agent.agent_v1 import ReActAgentV1
from src.core.llm_provider import LLMProvider


class ScriptedLLM(LLMProvider):
    """Fake provider: moi lan generate() tra ve response tiep theo trong kich ban."""

    def __init__(self, responses: List[str]):
        super().__init__(model_name="scripted-test-model")
        self._responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        self.call_count += 1
        content = self._responses.pop(0) if self._responses else "Final Answer: het kich ban"
        return {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "latency_ms": 5,
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        yield self.generate(prompt, system_prompt)["content"]


DUMMY_TOOLS = [
    {"name": "search_papers", "description": "test tool", "function": lambda args: f"Ket qua cho {args}"},
]


def test_agent_executes_tool_then_returns_final_answer():
    responses = [
        "Thought: can tim kiem\nAction: search_papers(transformer)",
        "Thought: da co du lieu\nFinal Answer: Tra loi dua tren ket qua tool",
    ]
    llm = ScriptedLLM(responses)
    agent = ReActAgent(llm=llm, tools=DUMMY_TOOLS, max_steps=5)
    result = agent.run("Tim bai ve transformer")

    assert result["answer"] == "Tra loi dua tren ket qua tool"
    assert result["steps"] == 2
    assert agent.trace[0]["action"] == "search_papers(transformer)"
    assert "Ket qua cho transformer" in agent.trace[0]["observation"]


def test_agent_parse_error_is_recorded_and_loop_continues():
    responses = [
        "day khong dung format gi ca",
        "Final Answer: cuoi cung cung tra loi duoc",
    ]
    llm = ScriptedLLM(responses)
    agent = ReActAgent(llm=llm, tools=DUMMY_TOOLS, max_steps=5)
    result = agent.run("cau hoi bat ky")

    assert result["answer"] == "cuoi cung cung tra loi duoc"
    assert agent.trace[0]["parse_error"] is True


def test_agent_timeout_after_max_steps():
    responses = ["Thought: van dang nghi\nAction: search_papers(x)"] * 10
    llm = ScriptedLLM(responses)
    agent = ReActAgent(llm=llm, tools=DUMMY_TOOLS, max_steps=2)
    result = agent.run("cau hoi lap vo han")

    assert result["steps"] == 2
    assert llm.call_count == 2


def test_agent_v1_rule4_blocks_final_answer_without_tool_call():
    responses = [
        "Thought: toi biet roi\nFinal Answer: BERT do Devlin viet nam 2019",  # bi RULE 4 chan
        "Thought: ok goi tool\nAction: search_papers(BERT)",
        "Thought: co du lieu\nFinal Answer: BERT (P002) theo ket qua tool",
    ]
    llm = ScriptedLLM(responses)
    agent = ReActAgentV1(llm=llm, tools=DUMMY_TOOLS, max_steps=4)
    result = agent.run("Tac gia bai bao BERT la ai")

    assert result["answer"] == "BERT (P002) theo ket qua tool"
    assert agent.trace[0]["constraint_triggered"] == "RULE_4"


def test_agent_v1_general_question_skips_rule4():
    responses = [
        "Thought: day la cau hoi khai niem\nFinal Answer: Attention la co che...",
    ]
    llm = ScriptedLLM(responses)
    agent = ReActAgentV1(llm=llm, tools=DUMMY_TOOLS, max_steps=4)
    result = agent.run("Attention mechanism la gi?")

    assert result["answer"] == "Attention la co che..."
    assert "constraint_triggered" not in agent.trace[0]

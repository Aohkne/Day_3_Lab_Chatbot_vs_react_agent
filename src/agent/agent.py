"""
ReAct Agent — Phien ban day du: tool calls + system prompt hoc thuat.
Su dung vong lap Thought-Action-Observation de tim kiem va phan tich bai bao.
Dung local model: Phi-3-mini-4k-instruct-q4.gguf
"""
import os
import re
from typing import List, Dict, Any
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger


class ReActAgent:
    """
    ReAct Agent voi day du tool calls, trace logging va error handling.
    Domain: Tim kiem va phan tich bai bao khoa hoc.
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5, extra_instructions: str = ""):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.extra_instructions = extra_instructions
        self.trace: List[Dict[str, Any]] = []

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )
        base = f"""Bạn là trợ lý nghiên cứu học thuật thông minh. Bạn có các công cụ sau:

{tool_descriptions}

Luôn tuân theo định dạng sau:
Thought: [phân tích yêu cầu và quyết định bước tiếp theo bằng tiếng Việt có dấu]
Action: tool_name(arguments)
Observation: [kết quả từ tool]
... (lặp lại nếu cần)
Final Answer: [câu trả lời đầy đủ bằng tiếng Việt có dấu, dựa trên kết quả tool]

Hướng dẫn:
- Gọi tool trước khi trả lời câu hỏi về bài báo cụ thể
- Viết Thought và Final Answer bằng tiếng Việt có dấu
- Khi đã có Final Answer, dừng ngay"""
        if self.extra_instructions:
            base += f"\n\nHướng dẫn bổ sung từ người dùng:\n{self.extra_instructions}"
        return base

    def run(self, user_input: str) -> dict:
        """
        Chay vong lap ReAct: Thought -> Action -> Observation -> Final Answer.
        Returns: dict voi answer, trace, total_tokens, total_latency_ms, steps
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        self.trace = []
        conversation = user_input
        total_tokens = 0
        total_latency = 0

        for step in range(1, self.max_steps + 1):
            result = self.llm.generate(conversation, system_prompt=self.get_system_prompt())
            ai_response = result["content"]
            total_tokens += result["usage"].get("total_tokens", 0)
            total_latency += result["latency_ms"]

            step_info: Dict[str, Any] = {
                "step": step,
                "llm_output": ai_response,
                "tokens": result["usage"],
                "latency_ms": result["latency_ms"],
            }

            # Check Final Answer
            if "Final Answer:" in ai_response:
                final_answer = ai_response.split("Final Answer:")[-1].strip()
                step_info["final_answer"] = final_answer
                self.trace.append(step_info)
                logger.log_event("AGENT_END", {"steps": step, "tokens": total_tokens})
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "total_tokens": total_tokens,
                    "total_latency_ms": total_latency,
                    "steps": step,
                }

            # Parse Action: tool_name(arguments)
            action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', ai_response)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_args = action_match.group(2).strip().strip('"').strip("'")

                observation = self._execute_tool(tool_name, tool_args)
                step_info["action"] = f"{tool_name}({tool_args})"
                step_info["observation"] = observation

                logger.log_event("AGENT_TOOL_CALL", {
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "obs_len": len(observation),
                })

                # Truncate observation to avoid exceeding context window
                obs_truncated = observation[:800] + "..." if len(observation) > 800 else observation
                conversation += f"\n{ai_response}\nObservation: {obs_truncated}\n"
            else:
                # Parse error: remind the model of the expected format
                logger.log_event("AGENT_PARSE_ERROR", {"step": step, "response": ai_response[:200]})
                step_info["parse_error"] = True
                conversation += (
                    f"\n{ai_response}\n"
                    "Observation: [Parse error] Vui long tuan theo format: Action: tool_name(arguments)\n"
                )

            self.trace.append(step_info)

        # Exhausted max steps
        logger.log_event("AGENT_TIMEOUT", {"steps": self.max_steps})
        last_out = self.trace[-1]["llm_output"] if self.trace else ""
        fallback = last_out.split("Thought:")[-1].split("Action:")[0].strip() or last_out[:400]
        return {
            "answer": fallback or "Da het so buoc toi da. Hay thu cau hoi cu the hon.",
            "trace": self.trace,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "steps": self.max_steps,
        }

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """Goi tool theo ten, xu ly exception."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                try:
                    return tool["function"](args)
                except Exception as exc:
                    logger.log_event("TOOL_ERROR", {"tool": tool_name, "error": str(exc)})
                    return f"Tool '{tool_name}' gap loi: {exc}"
        available = ", ".join(t["name"] for t in self.tools)
        return f"Tool '{tool_name}' khong ton tai. Cac tool co san: {available}"

    def get_trace_report(self) -> str:
        """Tra ve trace dang text de in ra terminal."""
        lines = []
        for s in self.trace:
            lines.append(f"  [Step {s['step']}]")
            text = s["llm_output"]
            if "Thought:" in text:
                thought = text.split("Thought:")[-1].split("Action:")[0].split("Final Answer:")[0].strip()
                lines.append(f"    Thought: {thought[:150]}")
            if "action" in s:
                lines.append(f"    Action : {s['action']}")
            if "observation" in s:
                obs_preview = s["observation"][:120].replace("\n", " ")
                lines.append(f"    Obs    : {obs_preview}...")
            if "final_answer" in s:
                lines.append(f"    Final  : {s['final_answer'][:150]}")
        return "\n".join(lines)


"""
ReAct Agent V1 — Phien ban nang cao voi rang buoc chong hallucination.

So sanh 3 approach:
  chatbot.py  : Khong tool, 1 lan goi LLM → de hallucinate thong tin bai bao
  agent.py    : ReAct + tools + system prompt co ban
  agent_v1.py : ReAct + tools + STRICT anti-hallucination rules (file nay)

Cac RULE chong hallucination:
  Chi trich dan thong tin tu tool output, cam bia thong tin bai bao
  Neu tool tra "Khong tim thay" → noi thang khong biet, KHONG doan
  Khong tu dat Paper ID (P001...) neu chua nhan duoc tu tool output
  Phai goi it nhat 1 tool truoc khi tra loi ve bai bao cu the
  Cau hoi tong quat (dinh nghia, khai niem) → co the tra loi truc tiep
  Toi da {max_steps} buoc, qua gioi han → khai bao ro rang
"""
import os
import re
from typing import List, Dict, Any
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger


class ReActAgentV1:
    """
    ReAct Agent V1 voi cac rang buoc chat che chong hallucination.
    Max steps ngan hon agent.py (4 vs 5) de ep agent hieu qua hon.
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 4, extra_instructions: str = ""):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.extra_instructions = extra_instructions
        self.trace: List[Dict[str, Any]] = []

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )
        base = f"""Bạn là trợ lý nghiên cứu học thuật NGHIÊM NGẶT. Bạn có các công cụ:

{tool_descriptions}

LUẬT BẮT BUỘC:
[RULE 1] CHỈ trích dẫn thông tin bài báo TỪ KẾT QUẢ TOOL. TUYỆT ĐỐI KHÔNG bịa đặt.
[RULE 2] Nếu tool trả "Không tìm thấy" → Final Answer phải nói: "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."
[RULE 3] KHÔNG tự đặt Paper ID (P001...) nếu chưa nhận được từ tool.
[RULE 4] PHẢI gọi ít nhất 1 tool trước Final Answer nếu câu hỏi về bài báo cụ thể.
[RULE 5] Câu hỏi định nghĩa/khái niệm tổng quát → có thể trả lời trực tiếp, không cần tool.
[RULE 6] Tối đa {self.max_steps} bước. Quá giới hạn → khai báo rõ ràng.

ĐỊNH DẠNG BẮT BUỘC:
Thought: [phân tích yêu cầu bằng tiếng Việt có dấu, quyết định có cần tool không]
Action: tool_name(arguments)
Observation: [kết quả từ tool]
... (lặp lại nếu cần, tối đa {self.max_steps} bước)
Final Answer: [câu trả lời CHÍNH XÁC dựa trên tool output, bằng tiếng Việt có dấu]

VÍ DỤ ĐÚNG:
Thought: Người dùng hỏi về bài báo BERT. Cần gọi search_papers trước.
Action: search_papers(BERT)
Observation: [P002] BERT... Devlin... 2019...
Final Answer: Theo kết quả tìm kiếm, bài báo BERT (P002) được viết bởi Devlin...

VÍ DỤ SAI (HALLUCINATION):
Thought: Tôi biết BERT rồi.
Final Answer: BERT được viết bởi... [SAI - chưa gọi tool]"""
        if self.extra_instructions:
            base += f"\n\nHướng dẫn bổ sung từ người dùng:\n{self.extra_instructions}"
        return base

    def run(self, user_input: str) -> dict:
        """
        Chay ReAct loop voi strict anti-hallucination constraints.
        Returns: dict voi answer, trace, total_tokens, total_latency_ms, steps
        """
        logger.log_event("AGENT_V1_START", {"input": user_input})

        self.trace = []
        conversation = user_input
        total_tokens = 0
        total_latency = 0
        tool_called = False

        # Detect general concept questions (don't need tool)
        general_keywords = ["la gi", "dinh nghia", "khai niem", "giai thich",
                            "what is", "explain", "how does", "tai sao", "vi sao"]
        is_general = any(kw in user_input.lower() for kw in general_keywords)

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

                # RULE 4: specific questions must use tool first
                if not tool_called and not is_general:
                    constraint_msg = (
                        "[V1 CONSTRAINT - RULE 4] Ban PHAI goi tool truoc khi "
                        "dua Final Answer ve bai bao cu the. Hay goi search_papers truoc."
                    )
                    conversation += f"\n{ai_response}\nObservation: {constraint_msg}\n"
                    step_info["constraint_triggered"] = "RULE_4"
                    logger.log_event("AGENT_V1_RULE4_TRIGGERED", {"step": step})
                    self.trace.append(step_info)
                    continue

                step_info["final_answer"] = final_answer
                self.trace.append(step_info)
                logger.log_event("AGENT_V1_END", {
                    "steps": step,
                    "tokens": total_tokens,
                    "tool_called": tool_called,
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "total_tokens": total_tokens,
                    "total_latency_ms": total_latency,
                    "steps": step,
                }

            # Parse Action
            action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', ai_response)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_args = action_match.group(2).strip().strip('"').strip("'")

                observation = self._execute_tool(tool_name, tool_args)
                tool_called = True

                # RULE 2: if tool found nothing, append reminder
                if "Khong tim thay" in observation or "not found" in observation.lower():
                    observation += (
                        "\n[V1 RULE 2] Tool khong tim thay ket qua. "
                        "Final Answer PHAI noi 'Toi khong tim thay' — KHONG doan."
                    )

                step_info["action"] = f"{tool_name}({tool_args})"
                step_info["observation"] = observation

                logger.log_event("AGENT_V1_TOOL_CALL", {
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                })
                # Truncate observation to avoid exceeding context window
                obs_truncated = observation[:800] + "..." if len(observation) > 800 else observation
                conversation += f"\n{ai_response}\nObservation: {obs_truncated}\n"
            else:
                logger.log_event("AGENT_V1_PARSE_ERROR", {"step": step})
                step_info["parse_error"] = True
                conversation += (
                    f"\n{ai_response}\n"
                    "Observation: [Parse error] Tuan theo format: Action: tool_name(arguments)\n"
                )

            self.trace.append(step_info)

        # RULE 6: max steps reached
        logger.log_event("AGENT_V1_TIMEOUT", {"steps": self.max_steps})
        return {
            "answer": (
                f"Hiện tại tui không tìm thấy thông tin chính xác. Hãy Thử lại sau nhé!"
            ),
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
                    return (
                        f"Tool '{tool_name}' gap loi: {exc}. "
                        "[V1 RULE 2] Khong duoc doan ket qua."
                    )
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
            if s.get("constraint_triggered"):
                lines.append(f"    [CONSTRAINT: {s['constraint_triggered']} triggered]")
        return "\n".join(lines)

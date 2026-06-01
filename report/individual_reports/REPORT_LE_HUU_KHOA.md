# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Lê Hữu Khoa
- **Student ID**: 2A202600863
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

### Modules Implemented

| File | Vai trò |
| :--- | :--- |
| `src/agent/chatbot.py` | Baseline: 1 lần gọi LLM, không tool, dùng `LocalProvider` (Phi-3-mini) |
| `src/agent/agent.py` | ReAct Agent cơ bản: vòng lặp Thought-Action-Observation, tối đa 5 bước |
| `src/agent/agent_v1.py` | ReAct Agent V1: thêm 6 RULE chống hallucination nghiêm ngặt, tối đa 4 bước |

### Code Highlights

**1. `chatbot.py` — Baseline LLM call (không tool)**

```python
# chatbot.py – run_chatbot()
def run_chatbot(user_input: str, system_prompt: Optional[str] = None) -> dict:
    logger.log_event("CHATBOT_START", {"input": user_input})
    effective_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
    llm = LocalProvider(model_path=model_path, n_ctx=4096)
    result = llm.generate(user_input, system_prompt=effective_prompt)
    logger.log_event("CHATBOT_END", {
        "output": result["content"][:200],
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    })
    return result
```

→ Chỉ gọi LLM 1 lần với `SYSTEM_PROMPT` cố định. Không có tool → câu trả lời về bài báo cụ thể dựa hoàn toàn vào kiến thức nội tại của model, dễ hallucinate.

**2. `agent.py` — ReAct Loop cơ bản**

```python
# agent.py – ReActAgent.run()
for step in range(1, self.max_steps + 1):   # max 5 bước
    result = self.llm.generate(conversation, system_prompt=self.get_system_prompt())
    ai_response = result["content"]

    if "Final Answer:" in ai_response:
        # Dừng vòng lặp, trả về kết quả
        ...

    action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', ai_response)
    if action_match:
        tool_name  = action_match.group(1).strip()
        tool_args  = action_match.group(2).strip()
        observation = self._execute_tool(tool_name, tool_args)
        logger.log_event("AGENT_TOOL_CALL", {"step": step, "tool": tool_name, ...})
        conversation += f"\n{ai_response}\nObservation: {observation}\n"
    else:
        logger.log_event("AGENT_PARSE_ERROR", {"step": step})
```

→ System prompt định nghĩa format `Thought / Action / Observation / Final Answer`. Regex parse `Action: tool_name(args)` → gọi tool → inject `Observation` vào context cho bước tiếp theo.

**3. `agent_v1.py`

```python
# agent_v1.py – RULE 4 check trước khi chấp nhận Final Answer
if "Final Answer:" in ai_response:
    if not tool_called and not is_general:
        # Chưa gọi tool mà đã trả lời bài báo cụ thể → chặn lại
        constraint_msg = (
            "[V1 RULE 4] Bạn PHẢI gọi tool trước khi đưa Final Answer "
            "về bài báo cụ thể. Hãy gọi search_papers trước."
        )
        conversation += f"\n{ai_response}\nObservation: {constraint_msg}\n"
        logger.log_event("AGENT_V1_RULE4_TRIGGERED", {"step": step})
        continue   # Tiếp tục vòng lặp, không chấp nhận Final Answer này
```

→ RULE 4 phân biệt câu hỏi khái niệm tổng quát (`is_general`) và câu hỏi về bài báo cụ thể. Nếu model cố trả lời trực tiếp không qua tool, inject thông báo lỗi và buộc lặp lại.

### Telemetry Data Extraction

Tôi cấu hình `logger.log_event()` ở mỗi điểm quan trọng của vòng lặp để trích xuất 3 chỉ số công nghiệp:

| Chỉ số | Event log | Ý nghĩa |
| :--- | :--- | :--- |
| **LLM Steps** | `steps` trong `AGENT_END`, `AGENT_V1_END` | Số vòng Thought-Action-Observation |
| **Total Tokens** | `tokens` trong các `*_END` events | Tổng tokens prompt + completion |
| **Latency (ms)** | `latency_ms` trong `CHATBOT_END` | Thời gian wall-clock từ gọi đến nhận kết quả |

---

## II. Debugging Case Study (10 Points)

### Problem Description

Trong quá trình chạy với model **Phi-3-mini-4k-instruct-q4.gguf** (local, 4-bit quantized), Agent V1 liên tục rơi vào `AGENT_V1_PARSE_ERROR` nhiều bước liên tiếp, cuối cùng dẫn đến `AGENT_V1_TIMEOUT`.

### Log Source

```json
// logs/2026-06-01.log
{"timestamp": "2026-06-01T09:57:52.574283", "event": "AGENT_V1_PARSE_ERROR", "data": {"step": 3}}
{"timestamp": "2026-06-01T09:58:06.889439", "event": "AGENT_V1_PARSE_ERROR", "data": {"step": 4}}
{"timestamp": "2026-06-01T09:58:06.889838", "event": "AGENT_V1_TIMEOUT",     "data": {"steps": 4}}
```

Step trước đó (step 2) xuất ra:

```
*   User Role: Academic research assistant.
    *   Constraint 1: Max 3-5 sentences for simple questions.
    ...
```

→ Model đang "echo" lại system prompt thay vì tuân theo format `Thought / Action / Final Answer`.

### Diagnosis

**Nguyên nhân gốc rễ là 3 yếu tố kết hợp:**

1. **Model quá nhỏ cho tiếng Việt + format nghiêm ngặt**: Phi-3-mini (3.8B, 4-bit) được pre-train chủ yếu bằng tiếng Anh. System prompt tiếng Việt có 6 RULE dài làm model "quên" format khi context tăng lên.

2. **Context window saturation**: Sau 2 vòng tool call, context đã chứa ~2,500 tokens (system prompt ~600 tokens + query + 2 observation). Phi-3-mini bắt đầu xuống cấp → lặp lại nội dung system prompt thay vì tiếp tục reasoning.

3. **Không có few-shot examples đủ mạnh**: System prompt của Agent V1 chỉ có 1 ví dụ đúng và 1 ví dụ sai. Không đủ để model nhỏ học pattern ổn định qua nhiều bước.

### Solution

Đã áp dụng 2 biện pháp:

1. **Truncate observation** — Giới hạn observation xuống 800 ký tự trước khi inject vào context để tránh đẩy context quá giới hạn:
   ```python
   obs_truncated = observation[:800] + "..." if len(observation) > 800 else observation
   conversation += f"\n{ai_response}\nObservation: {obs_truncated}\n"
   ```

2. **Chuyển sang cloud API** — Khi test với `gemma-4-26b-a4b-it` (Gemini), parse error giảm hẳn: agent hoàn thành trong 1–2 steps thay vì timeout tại step 4. (Log: `AGENT_V1_END steps=2 tokens=2655`)

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning — Thought block giúp Agent thế nào?

Block `Thought` ép Agent phải **suy nghĩ trước khi hành động**. Ví dụ Test Case 3 (`"Tìm bài báo về transformer trong NLP rồi so sánh 2 bài tốt nhất"`):

- **Chatbot**: Trả lời ngay → bịa thông số, tác giả/citations không có trong PAPER_DB
- **Agent Thought 1**: *"Cần tìm danh sách bài báo trước"* → gọi `search_papers("transformer NLP")`
- **Agent Thought 2**: *"Trong kết quả, P001 (98,000 citations) và P002 (74,000 citations) có chất lượng cao nhất"* → gọi `compare_papers(P001 P002)`
- **Agent Thought 3**: *"Đã có đủ data để tư vấn"* → Final Answer với số liệu cụ thể từ tool output

→ Thought block tạo ra **chuỗi suy luận rõ ràng**, giúp dễ debug và verify.

### 2. Reliability — Khi nào Agent tệ hơn Chatbot?

Agent tệ hơn khi:
- **Câu hỏi đơn giản** (`"Attention mechanism là gì?"`): Agent tốn gấp 2–3× tokens vì vẫn cố gọi tool không cần thiết
- **Latency cao hơn**: Agent V1 cần 2–4 vòng lặp, mỗi vòng ~7–15s với Phi-3-mini, tổng thời gian dài hơn nhiều so với Chatbot (~33–53s)
- **Chi phí cao hơn**: Agent V1 trung bình 3,825 tokens/query — gấp **~2.7×** so với Chatbot (1,295 tokens) = gấp 2.7× chi phí API

→ **Kết luận**: Không phải mọi bài toán đều cần Agent. Dùng Agentic Fit score để đánh giá trước: câu hỏi khái niệm/định nghĩa → Chatbot đủ tốt và rẻ hơn.

### 3. Observation — Feedback từ environment ảnh hưởng thế nào?

Observation (kết quả tool trả về) **thay đổi hướng suy luận** của Agent:
- Query `"Tìm transformer NLP"`: Observation từ `search_papers` chứa 3 bài báo kèm ID, citations → Agent tự chọn P001 (98,000 citations) làm bài chính → gọi `get_paper_details(P001)` để lấy thêm chi tiết
- Edge case `"Tìm bài báo tác giả XYZ"`: Observation trả `"Không tìm thấy"` + RULE 2 reminder → Agent V1 **không lặp vô ích**, thay vào đó trả Final Answer ngay: *"Tôi không tìm thấy thông tin này trong cơ sở dữ liệu"*

→ Observation là **feedback loop** giúp Agent **thích ứng** (adaptive) thay vì cứng nhắc.

---

## IV. Future Improvements (5 Points)

### Scalability
- **Async tool calls**: Gọi `search_papers` và `get_paper_details` song song bằng `asyncio` khi cần nhiều thông tin cùng lúc, giảm tổng latency
- **Caching**: Cache kết quả `search_papers` theo query hash — không gọi lại DB nếu cùng keyword trong cùng session

### Safety
- **Supervisor Agent**: Thêm 1 LLM nhẹ kiểm tra từng `Action` trước khi thực thi (audit layer) — phát hiện prompt injection qua tool argument, ngăn tool argument giả mạo Paper ID
- **Input sanitization**: Validate và sanitize query string trước khi truyền vào tools, đặc biệt kiểm tra regex `P\d+` format trước khi query DB

### Performance
- **Vector Database**: Khi PAPER_DB mở rộng lên hàng nghìn bài báo, thay string matching bằng **ChromaDB** + Sentence-Transformers (`all-MiniLM-L6-v2`) để tìm kiếm ngữ nghĩa
- **Multi-Agent System**: Tách thành các agent chuyên biệt (SearchAgent, DetailAgent, CompareAgent) phối hợp qua orchestrator để xử lý song song các bước độc lập
- **RAG Integration**: Kết hợp Retrieval-Augmented Generation để agent truy xuất nội dung full-text bài báo thực (PDF), không chỉ metadata mock


# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: DAY03_2A202600863_LeHuuKhoa
- **Team Members**:
  | Họ và Tên | MSSV | Module phụ trách |
  | :--- | :--- | :--- |
  | Lê Hữu Khoa | 2A202600863 | `agent.py`, `agent_v1.py`, `chatbot.py` |
  | Nguyễn Đức Thành | 2A202600955 | `src/tools/research_tools.py` |
  | Trần Tiến Đạt | 2A202600978 | `main.py` |
  | Hồ Trọng Nhật Minh | 2A202600768 | `web_demo.py` |
- **Deployment Date**: 2026-06-01

---

## 1. Executive Summary

Hệ thống **Academic Research Assistant** so sánh ba cách tiếp cận để hỗ trợ sinh viên và nhà nghiên cứu tìm kiếm bài báo khoa học:

1. **Chatbot baseline** (`chatbot.py`) — 1 lần gọi LLM, không có tool, dễ hallucinate thông tin bài báo cụ thể.
2. **ReAct Agent** (`agent.py`) — Vòng lặp Thought-Action-Observation, tối đa 5 bước.
3. **ReAct Agent V1** (`agent_v1.py`) — Như agent.py nhưng thêm 6 RULE chống hallucination nghiêm ngặt, tối đa 4 bước.

- **Success Rate**: 65% trên ~20 test case (từ log 2026-06-01)
- **Key Outcome**: Agent trả lời đúng 100% các truy vấn tìm kiếm bài báo cụ thể bằng cách gọi `search_papers`, trong khi Chatbot hallucinate thông tin bài báo (tác giả, năm, citations) trong mọi trường hợp DB-specific. Agent V1 phát hiện và chặn thêm 1 trường hợp trả lời trực tiếp không qua tool (RULE4_TRIGGERED).

### 1.1 Team Contributions (Phân chia công việc)

Dự án được phân chia thành 4 module chuyên biệt, tương ứng với 4 thành viên trong nhóm để tối ưu hóa hiệu suất và dễ dàng triển khai:

1. **Lê Hữu Khoa — 2A202600863 (Team Lead & Core Agent)**: Chịu trách nhiệm thiết kế thuật toán cốt lõi cho ReAct Agent. Viết System Prompt, vòng lặp suy luận Thought-Action-Observation, cơ chế phát hiện lỗi parse, RULE chống hallucination nghiêm ngặt (RULE 1–6) và logic phân biệt câu hỏi khái niệm vs. câu hỏi DB-specific (RULE 4 trigger). Đồng thời xây dựng Chatbot baseline với `LocalProvider`. *(Phụ trách: `src/agent/agent.py`, `src/agent/agent_v1.py`, `src/agent/chatbot.py`)*

2. **Nguyễn Đức Thành — 2A202600955 (Tooling & Data Engineer)**: Thiết kế 3 tools nghiên cứu học thuật (`search_papers`, `get_paper_details`, `compare_papers`) và xây dựng mock PAPER_DB gồm 10 bài báo khoa học thực tế (Transformer, BERT, GPT-3, ResNet, ViT, RAG, ...) với đầy đủ metadata: citations, quality score, research gaps, DOI. Viết logic keyword matching và filter theo năm/lĩnh vực. *(Phụ trách: `src/tools/research_tools.py`)*

3. **Trần Tiến Đạt — 2A202600978 (QA / Evaluation Engineer)**: Thiết lập môi trường thử nghiệm so sánh 3 approach trên 5 test case học thuật chuẩn hóa (Agent-Favorable, Chatbot-Favorable, Edge Case hallucination test). Viết script benchmark tự động đo Latency, Token usage, Steps cho cả Chatbot, Agent và Agent V1 rồi in kết quả dạng bảng ra terminal. *(Phụ trách: `main.py`)*

4. **Hồ Trọng Nhật Minh — 2A202600768 (Frontend & System Dev)**: Thiết kế và xây dựng giao diện Web Demo phong cách Neo Brutalism (Flask). Tích hợp cả 3 approach (Chatbot / Agent / Agent V1) vào một UI duy nhất với thread-safe shared LLM instance và mutex lock. Hiển thị Trace log (Thought-Action-Observation) trực tiếp trên trình duyệt để minh họa luồng suy nghĩ của AI. *(Phụ trách: `web_demo.py`)*

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

```
User Input
    │
    ▼
┌──────────────────┐
│  System Prompt    │ ← Identity + Tools + Format + Constraints (RULE 1–6 cho V1)
│  + User Query     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐       ┌───────────────────────┐
│  LLM Generate    │       │    AVAILABLE TOOLS     │
│  (Gemini/Local)  │       │                        │
│                  │       │ • search_papers         │
│  Output:         │       │ • get_paper_details     │
│  Thought + Action│       │ • compare_papers        │
└────────┬─────────┘       └───────────────────────┘
         │
    ┌────┴────┐
    │  Parse  │ ← Regex: r'Action:\s*(\w+)\(([^)]*)\)'
    │Response │
    └────┬────┘
         │
   ┌─────┴──────┐───────────┐
   │            │           │
   ▼            ▼           ▼
Final Answer  Action    Parse Error
   │          found?        │
   │            │           │ (V1: inject format correction)
   │       Execute Tool  Send format
   │            │       reminder
   │       Observation      │
   │            │           │
   │         Append ◄───────┘
   │       to Prompt
   │            │
   │            └──→ Back to LLM (max 4–5 loops)
   ▼
Return Answer to User
```

### 2.2 Tool Definitions (Inventory)

| Tool Name | Input Format | Use Case |
| :--- | :--- | :--- |
| `search_papers` | `string` (từ khóa / lĩnh vực / năm) | Tìm bài báo trong PAPER_DB theo keyword, field, hoặc khoảng năm. Trả về tối đa 5 kết quả kèm ID, citations, quality score. |
| `get_paper_details` | `string` (Paper ID, ví dụ `P001`) | Lấy thông tin đầy đủ: tóm tắt, DOI, research gaps, từ khóa. |
| `compare_papers` | `string` (2 Paper ID cách nhau bởi dấu cách, ví dụ `P001 P002`) | So sánh 2 bài báo theo quality score, citations, và research gaps. |

### 2.3 LLM Providers Used

- **Primary**: `gemma-4-26b-a4b-it` (Google Gemini API — `GeminiProvider`)
- **Secondary**: `deepseek-v4-flash` (OpenAI-compatible API — `OpenAIProvider`)
- **Local (Offline)**: `Phi-3-mini-4k-instruct-q4.gguf` (llama-cpp-python — `LocalProvider`)

---

## 3. Agent Basic → Agent V1 Evolution

### 3.1 Các lỗi của `agent.py` (Basic)

| Lỗi | Mô tả | Hậu quả |
| :--- | :--- | :--- |
| **System prompt đơn giản** | Không có constraints cứng, không RULE chống hallucination | Agent có thể bịa thông tin bài báo mà không gọi tool |
| **Không xử lý parse error** | Khi LLM trả sai format, chỉ log rồi tiếp tục vòng lặp | Agent lặp lại cùng action gây infinite loop với Phi-3-mini |
| **Không kiểm soát Final Answer** | Không phân biệt câu hỏi cần tool và không cần tool | Trả lời trực tiếp câu hỏi bài báo cụ thể mà không search |
| **Không có deduplication** | Không phát hiện khi gọi cùng tool với cùng argument nhiều lần | Lãng phí token và vẫn timeout |

### 3.2 Cải tiến trong `agent_v1.py` (Strict Anti-Hallucination)

| Vấn đề Basic | Fix trong V1 | Code |
| :--- | :--- | :--- |
| Prompt không có ràng buộc | Thêm 6 RULE cứng: cấm bịa, bắt buộc tool, cấm tự đặt Paper ID... | `get_system_prompt()` với RULE 1–6 |
| Parse error → loop | Inject format reminder: `"[Parse error] Tuân theo format: Action: tool_name(arguments)"` | `AGENT_V1_PARSE_ERROR` + correction message |
| Trả lời thẳng không qua tool | RULE 4 check: nếu `Final Answer` xuất hiện mà chưa gọi tool → chặn + buộc gọi `search_papers` | `AGENT_V1_RULE4_TRIGGERED` |
| Tool trả "Không tìm thấy" → loop | RULE 2: append reminder "KHÔNG được đoán, phải thừa nhận không biết" | `[V1 RULE 2]` trong observation |
| Observation quá dài → context overflow | Truncate observation xuống 800 ký tự | `obs_truncated = observation[:800] + "..."` |

### 3.3 So sánh System Prompt Basic vs V1

**Basic (`agent.py`) — đơn giản, thiếu constraints:**

```
Bạn là trợ lý nghiên cứu học thuật thông minh. Bạn có các công cụ sau:
- search_papers: ...

Luôn tuân theo định dạng sau:
Thought: [phân tích yêu cầu]
Action: tool_name(arguments)
Observation: [kết quả từ tool]
Final Answer: [câu trả lời đầy đủ]
```

**V1 (`agent_v1.py`) — đầy đủ RULE, có few-shot:**

```
Bạn là trợ lý nghiên cứu học thuật NGHIÊM NGẶT. Bạn có các công cụ:
...
LUẬT BẮT BUỘC:
[RULE 1] CHỈ trích dẫn thông tin bài báo TỪ KẾT QUẢ TOOL. TUYỆT ĐỐI KHÔNG bịa đặt.
[RULE 2] Nếu tool trả "Không tìm thấy" → Final Answer PHẢI nói: "Tôi không tìm thấy..."
[RULE 3] KHÔNG tự đặt Paper ID (P001...) nếu chưa nhận được từ tool.
[RULE 4] PHẢI gọi ít nhất 1 tool trước Final Answer nếu câu hỏi về bài báo cụ thể.
[RULE 5] Câu hỏi định nghĩa/khái niệm tổng quát → có thể trả lời trực tiếp.
[RULE 6] Tối đa 4 bước. Quá giới hạn → khai báo rõ ràng.

VÍ DỤ ĐÚNG: ...
VÍ DỤ SAI (HALLUCINATION): ...
```

**Kết quả cải tiến:**
- V1 chặn được 100% trường hợp trả lời trực tiếp về bài báo cụ thể mà không qua tool (`RULE4_TRIGGERED`)
- V1 xử lý edge case "Không tìm thấy" gracefully (RULE 2) thay vì loop lại
- V1 có trace đầy đủ và log event cho mọi bước để debug

---

## 4. Tool Design Evolution

### Tool Description Basic (quá ngắn, thiếu format):

```
- search_papers: Tìm bài báo theo từ khóa
- get_paper_details: Lấy thông tin bài báo
- compare_papers: So sánh 2 bài báo
```

→ **Vấn đề**: LLM không biết input format → gọi `search_papers({"keywords": "transformer"})` hoặc `search_papers(query='NLP', start_year='2023')` — argument sai format → Observation lỗi → loop.

### Tool Description V1 (rõ ràng, chi tiết):

```
- search_papers: Tìm bài báo theo từ khóa, lĩnh vực hoặc năm.
    Input: chuỗi từ khóa, ví dụ 'transformer NLP' hoặc 'education 2022-2023'.
    Output: danh sách bài báo với ID, tiêu đề, tác giả, năm, citations, chất lượng.

- get_paper_details: Lấy thông tin đầy đủ của một bài báo theo ID.
    Input: Paper ID hợp lệ, ví dụ 'P001'.
    Output: tóm tắt, DOI, research gaps, từ khóa đầy đủ.

- compare_papers: So sánh 2 bài báo theo research gap và chất lượng.
    Input: 2 Paper ID cách nhau bởi dấu cách, ví dụ 'P001 P002'.
    Output: bảng so sánh chi tiết.
```

→ **Kết quả**: Agent V1 gọi đúng tool, đúng format trong các run thành công với Gemini (log: `AGENT_V1_TOOL_CALL step=1 tool=search_papers args=transformer NLP`).

---

## 5. Telemetry & Performance Dashboard

*Phân tích metrics thu thập từ file log `2026-06-01.log`.*

| Metric | Chatbot | Agent V1 (Strict) |
| :--- | :--- | :--- |
| **Avg Tokens / Task** | ~1,295 tokens | ~3,825 tokens |
| **Avg Latency (P50)** | ~37,894 ms | ~15,000 ms/step |
| **Max Latency (P99)** | 53,202 ms | timeout ~60s |
| **Avg Steps** | 1 (direct) | ~2.2 steps |
| **Cost Estimate** | ~$0.13 | ~$0.34 |

**Latency samples — Chatbot (ms):** 33,039 · 33,473 · 35,105 · 37,064 · 38,724 · 41,765 · 46,793 · 53,202

**Token samples — Agent V1 (tokens):** 2,655 · 3,000 · 3,497 · 3,596 · 3,612 · 3,614 · 3,621 · 3,678 · 7,151

---

## 6. Root Cause Analysis (RCA) - Failure Traces

### Case 1: Parse Error → Vòng lặp vô hạn (`AGENT_V1_TIMEOUT`)

- **Input**: `"Tìm 3 bài báo về NLP"` (model: Phi-3-mini)
- **V1 Behavior**: Agent gọi `search_papers(NLP', '2023)` lặp lại 4 lần với argument **giống hệt nhau** (kèm lỗi dấu nháy đơn thừa). Regex không parse được → không gọi được tool → loop đến max_steps.
- **V1 Fix**: Inject parse correction reminder sau mỗi `AGENT_V1_PARSE_ERROR` → Gemini tự sửa format ngay bước tiếp. Phi-3-mini vẫn thất bại do model quá nhỏ.

### Case 2: Edge Case — Tool không có data (`RULE 2` graceful degradation)

- **Input**: `"Tìm bài báo của tác giả Nguyễn Văn XYZ về quantum computing"`
- **Basic Agent**: `search_papers` trả "Không tìm thấy" → Agent không biết xử lý → tiếp tục gọi lại tool → loop → timeout.
- **V1 Fix**: Nhận Observation "Không tìm thấy" + RULE 2 reminder → Thought: "Không có data trong DB" → Final Answer ngay: *"Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."* Graceful degradation hoàn toàn.

---

## 7. Ablation Studies & Experiments

### Chatbot vs Agent trên 5 test cases

| # | Case | Chatbot | Agent V1 | Winner |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Kiến thức chung (`"Attention là gì?"`) |  Đúng (~1,144 tokens) |  Đúng nhưng dư tokens | **Chatbot** |
| 2 | Tìm bài báo transformer NLP |  Hallucinated (bịa citations/năm) |  `search_papers` → P001, P002 chính xác | **Agent V1** |
| 3 | So sánh BERT vs GPT-3 |  Fabricated (bịa research gaps) |  `compare_papers(P002 P003)` → data thật | **Agent V1** |
| 4 | Tìm bài báo vanishing gradient |  Partial (không có DB) |  `search_papers("gradient descent")` → P005 | **Agent V1** |
| 5 | Tác giả không tồn tại (edge case) |  Bịa bài báo |  Thừa nhận "Không tìm thấy" (RULE 2) | **Agent V1** |

---

## 8. Production Readiness Review

- **Security**: Tất cả tool argument được validate bằng regex trước khi query DB (`re.search(r'P\d+', query.upper())` trong `get_paper_details`; keyword cleaning trong `search_papers`). Ngăn chặn injection qua chuỗi tool argument tùy ý.
- **Guardrails**: `max_steps=5` (agent) và `max_steps=4` (agent_v1) ngăn vòng lặp vô hạn và chi phí không kiểm soát. RULE 4 trong agent_v1 bắt buộc tool call trước Final Answer.
- **Scaling**:
  - Kết nối API thật (Semantic Scholar, arXiv API) thay mock PAPER_DB
  - RAG + Vector Database (ChromaDB) cho catalog hàng triệu bài báo
  - Multi-agent system: SearchAgent + AnalysisAgent + CompareAgent phối hợp qua orchestrator
  - Supervisor Agent để audit quyết định và phát hiện hallucination

---

> [!NOTE]
> Submit this report by renaming it to `GROUP_REPORT_[TEAM_NAME].md` and placing it in this folder.

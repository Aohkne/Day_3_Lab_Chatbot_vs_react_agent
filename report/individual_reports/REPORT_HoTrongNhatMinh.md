# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Hồ Trọng Nhật Minh
- **Student ID**: 2A202600768
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

Tôi phụ trách xây dựng **Web Demo** (`web_demo.py`) — giao diện Flask phong cách Neo Brutalism cho phép so sánh trực quan cả 3 cách tiếp cận (Chatbot / ReAct Agent / Agent V1) trên cùng một câu hỏi, hiển thị song song 3 cột kết quả kèm trace suy luận.

- **Modules Implemented**: `web_demo.py`
- **Code Highlights**:
  - **3 endpoint riêng biệt, gọi song song từ frontend**: `/ask/chatbot`, `/ask/agent`, `/ask/agentv1` — mỗi endpoint khởi tạo đúng approach tương ứng (`run_chatbot`, `ReActAgent`, `ReActAgentV1`) và trả JSON gồm `answer`, `tokens`, `latency_ms`, `steps`, `trace`.
  - **Shared LLM instance + Lock vì llama-cpp-python không thread-safe**:
    ```python
    _llm_instance: LocalProvider = None
    _llm_lock = threading.Lock()

    def get_llm() -> LocalProvider:
        global _llm_instance
        if _llm_instance is None:
            _llm_instance = LocalProvider(model_path=model_path, n_ctx=4096)
        return _llm_instance
    ```
    Model Phi-3 chỉ được load 1 lần (singleton), mọi lệnh gọi `generate()`/`agent.run()` đều bọc trong `with _llm_lock:` để tránh 2 request đồng thời cùng ghi vào 1 context của `Llama` object.
  - **Custom System Prompt**: cho phép người dùng ghi đè hoàn toàn system prompt của Chatbot, hoặc nối thêm `extra_instructions` vào cuối prompt gốc của Agent/Agent V1 (giữ nguyên định dạng `Thought/Action/Observation/Final Answer` bắt buộc để không phá vỡ parser).
  - **`_build_trace()`**: chuyển raw trace (`list[dict]` chứa `llm_output` thô) thành JSON gọn — tách riêng `thought`, `action`, `observation`, `final_answer`, `constraint` — để render từng bước suy luận trực tiếp trên trình duyệt (nút "Xem quá trình suy luận").

---

## II. Debugging Case Study (10 Points)

### Vấn đề: Race condition khi nhiều request cùng lúc gọi vào 1 instance `Llama`

- **Problem Description**: Khi mở 2 tab trình duyệt và gửi câu hỏi gần như đồng thời, ứng dụng chạy với `threaded=True` của Flask xử lý 2 request song song trên 2 thread khác nhau, nhưng cả hai cùng gọi `generate()` trên **cùng một object `Llama`** (được load 1 lần cho tiết kiệm RAM/thời gian init).
- **Diagnosis**: `llama-cpp-python` không đảm bảo thread-safety cho một instance `Llama` dùng chung — hai lệnh `self.llm(...)` gọi đồng thời có thể ghi đè context nội bộ (KV-cache) của nhau, dẫn đến câu trả lời bị trộn lẫn giữa 2 câu hỏi khác nhau, hoặc trong trường hợp xấu hơn crash tiến trình.
- **Solution**: Thêm `_llm_lock = threading.Lock()` ở scope module, bọc **mọi** lời gọi `llm.generate()` / `agent.run()` (cả 3 endpoint) trong `with _llm_lock:`. Việc này serializes tất cả các lượt inference — chỉ 1 request được chạy model tại một thời điểm, các request khác phải đợi — đổi lấy độ chính xác và ổn định thay vì throughput đồng thời cao. Đây là đánh đổi hợp lý cho một demo cục bộ (single-user hoặc vài người dùng cùng lúc trong buổi demo).

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Việc hiển thị 3 panel cạnh nhau trên cùng 1 UI làm rõ sự khác biệt về cách "suy nghĩ": panel Chatbot trả lời ngay lập tức không có bước trung gian nào để xem, trong khi panel Agent/Agent V1 có nút trace hiển thị từng `Thought → Action → Observation` — người xem thấy được agent quyết định gọi `search_papers` trước khi trả lời câu hỏi về một bài báo cụ thể, thay vì đoán mò.
2. **Reliability**: Đặt 3 câu trả lời song song giúp phát hiện hallucination trực quan hơn nhiều so với đọc log — ví dụ với câu hỏi edge case ("tác giả không tồn tại"), panel Chatbot vẫn tự tin đưa ra một câu trả lời nghe hợp lý, trong khi panel Agent V1 hiển thị rõ ràng `Observation: Không tìm thấy` rồi kết luận trung thực "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."
3. **Observation**: Vì trace được render trực tiếp trên UI (không chỉ nằm trong file log), người dùng không rành kỹ thuật (ví dụ giảng viên chấm demo) cũng có thể theo dõi được `Observation` từ tool đã thay đổi hướng suy luận của Agent như thế nào ở mỗi bước, thay vì phải đọc JSON log thô.

---

## IV. Future Improvements (5 Points)

- **Scalability**: Thay vì 1 instance `Llama` dùng chung + lock (serialize toàn bộ request), có thể dùng một pool nhỏ các instance model (process pool) để phục vụ nhiều người dùng đồng thời mà không phải xếp hàng chờ từng lượt.
- **Safety**: Thêm rate limiting và validate độ dài input ở endpoint Flask để tránh một client gửi query cực dài làm tràn context window hoặc gây từ chối dịch vụ cho các client khác đang xếp hàng chờ `_llm_lock`.
- **Performance**: Chuyển từ response chặn (blocking) sang streaming (SSE/WebSocket) để hiển thị câu trả lời dần dần thay vì đợi toàn bộ vòng lặp ReAct hoàn tất mới trả về, cải thiện trải nghiệm chờ đợi trên UI.

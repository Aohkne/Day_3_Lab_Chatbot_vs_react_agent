# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Hồ Trọng Nhật Minh
- **Student ID**: 2A202600768
- **Date**: 01/06/2026

---

## I. Technical Contribution (15 Points)

Trong bài tập Lab 3 này, tôi chịu trách nhiệm chính trong việc cấu trúc lại toàn bộ hệ thống lõi Agent nhằm chuyển đổi từ kịch bản mẫu sang chủ đề **Retail Tinh Gọn (Trợ lý Kiểm tra Kho & Tính giá Đơn hàng Đa kênh)**, đồng thời hiện thực hóa các khối logic rẽ nhánh tự động.

- **Modules Implementated**: `src/agent/agent.py`, và phát triển công cụ thực nghiệm `chatbot_baseline.py`, `run_retail_demo.py`.
- **Code Highlights**: 
    * **Cơ chế rẽ nhánh thông minh theo dữ liệu công cụ đầu vào:**
        ```python
        if not self.tools:
            # CHẾ ĐỘ 1: CHATBOT THƯỜNG - Duy trì lịch sử hội thoại liên tục
            self.history.append({"role": "user", "content": user_input})
            ...
            return bot_response
        ```
    * **Tầng phòng vệ tham số (Fault-Tolerant Argument Parser):** Tôi đã lập trình bộ bóc tách chuỗi linh hoạt tại hàm `_execute_tool` để làm sạch dấu ngoặc/nháy và tự động xử lý/eval các biểu thức toán học dạng chuỗi do mô hình nhỏ (SLM) sinh ra, tránh làm sập vòng lặp.
- **Documentation**: Khi danh sách công cụ trống, hệ thống tự động đưa ngữ cảnh hội thoại vào mảng `self.history` để `Phi-3` nhớ tên và ngành học của sinh viên ở lượt chat sau. Khi có công cụ, hệ thống sẽ chèn chuỗi chỉ thị ép cấu trúc `Thought -> Action -> Observation` để kích hoạt vòng lặp tuần hoàn cho đến khi nhận diện được từ khóa `Final Answer:`.

---

## II. Debugging Case Study (10 Points)

Tôi đã phân tích và giải quyết thành công hai sự cố kỹ thuật nghiêm trọng trong quá trình phát triển thông qua hệ thống telemetry log:

### 1. Sự cố sập chỉ thị phần cứng trên Windows (Lỗi 0xc000001d)
- **Problem Description**: Tiến trình chạy thử nghiệm cục bộ bị ngắt lập tức với lỗi `Windows Error 0xc000001d (STATUS_ILLEGAL_INSTRUCTION)`.
- **Diagnosis**: Bộ phân phối binary mặc định của thư viện `llama-cpp-python` cố tình sử dụng tập lệnh toán học ma trận nâng cao (AVX2/AVX512) vượt quá tập lệnh mà CPU hiện tại của máy host hỗ trợ hoặc bị Windows ngăn chặn.
- **Solution**: Chạy lệnh xóa sạch bộ nhớ đệm ẩn (`pip cache purge`) và hạ cấp chỉ định rõ gói dựng sẵn tương thích cao ổn định `llama-cpp-python==0.2.90`.

### 2. Sự cố lặp bẫy tham số của mô hình ngôn ngữ nhỏ (SLM Parameter Loop)
- **Problem Description**: Ở phiên bản thử nghiệm đầu tiên, Agent bị kẹt và báo lỗi `calculate_shipping() missing 1 required positional argument` ở Step 3 và Step 4, dẫn đến việc cạn kiệt lượt chạy (`max_steps = 5`) và trả về kết quả thất bại.
- **Log Trace**:
  ```json
  [Step 3] Action: calculate_shipping('Hà Nội', (2 * 0 bonet's weight))
  Observation: Error executing tool calculate_shipping...
  [Step 4] Action: calculate_shipping('Hà Nội', (2 * 0.25))
  Observation: Error executing tool calculate_shipping...
  ```
- **Diagnosis**: Mô hình Phi-3 gặp giới hạn về số lượng tham số khi tự tính toán khối lượng vật lý. Thay vì truyền một chuỗi thô, nó cố tình đóng mở ngoặc đơn toán học khiến bộ lọc Regex hiểu nhầm toàn bộ cụm phía sau là một đối số duy nhất.
- **Solution**: Tôi đã tối ưu hóa lại Tool Description trong get_item_price để định nghĩa rõ ràng cấu trúc dữ liệu đầu ra và nâng cấp hàm _execute_tool để tự động làm sạch ký tự ngoặc () và eval biểu thức số học hộ LLM. Đồng thời nâng mức trần ranh giới an toàn max_steps lên mức 8. Kết quả ở lượt chạy sau, Agent đã tự sửa sai thành công tại Step 5 và kết thúc hoàn hảo.

<!-- - **Problem Description**: [e.g., Agent caught in an infinite loop with `Action: search(None)`]
- **Log Source**: [Link or snippet from `logs/YYYY-MM-DD.log`]
- **Diagnosis**: [Why did the LLM do this? Was it the prompt, the model, or the tool spec?]
- **Solution**: [How did you fix it? (e.g., updated `Thought` examples in the system prompt)] -->

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1.  **Reasoning**: Khối suy luận Thought đóng vai trò như một không gian nháp (Scratchpad) kích hoạt cơ chế tự kiểm tra của mô hình (Chain-of-Thought). Thay vì lao ngay vào việc đoán từ ngẫu nhiên, Thought giúp Phi-3 định hình rõ ràng lộ trình logic: "Cần kiểm kho trước $\rightarrow$ Lấy đơn giá để nhân số lượng $\rightarrow$ Lấy cân nặng để tính ship". Điều này giúp một mô hình nhỏ 3.8B đạt được tỷ lệ xử lý chính xác tương đương các mô hình lớn.
2.  **Reliability**: Chatbot Baseline không dùng công cụ bộc lộ điểm yếu chết người là Ảo tưởng dữ liệu (Hallucination). Khi nhận câu hỏi tính giá đơn hàng, Chatbot Baseline lập tức tự bịa ra giá tiền áo thun và khẳng định chắc chắn shop còn hàng kèm chính sách freeship Hà Nội. Chatbot thường tệ hơn Agent khi đối mặt với dữ liệu động (kho bãi) và các tác vụ cần tính toán bắc cầu.
3.  **Observation**: Phản hồi từ môi trường (Observation) đóng vai trò là chiếc mỏ neo giữ mô hình ở lại với thực tế. Khi nhận được dữ liệu thực tế từ hàm Python trả về ở Step 1 (In-stock quantity: 5), LLM lập tức cập nhật trạng thái niềm tin (Belief State) để chuyển sang bước lấy giá ở Step 2 một cách tự tin mà không cần phỏng đoán.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**: Thay thế cơ chế gọi hàm đồng bộ bằng kiến trúc hàng đợi không đồng bộ (Asynchronous Task Queue sử dụng Celery/Redis) cho các hàm gọi tool. Việc này giúp hệ thống không bị nghẽn (Block) khi có hàng ngàn khách hàng cùng check kho một lúc.
- **Safety**: Áp dụng cơ chế Structured Outputs (ép định dạng JSON Schema thông qua thư viện Pydantic kết hợp với hàm sinh của LLM Provider) thay thế hoàn toàn cho bộ lọc Regex thô hiện tại. Điều này triệt tiêu hoàn toàn lỗi vỡ định dạng chuỗi của LLM.
- **Performance**: Tích hợp tầng cơ sở dữ liệu vector (Vector DB) để làm kho lưu trữ và trích xuất công cụ (Tool Retrieval). Khi hệ thống Retail mở rộng lên hàng trăm công cụ khác nhau (Mã giảm giá, hoàn tiền, đối tác vận chuyển), Agent chỉ cần bốc các công cụ thực sự liên quan vào ngữ cảnh prompt để tiết kiệm chi phí Token và tối ưu hóa độ trễ Latency.

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_[YOUR_NAME].md` and placing it in this folder.

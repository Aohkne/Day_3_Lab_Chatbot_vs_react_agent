# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Trần Tiến Đạt
- **Student ID**: 2A202600978
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implemented**: `main.py`
- **Code Highlights**:
  - Thiết kế và triển khai kịch bản đánh giá so sánh ba phương pháp: Chatbot baseline, ReAct Agent và ReAct Agent V1.
  - Viết logic benchmark tự động để đo `Latency`, `Token usage`, `Steps` cho từng approach.
  - Xây dựng báo cáo kết quả dạng bảng trong terminal nhằm đối chiếu hiệu suất và độ chính xác giữa các mô hình.
- **Documentation**:
  - Giải thích cách tích hợp `main.py` với các module agent và provider.
  - Mô tả quy trình chạy thử nghiệm trong README và tạo ra metric đầu ra hỗ trợ đánh giá khả năng suy luận của ReAct loop.

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**: Khi chạy benchmark, `main.py` bị gọi hai lần bởi một lệnh `main()` thừa ở cuối file, khiến cùng một tập test chạy lại và làm kết quả terminal bị lặp đôi.
- **Log Source**: Output terminal từ lệnh `python main.py` khi file được chạy.
- **Diagnosis**: Lỗi thực sự là dòng `main()` nằm bên ngoài khối `if __name__ == '__main__':`; đó khiến chương trình chạy một lần trong điều kiện entrypoint và một lần nữa ngay lập tức sau đó.
- **Solution**: Xóa dòng `main()` thừa ở cuối file để chương trình chỉ chạy đúng một lần và báo cáo benchmark không bị lặp lại.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1. **Reasoning**: `Thought` block giúp agent phân tách rõ ràng giữa bước suy luận và quyết định gọi tool, thay vì yêu cầu trực tiếp. Điều này làm cho hệ thống có thể thực hiện nhiều bước kiểm chứng và giảm khả năng trả lời bịa thông tin.
2. **Reliability**: Agent có thể kém hơn Chatbot khi câu hỏi là khái niệm chung, không cần tra cứu dữ liệu cụ thể. Trong các trường hợp như vậy, Chatbot direct answer thường nhanh hơn và đủ chính xác, còn ReAct loop có thể lãng phí bước và token.
3. **Observation**: Feedback từ `Observation` giúp agent xác định xem tool đã trả dữ liệu đúng hay chưa, từ đó quyết định tiếp tục gọi tool khác hoặc hoàn thiện câu trả lời. Đây là yếu tố then chốt để tránh vòng lặp vô hạn và điều chỉnh hành vi khi dữ liệu không tìm thấy.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**: Sử dụng hàng đợi bất đồng bộ (`async queue`) để chạy benchmark và gọi tool song song cho nhiều yêu cầu.
- **Safety**: Triển khai bộ giám sát audit log tự động cho mỗi bước Action/Observation, thêm bước kiểm tra `output validation` trước khi trả về người dùng.
- **Performance**: Kết hợp bộ nhớ đệm và profiler cho `main.py` để phân tích và giảm thiểu chi phí token, đồng thời ưu tiên các tác vụ ít bước khi thích hợp.

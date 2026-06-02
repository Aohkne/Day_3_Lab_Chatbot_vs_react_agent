"""
Research Tools — 3 công cụ tìm kiếm và phân tích bài báo khoa học.
"""
import re

# Mock Paper Database (10 bài báo)
PAPER_DB = {
    "P001": {
        "id": "P001",
        "title": "Attention Is All You Need",
        "authors": "Vaswani, Shazeer, Parmar và cộng sự",
        "year": 2017, "venue": "NeurIPS 2017", "field": "NLP",
        "keywords": ["transformer", "attention", "NLP", "self-attention", "sequence-to-sequence"],
        "citations": 98000, "quality_score": 9.8,
        "doi": "10.48550/arXiv.1706.03762",
        "abstract": "Đề xuất kiến trúc Transformer dựa hoàn toàn vào cơ chế attention, không dùng RNN/CNN. Đạt SOTA trên các tác vụ dịch máy với khả năng song song hoá cao.",
        "research_gaps": [
            "Độ phức tạp bậc hai theo độ dài chuỗi chưa được giải quyết",
            "Chưa khảo sát trong ngữ cảnh ít dữ liệu (low-resource)",
        ],
    },
    "P002": {
        "id": "P002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": "Devlin, Chang, Lee, Toutanova",
        "year": 2019, "venue": "NAACL 2019", "field": "NLP",
        "keywords": ["BERT", "pre-training", "NLP", "bidirectional", "transfer learning", "MLM"],
        "citations": 74000, "quality_score": 9.6,
        "doi": "10.18653/v1/N19-1423",
        "abstract": "BERT dùng Masked Language Modeling để pre-train biểu diễn ngôn ngữ hai chiều, sau đó fine-tune cho nhiều tác vụ NLP downstream.",
        "research_gaps": [
            "Chi phí tính toán cao khi fine-tune trên phần cứng hạn chế",
            "Giới hạn 512 token, khó xử lý tài liệu dài",
        ],
    },
    "P003": {
        "id": "P003",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "authors": "Brown, Mann, Ryder và cộng sự",
        "year": 2020, "venue": "NeurIPS 2020", "field": "NLP",
        "keywords": ["GPT-3", "few-shot", "language model", "in-context learning", "scaling"],
        "citations": 52000, "quality_score": 9.5,
        "doi": "10.48550/arXiv.2005.14165",
        "abstract": "GPT-3 với 175 tỷ tham số cho thấy khả năng few-shot learning mạnh mẽ mà không cần fine-tune, chỉ cần vài ví dụ trong prompt.",
        "research_gaps": [
            "Quá trình suy luận thiếu minh bạch, khó diễn giải",
            "Dễ hallucinate trong các tác vụ yêu cầu thông tin thực tế",
        ],
    },
    "P004": {
        "id": "P004",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT)",
        "authors": "Dosovitskiy, Beyer, Kolesnikov và cộng sự",
        "year": 2021, "venue": "ICLR 2021", "field": "Computer Vision",
        "keywords": ["ViT", "vision transformer", "image classification", "patch embedding", "computer vision"],
        "citations": 38000, "quality_score": 9.3,
        "doi": "10.48550/arXiv.2010.11929",
        "abstract": "Áp dụng Transformer thuần cho phân loại ảnh bằng cách chia ảnh thành các patch 16x16, đạt kết quả tốt hơn CNN khi có đủ dữ liệu pre-train.",
        "research_gaps": [
            "Cần lượng dữ liệu pre-train rất lớn để vượt CNN",
            "Tiêu tốn bộ nhớ cao với ảnh độ phân giải lớn",
        ],
    },
    "P005": {
        "id": "P005",
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "authors": "He, Zhang, Ren, Sun",
        "year": 2016, "venue": "CVPR 2016", "field": "Computer Vision",
        "keywords": ["ResNet", "residual learning", "deep learning", "skip connections", "image recognition"],
        "citations": 148000, "quality_score": 9.7,
        "doi": "10.1109/CVPR.2016.90",
        "abstract": "Đề xuất residual block với skip connection giúp huấn luyện mạng rất sâu (152 lớp) ổn định, giảm vấn đề vanishing gradient.",
        "research_gaps": [
            "Skip connection chưa tối ưu cho thiết bị bộ nhớ hạn chế",
            "Chưa phân tích kỹ trong bài toán học liên tục (continual learning)",
        ],
    },
    "P008": {
        "id": "P008",
        "title": "Reinforcement Learning for Educational Adaptive Systems: A Survey",
        "authors": "Nguyễn Văn An, Trần Thị Bích, Lê Quốc Cường",
        "year": 2022, "venue": "IEEE Transactions on Learning Technologies", "field": "Education Technology",
        "keywords": ["reinforcement learning", "education", "adaptive learning", "intelligent tutoring", "survey"],
        "citations": 420, "quality_score": 7.8,
        "doi": "10.1109/TLT.2022.3180267",
        "abstract": "Khảo sát 87 bài báo về ứng dụng RL trong hệ thống học thích nghi từ 2015–2021, phân loại theo loại môi trường học.",
        "research_gaps": [
            "Hầu hết thử nghiệm trong môi trường lab, chưa triển khai lớp học thực",
            "Thiếu đánh giá dài hạn (hơn 1 học kỳ)",
        ],
    },
    "P009": {
        "id": "P009",
        "title": "Large Language Models in Education: Opportunities and Challenges",
        "authors": "Wei Zhang, Sarah Johnson, Mohammed Al-Rashid",
        "year": 2023, "venue": "Computers & Education", "field": "Education Technology",
        "keywords": ["LLM", "education", "ChatGPT", "personalized learning", "hallucination", "academic integrity"],
        "citations": 1850, "quality_score": 8.2,
        "doi": "10.1016/j.compedu.2023.104787",
        "abstract": "Phân tích cơ hội và thách thức khi tích hợp LLM như GPT-4 vào giáo dục K-12 và đại học qua 12 tình huống nghiên cứu.",
        "research_gaps": [
            "Chưa có khung đánh giá chuẩn cho nội dung do LLM tạo ra",
            "Ít nghiên cứu về hiệu quả LLM với học sinh không nói tiếng Anh",
        ],
    },
    "P011": {
        "id": "P011",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "authors": "Lewis, Perez, Piktus, Petroni và cộng sự",
        "year": 2020, "venue": "NeurIPS 2020", "field": "NLP",
        "keywords": ["RAG", "retrieval augmentation", "open-domain QA", "dense retrieval", "knowledge"],
        "citations": 8900, "quality_score": 9.1,
        "doi": "10.48550/arXiv.2005.11401",
        "abstract": "Kết hợp mô hình seq2seq với cơ chế truy xuất dày đặc (DPR) để trả lời câu hỏi miền mở, đạt SOTA trên NQ và TriviaQA.",
        "research_gaps": [
            "Chất lượng phụ thuộc nhiều vào corpus và cách đánh chỉ số",
            "Độ trễ cao trong hệ thống hỏi đáp thời gian thực",
        ],
    },
    "P014": {
        "id": "P014",
        "title": "Deep Knowledge Tracing (DKT)",
        "authors": "Piech, Bassen, Huang, Ganguli",
        "year": 2015, "venue": "NeurIPS 2015", "field": "Education Technology",
        "keywords": ["deep knowledge tracing", "DKT", "LSTM", "education", "student performance", "knowledge tracing"],
        "citations": 2800, "quality_score": 8.6,
        "doi": "10.48550/arXiv.1506.05908",
        "abstract": "Dùng LSTM để dự đoán hiệu suất học sinh theo chuỗi thời gian, vượt trội so với Bayesian Knowledge Tracing truyền thống trên dataset ASSISTment.",
        "research_gaps": [
            "LSTM không nắm bắt mối quan hệ giữa các khái niệm kiến thức khác nhau",
            "Khả năng diễn giải hạn chế, khó dùng thực tế cho giáo viên",
        ],
    },
    "P015": {
        "id": "P015",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Touvron, Lavril, Izacard và cộng sự",
        "year": 2023, "venue": "arXiv 2023", "field": "NLP",
        "keywords": ["LLaMA", "open-source LLM", "foundation model", "efficient training", "language model"],
        "citations": 12400, "quality_score": 9.0,
        "doi": "10.48550/arXiv.2302.13971",
        "abstract": "Bộ mô hình nền 7B–65B tham số huấn luyện hoàn toàn trên dữ liệu công khai, mô hình nhỏ hơn nhưng hiệu quả hơn nhờ dùng nhiều dữ liệu hơn.",
        "research_gaps": [
            "Không có instruction fine-tuning trong base model",
            "Chưa đánh giá chính thức về độ an toàn và RLHF alignment",
        ],
    },
}


# Tool Functions

def search_papers(query: str) -> str:
    """Tìm bài báo theo từ khóa, lĩnh vực hoặc năm."""
    query_lower = query.lower()

    # Trích xuất khoảng năm
    year_from, year_to = None, None
    m = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', query)
    if m:
        year_from, year_to = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r'\b(20\d{2}|19\d{2})\b', query)
        if m:
            year_from = year_to = int(m.group(1))

    clean_q = re.sub(r'\b\d{4}\b', '', query_lower).strip()
    keywords = [kw for kw in clean_q.split() if len(kw) > 2]

    matched = []
    for p in PAPER_DB.values():
        if year_from and not (year_from <= p["year"] <= (year_to or year_from)):
            continue
        text = (p["title"] + " " + " ".join(p["keywords"]) + " " + p["field"] + " " + p["authors"]).lower()
        if any(kw in text for kw in keywords):
            matched.append(p)

    if not matched:
        return (
            f"Không tìm thấy bài báo nào phù hợp với '{query}'.\n"
            f"Thử các từ khóa: transformer, BERT, GPT, ResNet, ViT, RAG, LLaMA, "
            f"reinforcement learning, education, knowledge tracing."
        )

    out = f"Tìm thấy {len(matched)} bài báo cho '{query}':\n"
    for p in matched[:5]:
        out += (
            f"\n[{p['id']}] {p['title']} ({p['year']})\n"
            f"  Tác giả: {p['authors']}\n"
            f"  Lĩnh vực: {p['field']} | Citations: {p['citations']:,} | Chất lượng: {p['quality_score']}/10\n"
        )
    if len(matched) > 5:
        out += f"\n... và {len(matched) - 5} bài khác.\n"
    out += "\nDùng get_paper_details(ID) để xem chi tiết hoặc compare_papers(ID1 ID2) để so sánh."
    return out


def compare_papers(query: str) -> str:
    """So sánh 2 bài báo theo research gap và chất lượng."""
    ids = re.findall(r'P\d+', query.upper())
    if len(ids) < 2:
        return f"Cần 2 Paper ID. Ví dụ: compare_papers(P001 P002). Có sẵn: {', '.join(sorted(PAPER_DB.keys()))}"

    p1, p2 = PAPER_DB.get(ids[0]), PAPER_DB.get(ids[1])
    if not p1:
        return f"Không tìm thấy '{ids[0]}'. Có sẵn: {', '.join(sorted(PAPER_DB.keys()))}"
    if not p2:
        return f"Không tìm thấy '{ids[1]}'. Có sẵn: {', '.join(sorted(PAPER_DB.keys()))}"

    out = f"So sánh [{p1['id']}] vs [{p2['id']}]\n\n"
    for p in (p1, p2):
        out += f"[{p['id']}] {p['title']} ({p['year']})\n"
        out += f"  Tác giả: {p['authors']} | Venue: {p['venue']}\n"
        out += f"  Citations: {p['citations']:,} | Chất lượng: {p['quality_score']}/10\n"
        out += f"  Khoảng trống nghiên cứu:\n"
        for g in p["research_gaps"]:
            out += f"    - {g}\n"
        out += "\n"

    best_q = p1 if p1["quality_score"] >= p2["quality_score"] else p2
    best_c = p1 if p1["citations"] >= p2["citations"] else p2
    out += f"Khuyến nghị:\n"
    out += f"  Chất lượng cao hơn : [{best_q['id']}] {best_q['title'][:40]} ({best_q['quality_score']}/10)\n"
    out += f"  Ảnh hưởng cao hơn  : [{best_c['id']}] {best_c['title'][:40]} ({best_c['citations']:,} citations)\n"
    return out


def get_paper_details(query: str) -> str:
    """Lấy thông tin đầy đủ của một bài báo theo ID."""
    m = re.search(r'P\d+', query.upper())
    if not m:
        return f"Cần Paper ID hợp lệ (ví dụ: P001). Có sẵn: {', '.join(sorted(PAPER_DB.keys()))}"

    p = PAPER_DB.get(m.group())
    if not p:
        return f"Không tìm thấy '{m.group()}'. Có sẵn: {', '.join(sorted(PAPER_DB.keys()))}"

    out = f"[{p['id']}] {p['title']}\n"
    out += f"Tác giả  : {p['authors']}\n"
    out += f"Năm/Venue: {p['year']} — {p['venue']}\n"
    out += f"Lĩnh vực : {p['field']} | DOI: {p['doi']}\n"
    out += f"Citations: {p['citations']:,} | Chất lượng: {p['quality_score']}/10\n\n"
    out += f"Tóm tắt:\n{p['abstract']}\n\n"
    out += f"Từ khóa: {', '.join(p['keywords'])}\n\n"
    out += f"Khoảng trống nghiên cứu:\n"
    for i, g in enumerate(p["research_gaps"], 1):
        out += f"  {i}. {g}\n"
    return out


# Đăng ký tools cho Agent
TOOLS = [
    {
        "name": "search_papers",
        "description": (
            "Tìm bài báo theo từ khóa, lĩnh vực hoặc khoảng năm. "
            "Input: chuỗi từ khóa, ví dụ 'transformer NLP' hoặc 'education 2022-2023'. "
            "Output: danh sách bài báo phù hợp với ID, tiêu đề, tác giả, năm, citations, chất lượng."
        ),
        "function": search_papers,
    },
    {
        "name": "compare_papers",
        "description": (
            "So sánh 2 bài báo theo research gap và chất lượng. "
            "Input: 2 Paper ID cách nhau bởi dấu cách, ví dụ 'P001 P002'. "
            "Output: bảng so sánh chi tiết khoảng trống nghiên cứu, phương pháp, chất lượng."
        ),
        "function": compare_papers,
    },
    {
        "name": "get_paper_details",
        "description": (
            "Lấy thông tin đầy đủ của 1 bài báo: tóm tắt, tác giả, DOI, research gaps. "
            "Input: Paper ID, ví dụ 'P001' hoặc 'P009'. "
            "Output: thông tin chi tiết của bài báo."
        ),
        "function": get_paper_details,
    },
]

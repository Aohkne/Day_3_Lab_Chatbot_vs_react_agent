"""
Tests cho 3 tools nghien cuu hoc thuat (search_papers, compare_papers, get_paper_details).
Thuan logic, khong phu thuoc LLM/provider nao -> chay duoc o moi moi truong.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.research_tools import search_papers, compare_papers, get_paper_details


def test_search_papers_finds_by_keyword():
    result = search_papers("transformer NLP")
    assert "P001" in result  # Attention Is All You Need
    assert "Tìm thấy" in result


def test_search_papers_filters_by_year_range():
    result = search_papers("education 2022-2023")
    assert "P008" in result or "P009" in result
    assert "P005" not in result  # ResNet (2016) khong nam trong khoang nam


def test_search_papers_no_match_returns_helpful_message():
    result = search_papers("Nguyen Van XYZ quantum computing")
    assert "Không tìm thấy" in result


def test_get_paper_details_valid_id():
    result = get_paper_details("P002")
    assert "BERT" in result
    assert "Devlin" in result


def test_get_paper_details_invalid_id():
    result = get_paper_details("P999")
    assert "Không tìm thấy" in result


def test_get_paper_details_missing_id_in_query():
    result = get_paper_details("bai bao khong co ma so")
    assert "Cần Paper ID hợp lệ" in result


def test_compare_papers_valid_ids():
    result = compare_papers("P001 P002")
    assert "P001" in result and "P002" in result
    assert "Khuyến nghị" in result


def test_compare_papers_missing_second_id():
    result = compare_papers("P001")
    assert "Cần 2 Paper ID" in result


def test_compare_papers_unknown_id():
    result = compare_papers("P001 P999")
    assert "Không tìm thấy" in result

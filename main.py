"""
main.py — So sanh 3 approach tren 5 test cases nghien cuu hoc thuat.
Provider (openai | google | local) chon qua DEFAULT_PROVIDER trong .env,
mac dinh "local" (Phi-3-mini-4k-instruct-q4.gguf, khong can internet).
Chay: python main.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from src.core.llm_provider import LLMProvider
from src.core.provider_factory import get_provider
from src.agent.chatbot import run_chatbot
from src.agent.agent import ReActAgent
from src.agent.agent_v1 import ReActAgentV1
from src.tools.research_tools import TOOLS

load_dotenv()

# 5 Test Cases Hoc Thuat
TEST_CASES = [
    {
        "id": 1,
        "type": "Agent-Favorable",
        "query": "Tim bai bao ve transformer trong NLP",
        "note": "Can tool search_papers de lay danh sach paper cu the",
    },
    {
        "id": 2,
        "type": "Chatbot-Favorable",
        "query": "Attention mechanism la gi va tai sao no quan trong trong deep learning?",
        "note": "Cau hoi khai niem tong quat, chatbot co the tra loi tot",
    },
    {
        "id": 3,
        "type": "Agent-Favorable",
        "query": "So sanh bai bao BERT va GPT-3 ve research gap va chat luong",
        "note": "Can search_papers + compare_papers de phan tich cu the",
    },
    {
        "id": 4,
        "type": "Agent-Favorable",
        "query": "Tim bai bao nam 2022-2023 ve reinforcement learning trong giao duc",
        "note": "Can tool voi filter nam va linh vuc cu the",
    },
    {
        "id": 5,
        "type": "Edge Case (Hallucination Test)",
        "query": "Tim bai bao cua tac gia Nguyen Van XYZ ve quantum computing",
        "note": "Tac gia va topic khong co trong DB -> test hallucination: chatbot de bia, V1 phai tu choi",
    },
]


def get_llm() -> LLMProvider:
    """
    Khoi tao LLMProvider dung chung cho ca 3 approach (Chatbot/Agent/Agent V1).
    Provider duoc chon qua bien moi truong DEFAULT_PROVIDER (openai|google|local,
    mac dinh "local") — doi provider chi can sua .env, khong can sua code o day.
    """
    provider_name = os.getenv("DEFAULT_PROVIDER", "local")
    llm = get_provider(provider_name=provider_name)
    print(f"  [Provider] {provider_name} | [Model] {llm.model_name}")
    return llm


def run_chatbot_test(query: str) -> dict:
    result = run_chatbot(query)
    return {
        "answer": result["content"],
        "tokens": result["usage"].get("total_tokens", 0),
        "latency_ms": result["latency_ms"],
        "steps": 1,
        "trace": "",
    }


def run_agent_test(llm: LLMProvider, query: str) -> dict:
    agent = ReActAgent(llm=llm, tools=TOOLS, max_steps=5)
    result = agent.run(query)
    return {
        "answer": result["answer"],
        "tokens": result["total_tokens"],
        "latency_ms": result["total_latency_ms"],
        "steps": result["steps"],
        "trace": agent.get_trace_report(),
    }


def run_agent_v1_test(llm: LLMProvider, query: str) -> dict:
    agent = ReActAgentV1(llm=llm, tools=TOOLS, max_steps=4)
    result = agent.run(query)
    return {
        "answer": result["answer"],
        "tokens": result["total_tokens"],
        "latency_ms": result["total_latency_ms"],
        "steps": result["steps"],
        "trace": agent.get_trace_report(),
    }


def sep(char="=", w=80):
    print(char * w)


def main():
    sep()
    print("  LAB 3: CHATBOT vs REACT AGENT vs AGENT V1")
    print("  Domain  : Academic Research Assistant")
    print("  Provider: chon qua DEFAULT_PROVIDER trong .env (openai | google | local)")
    sep()

    print("\n[INIT] Loading LLM provider...")
    try:
        llm = get_llm()
        print("  [OK] Provider ready!\n")
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        print("  Neu dung provider 'local', dam bao models/Phi-3-mini-4k-instruct-q4.gguf ton tai.")
        sys.exit(1)
    except ValueError as exc:
        print(f"  [ERROR] {exc}")
        sys.exit(1)
    except ModuleNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        print("  Chay 'pip install -r requirements.txt' de cai dat dependency cua provider dang chon.")
        sys.exit(1)

    all_results = []

    for test in TEST_CASES:
        sep()
        print(f"  TEST {test['id']}/5 | {test['type']}")
        print(f"  Query: {test['query']}")
        print(f"  Note : {test['note']}")
        sep("-")

        row: dict = {"test": test}

        # --- Chatbot ---
        print("\n[CHATBOT BASELINE]")
        try:
            cr = run_chatbot_test(test["query"])
            print(f"  Answer : {cr['answer'][:280]}...")
            print(f"  Metrics: {cr['latency_ms']:.0f}ms | {cr['tokens']} tokens | 1 step")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            cr = {"answer": f"Error: {exc}", "tokens": 0, "latency_ms": 0, "steps": 0, "trace": ""}
        row["chatbot"] = cr

        # --- Agent ---
        print("\n[REACT AGENT]")
        try:
            ar = run_agent_test(llm, test["query"])
            print(f"  Answer : {ar['answer'][:280]}...")
            print(f"  Metrics: {ar['latency_ms']:.0f}ms | {ar['tokens']} tokens | {ar['steps']} steps")
            if ar["trace"]:
                print(f"  Trace  :\n{ar['trace']}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            ar = {"answer": f"Error: {exc}", "tokens": 0, "latency_ms": 0, "steps": 0, "trace": ""}
        row["agent"] = ar

        # --- Agent V1 ---
        print("\n[REACT AGENT V1 (Anti-Hallucination)]")
        try:
            a1r = run_agent_v1_test(llm, test["query"])
            print(f"  Answer : {a1r['answer'][:280]}...")
            print(f"  Metrics: {a1r['latency_ms']:.0f}ms | {a1r['tokens']} tokens | {a1r['steps']} steps")
            if a1r["trace"]:
                print(f"  Trace  :\n{a1r['trace']}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            a1r = {"answer": f"Error: {exc}", "tokens": 0, "latency_ms": 0, "steps": 0, "trace": ""}
        row["agent_v1"] = a1r

        all_results.append(row)
        print()


    # Summary table
    sep()
    print("  BANG TONG KET")
    sep()
    hdr = (f"{'#':<4} {'Type':<28} {'Chat Tok':<10} {'Agt Tok':<10} "
           f"{'V1 Tok':<10} {'Agt Step':<10} {'V1 Step'}")
    print(hdr)
    sep("-")
    for r in all_results:
        tid   = r["test"]["id"]
        ttype = r["test"]["type"][:26]
        ct    = r["chatbot"]["tokens"]
        at    = r["agent"]["tokens"]
        v1t   = r["agent_v1"]["tokens"]
        ags   = r["agent"]["steps"]
        v1s   = r["agent_v1"]["steps"]
        print(f"{tid:<4} {ttype:<28} {ct:<10} {at:<10} {v1t:<10} {ags:<10} {v1s}")

    sep()
    print("  Ghi chu:")
    print("  - Agent & V1 su dung tools -> nhieu tokens hon Chatbot (binh thuong)")
    print("  - V1 co 6 RULE chong hallucination (max_steps=4, bat buoc dung tool)")
    print("  - Test 5 (Edge): Chatbot de bia, V1 phai bao 'Khong tim thay'")
    sep()


if __name__ == "__main__":
    main()
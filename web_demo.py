"""
web_demo.py — Demo giao dien web Neo Brutalism: Chatbot vs Agent vs Agent V1.
Model: Phi-3-mini-4k-instruct-q4.gguf (local)
Chay: python web_demo.py
Mo trinh duyet: http://localhost:5000
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from src.core.local_provider import LocalProvider
from src.agent.chatbot import SYSTEM_PROMPT as DEFAULT_CHATBOT_PROMPT
from src.agent.agent import ReActAgent
from src.agent.agent_v1 import ReActAgentV1
from src.tools.research_tools import TOOLS

load_dotenv()

app = Flask(__name__)

# ======
# Single shared LLM instance + lock (llama-cpp is NOT thread-safe)
# ======
_llm_instance: LocalProvider = None
_llm_lock = threading.Lock()


def get_llm() -> LocalProvider:
    global _llm_instance
    if _llm_instance is None:
        model_path = os.getenv("LOCAL_MODEL_PATH", "")
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models", "Phi-3-mini-4k-instruct-q4.gguf",
            )
        model_path = os.path.normpath(model_path)
        _llm_instance = LocalProvider(model_path=model_path, n_ctx=4096)
    return _llm_instance


# ======
# HTML Template — Neo Brutalism
# ======
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LAB 3 — Research Assistant</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', system-ui, sans-serif; background: #f4f4f5; color: #18181b; font-size: 14px; }

    header {
      background: #fff; border-bottom: 1px solid #e4e4e7;
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    header h1 { font-size: 1.1rem; font-weight: 700; color: #18181b; }
    header p  { font-size: 0.75rem; color: #71717a; margin-top: 2px; }
    .model-tag {
      font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700;
      background: #18181b; color: #fafafa; padding: 4px 10px; border-radius: 4px;
    }

    .main { max-width: 1320px; margin: 0 auto; padding: 20px 16px; }

    .input-card {
      background: #fff; border: 1px solid #e4e4e7; border-radius: 8px;
      padding: 16px; margin-bottom: 20px;
    }
    .input-row { display: flex; border: 1px solid #d4d4d8; border-radius: 6px; overflow: hidden; }
    .input-row input {
      flex: 1; padding: 11px 14px; font-size: 0.9rem; font-family: 'Inter', sans-serif;
      border: none; outline: none; background: #fff; color: #18181b;
    }
    .btn-send {
      padding: 11px 20px; background: #18181b; color: #fff;
      font-weight: 600; font-size: 0.85rem; border: none; cursor: pointer;
    }
    .btn-send:hover { background: #3f3f46; }
    .btn-send:disabled { background: #a1a1aa; cursor: wait; }

    .suggestions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
    .sug-label { font-size: 0.72rem; color: #71717a; align-self: center; margin-right: 4px; }
    .sug-btn {
      padding: 5px 11px; border: 1px solid #e4e4e7; background: #fafafa;
      font-size: 0.75rem; font-family: 'Inter', sans-serif; cursor: pointer; border-radius: 4px;
      color: #3f3f46;
    }
    .sug-btn:hover { background: #f4f4f5; border-color: #a1a1aa; }

    .sp-section { margin-top: 14px; border-top: 1px solid #f4f4f5; padding-top: 12px; }
    .sp-toggle-btn {
      background: none; border: 1px solid #e4e4e7; padding: 5px 12px;
      font-size: 0.75rem; color: #52525b; cursor: pointer; border-radius: 4px;
      font-family: 'Inter', sans-serif;
    }
    .sp-toggle-btn:hover { background: #f4f4f5; }
    .sp-panel { margin-top: 10px; }
    .sp-textarea {
      width: 100%; padding: 10px 12px; font-size: 0.8rem;
      font-family: 'JetBrains Mono', monospace;
      border: 1px solid #d4d4d8; border-radius: 6px; resize: vertical;
      outline: none; color: #18181b; background: #fafafa; min-height: 80px;
    }
    .sp-textarea:focus { border-color: #a1a1aa; background: #fff; }
    .sp-hint { font-size: 0.7rem; color: #a1a1aa; margin-top: 5px; }

    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

    .panel { background: #fff; border: 1px solid #e4e4e7; border-radius: 8px; overflow: hidden; }

    .panel-header {
      padding: 10px 14px; font-weight: 600; font-size: 0.8rem;
      display: flex; align-items: center; gap: 8px;
      border-bottom: 1px solid #e4e4e7;
    }
    .panel-chatbot .panel-header  { background: #eff6ff; color: #1d4ed8; }
    .panel-agent   .panel-header  { background: #fff1f2; color: #be123c; }
    .panel-agentv1 .panel-header  { background: #f0fdf4; color: #15803d; }

    .badge {
      font-size: 0.62rem; padding: 2px 6px; border-radius: 3px; font-weight: 600;
      background: rgba(0,0,0,0.08);
    }

    .panel-body { padding: 14px; min-height: 150px; }
    .answer-text { font-size: 0.85rem; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
    .placeholder { color: #a1a1aa; font-size: 0.82rem; font-style: italic; }

    .metrics {
      border-top: 1px solid #f4f4f5; padding: 8px 14px;
      display: flex; gap: 14px; background: #fafafa;
    }
    .metric { font-size: 0.72rem; color: #71717a; }
    .metric b { color: #3f3f46; }

    .trace-btn {
      width: 100%; border: none; border-top: 1px solid #f4f4f5;
      background: #fafafa; padding: 7px 14px; text-align: left;
      font-size: 0.72rem; cursor: pointer; color: #71717a;
      font-family: 'Inter', sans-serif;
    }
    .trace-btn:hover { background: #f4f4f5; }

    .trace-body {
      display: none; border-top: 1px solid #e4e4e7;
      background: #18181b; padding: 12px; max-height: 220px; overflow-y: auto;
      font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    }
    .trace-body.open { display: block; }
    .trace-step { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #27272a; }
    .trace-step:last-child { border: none; margin: 0; }
    .t-num     { color: #52525b; margin-bottom: 4px; }
    .t-thought { color: #fbbf24; margin: 2px 0; white-space: pre-wrap; }
    .t-action  { color: #34d399; margin: 2px 0; }
    .t-obs     { color: #60a5fa; margin: 2px 0; }
    .t-final   { color: #fb923c; font-weight: 700; margin: 2px 0; }
    .t-rule    { color: #f87171; font-weight: 700; margin: 2px 0; }

    .loading { padding: 32px; text-align: center; color: #a1a1aa; }
    .spinner {
      display: inline-block; width: 20px; height: 20px;
      border: 2px solid #e4e4e7; border-top-color: #18181b;
      border-radius: 50%; animation: spin 0.7s linear infinite;
      vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .error { background: #fff1f2; border: 1px solid #fecdd3; padding: 10px 14px; color: #be123c; border-radius: 4px; }

    footer {
      margin-top: 24px; padding: 12px 16px; text-align: center;
      font-size: 0.7rem; color: #a1a1aa; border-top: 1px solid #e4e4e7;
    }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Academic Research Assistant — LAB 3</h1>
    <p>Chatbot vs ReAct Agent vs Agent V1 | Phi-3-mini-4k (local, không cần internet)</p>
  </div>
  <span class="model-tag">PHI-3-MINI · LOCAL</span>
</header>

<div class="main">

  <div class="input-card">
    <div class="input-row">
      <input type="text" id="queryInput"
             placeholder="Tìm bài báo về transformer, so sánh BERT vs GPT-3..."
             onkeydown="if(event.key==='Enter') sendQuery()">
      <button class="btn-send" id="sendBtn" onclick="sendQuery()">Gửi →</button>
    </div>
    <div class="suggestions">
      <span class="sug-label">Gợi ý:</span>
      <button class="sug-btn" onclick="ask('Tìm bài báo về transformer trong NLP')">Transformer NLP</button>
      <button class="sug-btn" onclick="ask('Attention mechanism là gì?')">Attention là gì?</button>
      <button class="sug-btn" onclick="ask('So sánh BERT và GPT-3 về research gap')">BERT vs GPT-3</button>
      <button class="sug-btn" onclick="ask('Tìm bài báo 2022-2023 về reinforcement learning giáo dục')">RL Giáo dục 2022-2023</button>
      <button class="sug-btn" onclick="ask('Lấy chi tiết bài báo P001')">Chi tiết P001</button>
      <button class="sug-btn" onclick="ask('Tìm bài báo của tác giả Nguyễn Văn XYZ về quantum computing')">Edge case: không tồn tại</button>
    </div>

    <div class="sp-section">
      <button class="sp-toggle-btn" onclick="toggleSP()">⚙ Tùy chỉnh System Prompt &nbsp;<span id="sp-arrow">▼</span></button>
      <div id="sp-panel" style="display:none" class="sp-panel">
        <div class="suggestions" style="margin-top:10px">
          <span class="sug-label">Ví dụ:</span>
          <button class="sug-btn" onclick="setSP('detailed')">&#128203; Chi tiết</button>
          <button class="sug-btn" onclick="setSP('concise')">&#9889; Ngắn gọn</button>
          <button class="sug-btn" onclick="setSP(null)">✕ Xoá (dùng mặc định)</button>
        </div>
        <textarea id="systemPromptInput" class="sp-textarea" rows="4"
          placeholder="Nhập system prompt tùy chỉnh... Để trống = dùng prompt mặc định."></textarea>
        <p class="sp-hint">💡 Áp dụng cho Chatbot (thay thế hoàn toàn). Với Agent, hướng dẫn được nối sau prompt gốc để giữ nguyên format.</p>
      </div>
    </div>
  </div>

  <div class="grid">

    <div class="panel panel-chatbot">
      <div class="panel-header">
        Chatbot Baseline <span class="badge">KHÔNG TOOL</span>
      </div>
      <div class="panel-body" id="chatbot-body">
        <p class="placeholder">Nhập câu hỏi để bắt đầu.</p>
      </div>
      <div class="metrics" id="chatbot-metrics" style="display:none">
        <span class="metric">Thời gian: <b id="c-lat">—</b></span>
        <span class="metric">Tokens: <b id="c-tok">—</b></span>
        <span class="metric">Bước: <b>1</b></span>
      </div>
    </div>

    <div class="panel panel-agent">
      <div class="panel-header">
        ReAct Agent <span class="badge">TOOLS + PROMPT</span>
      </div>
      <div class="panel-body" id="agent-body">
        <p class="placeholder">Nhập câu hỏi để bắt đầu.</p>
      </div>
      <div class="metrics" id="agent-metrics" style="display:none">
        <span class="metric">Thời gian: <b id="a-lat">—</b></span>
        <span class="metric">Tokens: <b id="a-tok">—</b></span>
        <span class="metric">Bước: <b id="a-stp">—</b></span>
      </div>
      <button class="trace-btn" id="agent-trace-toggle" style="display:none" onclick="toggleTrace('agent')">
        ▼ Xem quá trình suy luận
      </button>
      <div class="trace-body" id="agent-trace-body"></div>
    </div>

    <div class="panel panel-agentv1">
      <div class="panel-header">
        Agent V1 <span class="badge">CHỐNG HALLUCINATION</span>
      </div>
      <div class="panel-body" id="agentv1-body">
        <p class="placeholder">Nhập câu hỏi để bắt đầu.</p>
      </div>
      <div class="metrics" id="agentv1-metrics" style="display:none">
        <span class="metric">Thời gian: <b id="v-lat">—</b></span>
        <span class="metric">Tokens: <b id="v-tok">—</b></span>
        <span class="metric">Bước: <b id="v-stp">—</b></span>
      </div>
      <button class="trace-btn" id="agentv1-trace-toggle" style="display:none" onclick="toggleTrace('agentv1')">
        ▼ Xem quá trình suy luận
      </button>
      <div class="trace-body" id="agentv1-trace-body"></div>
    </div>

  </div>
</div>

<footer>
  LAB 3 — Lê Hữu Khoa — 2A202600863 — VinUniversity 2026 &nbsp;|&nbsp; Phi-3-mini-4k-instruct (local)
</footer>

<script>
  function ask(q) {
    document.getElementById('queryInput').value = q;
    sendQuery();
  }

  function esc(t) {
    return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }

  const SP_PRESETS = {
    detailed: 'Bạn là trợ lý nghiên cứu học thuật chuyên sâu. Hãy trả lời đầy đủ, có cấu trúc rõ ràng với các mục: Định nghĩa, Nguyên lý hoạt động, Ứng dụng thực tiễn, và Hạn chế. Trích dẫn ví dụ cụ thể khi có thể. Viết bằng tiếng Việt có dấu, học thuật.',
    concise:  'Bạn là trợ lý nghiên cứu. Trả lời ngắn gọn, súc tích trong 2–3 câu. Chỉ nêu ý chính quan trọng nhất, không giải thích dài dòng. Viết bằng tiếng Việt.',
  };

  function toggleSP() {
    const panel = document.getElementById('sp-panel');
    const arrow = document.getElementById('sp-arrow');
    const open = panel.style.display === 'none';
    panel.style.display = open ? 'block' : 'none';
    arrow.textContent = open ? '▲' : '▼';
  }

  function setSP(preset) {
    document.getElementById('sp-panel').style.display = 'block';
    document.getElementById('sp-arrow').textContent = '▲';
    document.getElementById('systemPromptInput').value = preset ? SP_PRESETS[preset] : '';
  }

  function toggleTrace(which) {
    const body = document.getElementById(which + '-trace-body');
    const btn  = document.getElementById(which + '-trace-toggle');
    const open = body.classList.toggle('open');
    btn.textContent = (open ? '▲' : '▼') + ' Xem quá trình suy luận';
  }

  function buildTrace(trace) {
    if (!trace || !trace.length) return '<p style="color:#52525b">Không có trace.</p>';
    return trace.map(s => {
      let h = '<div class="trace-step">';
      h += '<div class="t-num">BƯỚC ' + s.step + '</div>';
      if (s.thought)     h += '<div class="t-thought">Suy nghĩ: ' + esc(s.thought) + '</div>';
      if (s.action)      h += '<div class="t-action">Hành động: ' + esc(s.action) + '</div>';
      if (s.observation) h += '<div class="t-obs">Quan sát: ' + esc(s.observation.substring(0,200)) + '...</div>';
      if (s.final_answer)h += '<div class="t-final">Kết luận: ' + esc(s.final_answer.substring(0,200)) + '</div>';
      if (s.constraint)  h += '<div class="t-rule">[Ràng buộc: ' + esc(s.constraint) + ']</div>';
      return h + '</div>';
    }).join('');
  }

  function setLoading() {
    const html = '<div class="loading"><span class="spinner"></span>Đang xử lý...</div>';
    ['chatbot-body','agent-body','agentv1-body'].forEach(id => document.getElementById(id).innerHTML = html);
    ['chatbot-metrics','agent-metrics','agentv1-metrics'].forEach(id => document.getElementById(id).style.display = 'none');
    ['agent-trace-toggle','agentv1-trace-toggle'].forEach(id => document.getElementById(id).style.display = 'none');
    ['agent-trace-body','agentv1-trace-body'].forEach(id => document.getElementById(id).classList.remove('open'));
  }

  function sendQuery() {
    const q = document.getElementById('queryInput').value.trim();
    if (!q) return;
    const btn = document.getElementById('sendBtn');
    btn.disabled = true; btn.textContent = 'Đang xử lý...';
    setLoading();

    const headers = { 'Content-Type': 'application/json' };
    const system_prompt = document.getElementById('systemPromptInput').value.trim() || null;
    const body = JSON.stringify({ query: q, system_prompt });
    let pending = 3;

    function oneDone() {
      pending--;
      if (pending === 0) { btn.disabled = false; btn.textContent = 'Gửi →'; }
    }

    // --- Chatbot (kết quả nào xong trước hiện trước) ---
    fetch('/ask/chatbot', { method: 'POST', headers, body })
      .then(r => r.ok ? r.json() : Promise.reject('Lỗi server: ' + r.status))
      .then(d => {
        document.getElementById('chatbot-body').innerHTML = '<p class="answer-text">' + esc(d.answer) + '</p>';
        document.getElementById('chatbot-metrics').style.display = 'flex';
        document.getElementById('c-lat').textContent = d.latency_ms + 'ms';
        document.getElementById('c-tok').textContent = d.tokens;
      })
      .catch(err => {
        document.getElementById('chatbot-body').innerHTML = '<div class="error">Lỗi: ' + esc(String(err)) + '</div>';
      })
      .finally(oneDone);

    // --- ReAct Agent ---
    fetch('/ask/agent', { method: 'POST', headers, body })
      .then(r => r.ok ? r.json() : Promise.reject('Lỗi server: ' + r.status))
      .then(d => {
        document.getElementById('agent-body').innerHTML = '<p class="answer-text">' + esc(d.answer) + '</p>';
        document.getElementById('agent-metrics').style.display = 'flex';
        document.getElementById('a-lat').textContent = d.latency_ms + 'ms';
        document.getElementById('a-tok').textContent = d.tokens;
        document.getElementById('a-stp').textContent = d.steps;
        if (d.trace && d.trace.length) {
          document.getElementById('agent-trace-toggle').style.display = 'block';
          document.getElementById('agent-trace-body').innerHTML = buildTrace(d.trace);
        }
      })
      .catch(err => {
        document.getElementById('agent-body').innerHTML = '<div class="error">Lỗi: ' + esc(String(err)) + '</div>';
      })
      .finally(oneDone);

    // --- Agent V1 ---
    fetch('/ask/agentv1', { method: 'POST', headers, body })
      .then(r => r.ok ? r.json() : Promise.reject('Lỗi server: ' + r.status))
      .then(d => {
        document.getElementById('agentv1-body').innerHTML = '<p class="answer-text">' + esc(d.answer) + '</p>';
        document.getElementById('agentv1-metrics').style.display = 'flex';
        document.getElementById('v-lat').textContent = d.latency_ms + 'ms';
        document.getElementById('v-tok').textContent = d.tokens;
        document.getElementById('v-stp').textContent = d.steps;
        if (d.trace && d.trace.length) {
          document.getElementById('agentv1-trace-toggle').style.display = 'block';
          document.getElementById('agentv1-trace-body').innerHTML = buildTrace(d.trace);
        }
      })
      .catch(err => {
        document.getElementById('agentv1-body').innerHTML = '<div class="error">Lỗi: ' + esc(String(err)) + '</div>';
      })
      .finally(oneDone);
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def _build_trace(raw_trace):
    result = []
    for s in raw_trace:
        item = {"step": s["step"]}
        text = s.get("llm_output", "")
        if "Thought:" in text:
            item["thought"] = text.split("Thought:")[-1].split("Action:")[0].split("Final Answer:")[0].strip()[:300]
        if "action" in s:
            item["action"] = s["action"]
        if "observation" in s:
            item["observation"] = s["observation"][:400]
        if "final_answer" in s:
            item["final_answer"] = s["final_answer"][:400]
        if s.get("constraint_triggered"):
            item["constraint"] = s["constraint_triggered"]
        result.append(item)
    return result


@app.route("/ask/chatbot", methods=["POST"])
def ask_chatbot():
    data = request.json
    query = (data.get("query") or "").strip()
    system_prompt = data.get("system_prompt") or DEFAULT_CHATBOT_PROMPT
    if not query:
        return jsonify({"error": "Empty query"}), 400
    llm = get_llm()
    with _llm_lock:
        raw = llm.generate(query, system_prompt=system_prompt)
    return jsonify({
        "answer": raw["content"],
        "tokens": raw["usage"].get("total_tokens", 0),
        "latency_ms": int(raw["latency_ms"]),
        "steps": 1,
    })


@app.route("/ask/agent", methods=["POST"])
def ask_agent():
    data = request.json
    query = (data.get("query") or "").strip()
    extra = data.get("system_prompt") or ""
    if not query:
        return jsonify({"error": "Empty query"}), 400
    llm = get_llm()
    agent = ReActAgent(llm=llm, tools=TOOLS, max_steps=5, extra_instructions=extra)
    with _llm_lock:
        raw = agent.run(query)
    return jsonify({
        "answer": raw["answer"],
        "tokens": raw["total_tokens"],
        "latency_ms": int(raw["total_latency_ms"]),
        "steps": raw["steps"],
        "trace": _build_trace(raw["trace"]),
    })


@app.route("/ask/agentv1", methods=["POST"])
def ask_agentv1():
    data = request.json
    query = (data.get("query") or "").strip()
    extra = data.get("system_prompt") or ""
    if not query:
        return jsonify({"error": "Empty query"}), 400
    llm = get_llm()
    agent_v1 = ReActAgentV1(llm=llm, tools=TOOLS, max_steps=4, extra_instructions=extra)
    with _llm_lock:
        raw = agent_v1.run(query)
    return jsonify({
        "answer": raw["answer"],
        "tokens": raw["total_tokens"],
        "latency_ms": int(raw["total_latency_ms"]),
        "steps": raw["steps"],
        "trace": _build_trace(raw["trace"]),
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  ACADEMIC RESEARCH ASSISTANT — WEB DEMO")
    print("  Style  : Neo Brutalism")
    print("  Model  : Phi-3-mini-4k-instruct-q4.gguf (local)")
    print("  Access : http://localhost:5000")
    print("=" * 55)
    # Pre-load model before accepting requests
    print("[INIT] Loading local model...")
    get_llm()
    print("[OK]   Model ready.\n")
    app.run(debug=False, port=5000, threaded=True)


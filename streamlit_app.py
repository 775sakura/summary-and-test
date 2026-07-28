import os
import json
import io
import uuid
import base64
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from pypdf import PdfReader
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="講義ノート AI アシスタント",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini Client automatically from environment or Streamlit secrets
def fetch_env_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def get_gemini_client():
    key = fetch_env_api_key()
    if not key:
        return None
    return genai.Client(api_key=key)

# Initialize Session States
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "folders" not in st.session_state:
    st.session_state.folders = ["指定なし", "物理学", "数学", "コンピュータ科学", "経済学・ビジネス"]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
    }
    .file-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .page-box {
        background-color: #f1f5f9;
        border-left: 4px solid #4f46e5;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for idx, page in enumerate(reader.pages, 1):
        extracted = page.extract_text()
        if extracted:
            text += f"\n--- [ページ {idx}] ---\n" + extracted + "\n"
    return text

# Helper to calculate readable file size
def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

# Pure HTML5 Handwriting Canvas Component (Zero external Python dependencies)
def render_html5_canvas(quiz_id: str):
    canvas_html = f"""
    <div style="font-family: sans-serif; background: #0f172a; padding: 12px; border-radius: 12px; color: white;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 12px; font-weight: bold; color: #94a3b8;">🎨 手書きキャンバス（マウス/タッチ対応）</span>
            <button onclick="clearCanvas()" style="background: #334155; color: white; border: none; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px;">消去</button>
        </div>
        <canvas id="canvas_{quiz_id}" width="500" height="200" style="background: #1e293b; border: 1px solid #475569; border-radius: 8px; cursor: crosshair; touch-action: none;"></canvas>
        <p style="font-size: 11px; color: #94a3b8; margin-top: 6px;">※ 描画後、下の「ノート写真/キャンバス画像をアップロード」より手書き画像を添付するか、そのまま保存してください。</p>
    </div>
    <script>
        const canvas = document.getElementById("canvas_{quiz_id}");
        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.lineCap = "round";

        let drawing = false;

        function startDraw(e) {{
            drawing = true;
            ctx.beginPath();
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;
            ctx.moveTo(x, y);
        }}

        function draw(e) {{
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;
            ctx.lineTo(x, y);
            ctx.stroke();
        }}

        function stopDraw() {{
            drawing = false;
        }}

        function clearCanvas() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }}

        canvas.addEventListener("mousedown", startDraw);
        canvas.addEventListener("mousemove", draw);
        canvas.addEventListener("mouseup", stopDraw);
        canvas.addEventListener("mouseleave", stopDraw);

        canvas.addEventListener("touchstart", startDraw);
        canvas.addEventListener("touchmove", draw);
        canvas.addEventListener("touchend", stopDraw);
    </script>
    """
    components.html(canvas_html, height=270)

# Analyze lecture material using Gemini
def analyze_lecture_material(client: genai.Client, parts: list):
    system_instruction = """
あなたは熱心で教え上手な大学教授兼AI教育スペシャリストです。
アップロードされた講義資料（スライド画像、PDFテキスト、音声、メモ等）を細かく分析し、以下の仕様に従ってJSONデータを生成してください。

【出力要件】：
1. title: 講義内容から推測される最も適切なタイトル（日本語）
2. summaryText: 講義資料全体の概要・結論・背景を網羅した包括的要約（数式は $...$ や $$...$$ を使用）
3. pageExplanations: 資料のページ・スライドごと（または論理的セクションごと）に、講義の講師（先生）として親しみやすく詳しく語りかける形式の解説リスト。
   - page: ページ番号またはセクション番号 (数値)
   - title: そのページ/セクションの見出しタイトル
   - content: 講師としての詳しい口語調解説（「皆さんこんにちは。このページでは...」「特にこの公式は...」のように丁寧かつ具体的に講義する文章）
4. keyPoints: 重要ポイントの箇条書き（4〜8項目）
5. quizzes: デフォルトで【合計10問】の理解度テストを作成してください。
   - 選択問題 (type: "multiple-choice"): 4択形式
   - 穴埋め記述問題 (type: "fill-in-the-blank"): 短語・数式を答える形式
   - 思考・計算手書き問題 (type: "open-ended"): 途中の計算過程や概念図・手書き記述を要求する形式

【JSON構造仕様】:
{
  "title": "講義タイトル",
  "summaryText": "資料全体の要約テキスト...",
  "pageExplanations": [
    {
      "page": 1,
      "title": "第1ページ: 導入と基本概念",
      "content": "【講師解説】皆さん、こんにちは！本講義ではまず..."
    }
  ],
  "keyPoints": ["ポイント1", "ポイント2", ...],
  "quizzes": [
    {
      "id": "q1",
      "type": "multiple-choice",
      "question": "問題文1...",
      "choices": ["選択肢A", "選択肢B", "選択肢C", "選択肢D"],
      "correctAnswer": "選択肢A",
      "explanation": "詳しい解説..."
    },
    ... (合計10問分)
  ]
}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[*parts, "講義資料を詳細に分析し、全体要約、ページごとの講師風解説、重要ポイント、およびデフォルト10問のテスト問題を含むJSONを出力してください。"],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.3,
        )
    )
    return json.loads(response.text)

# Generate additional quizzes (+5 questions)
def generate_additional_quizzes(client: genai.Client, lecture_context: str, existing_quiz_count: int):
    system_instruction = f"""
あなたは大学のテスト作成担当教員です。
提供された講義コンテキストに基づき、理解度をさらに深めるための追加のテスト問題【5問】を作成してください。

【要件】：
- 選択問題 (multiple-choice)、穴埋め記述 (fill-in-the-blank)、手書き計算思考問題 (open-ended) をバランスよく含めること。
- 各問題のIDは q_{existing_quiz_count + 1} 〜 q_{existing_quiz_count + 5} とすること。

【JSON出力フォーマット】：
{{
  "quizzes": [
    {{
      "id": "q_new1",
      "type": "multiple-choice",
      "question": "追加問題1...",
      "choices": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
      "correctAnswer": "選択肢1",
      "explanation": "解説..."
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[f"【講義コンテキスト】:\n{lecture_context}\n\n追加のテスト問題を5問生成してください。"],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.5,
        )
    )
    return json.loads(response.text).get("quizzes", [])

# Grade user's handwritten / image / text answer using Gemini multimodal OCR
def grade_user_answer(client: genai.Client, question: str, correct_answer: str, user_answer_text: str = None, user_image: Image.Image = None):
    system_instruction = """
あなたは大学の厳格かつ丁寧な採点AI教員です。
学生から提出された解答（テキストまたは手書きノート/キャンバス画像）を精密に手書き認識（OCR）して採点してください。

【評価・認識の指針】：
1. 手書き画像が含まれている場合は、画像の文字・数式・図を正確に読み取り、認識結果（recognizedContent）を提示してください。
2. 計算過程や思考プロセス、数学的同値性（例: $1/2$ と $0.5$ や $\\frac{x}{2}$）を柔軟に評価します。
3. 判定（isCorrect: true/false）と、丁寧な解説・フィードバック（gradedFeedback）を作成してください。
4. 数式表現には必ず $...$ や $$...$$ のLaTeX記法を使用してください。

【出力JSON構造】:
{
  "recognizedContent": "手書き画像から読み取った数式・文字列...",
  "isCorrect": true,
  "gradedFeedback": "採点フィードバック詳細..."
}
"""

    contents = [
        f"【問題】: {question}\n【模範解答・基準】: {correct_answer}\n"
    ]
    if user_answer_text:
        contents.append(f"【学生のテキスト入力解答】: {user_answer_text}")
    if user_image:
        contents.append(user_image)
        contents.append("【学生の手書き画像解答】（添付画像）")

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.2,
        )
    )
    return json.loads(response.text)

# Chat response generator with Gemini
def generate_chat_response(client: genai.Client, history: list, new_message: str, lecture_context: str):
    system_instruction = f"""
あなたは講義資料に関する質問に答える親切で優秀なAI講義チューターです。
学生からの質問に対して、わかりやすく丁寧な日本語で回答してください。

【ルール】：
1. 講義資料の内容に基づき、正確かつ発展的に説明します。
2. 数式や公式は必ず標準的なLaTeX形式（$E=mc^2$ や $$\\int f(x)dx$$）で記述してください。
3. 分数は \\frac{{a}}{{b}} 、下付き文字は $x_1$ や $y_{{\\text{{max}}}}$ のように正しく表現してください。

【講義コンテキスト】：
{lecture_context}
"""

    formatted_contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        formatted_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=h["content"])]))
    
    formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=new_message)]))

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=formatted_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        )
    )
    return response.text


# ---------------- Streamlit UI Layout ----------------

client = get_gemini_client()

# Sidebar: Folders & Session History
with st.sidebar:
    st.title("🎓 講義アシスタント")
    
    if not client:
        st.error("⚠️ 環境変数 `GEMINI_API_KEY` が設定されていません。")
    
    st.divider()

    # --- Folder Management ---
    st.subheader("📁 フォルダ管理")
    
    new_folder_input = st.text_input("新しいフォルダを作成", placeholder="例: 認知科学", key="new_folder_input")
    if st.button("➕ フォルダ追加", use_container_width=True):
        if new_folder_input.strip() and new_folder_input.strip() not in st.session_state.folders:
            st.session_state.folders.append(new_folder_input.strip())
            st.success(f"フォルダ '{new_folder_input.strip()}' を追加しました！")
            st.rerun()

    st.divider()

    # --- Lecture Sessions organized by Folders ---
    st.subheader("📚 講義セッション一覧")
    
    if st.button("✨ 新しい講義を分析・登録", type="primary", use_container_width=True):
        st.session_state.active_session_id = None
        st.rerun()

    st.caption("フォルダ別に整理された講義一覧:")
    
    for folder in st.session_state.folders:
        folder_sessions = [s for s in st.session_state.sessions if s.get("folder", "指定なし") == folder]
        
        with st.expander(f"📁 {folder} ({len(folder_sessions)})", expanded=True):
            if not folder_sessions:
                st.caption("講義がありません")
            else:
                for sess in folder_sessions:
                    is_active = (sess['id'] == st.session_state.active_session_id)
                    btn_label = f"📖 {sess['title']}" if not is_active else f"▶ 📖 {sess['title']}"
                    if st.button(btn_label, key=f"sess_{sess['id']}", use_container_width=True):
                        st.session_state.active_session_id = sess['id']
                        st.rerun()


# Main View Header
st.markdown('<div class="main-header">🎓 AI 講義ノート & 理解度テストシステム</div>', unsafe_allow_html=True)

active_session = next((s for s in st.session_state.sessions if s['id'] == st.session_state.active_session_id), None)

# --- SCENARIO 1: No Active Session Selected -> Upload & New Session Creation ---
if not active_session:
    st.markdown('<div class="sub-header">講義スライド（画像/PDF）、音声データ、講義メモを入力してください。AIが講師風の解説、要約、デフォルト10問のテストを作成します。</div>', unsafe_allow_html=True)
    
    if not client:
        st.warning("⚠️ バックエンド環境に `GEMINI_API_KEY` を設定してください。")
        st.stop()

    col_f, col_m = st.columns([1, 2])
    with col_f:
        selected_folder = st.selectbox("保存先フォルダを選択", st.session_state.folders, index=0)
    
    input_method = st.radio("入力方法を選択", ["📁 ファイルアップロード (画像/PDF/音声)", "✍️ テキストを直接入力"], horizontal=True)
    
    uploaded_parts = []
    recorded_file_meta = [] # To record files in the session
    
    if "📁" in input_method:
        uploaded_files = st.file_uploader(
            "講義資料ファイル（複数選択可能）", 
            type=["png", "jpg", "jpeg", "webp", "pdf", "mp3", "wav", "m4a"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for u_file in uploaded_files:
                file_bytes = u_file.read()
                file_size_str = format_file_size(len(file_bytes))
                mime_type = u_file.type
                
                # Metadata record
                recorded_file_meta.append({
                    "id": str(uuid.uuid4()),
                    "name": u_file.name,
                    "type": mime_type,
                    "size": file_size_str,
                    "bytes": file_bytes
                })

                if "pdf" in mime_type:
                    pdf_text = extract_text_from_pdf(io.BytesIO(file_bytes))
                    uploaded_parts.append(pdf_text)
                    st.info(f"📄 {u_file.name}: PDFテキスト読み込み完了 ({len(pdf_text)} 文字)")
                elif "image" in mime_type:
                    img = Image.open(io.BytesIO(file_bytes))
                    uploaded_parts.append(img)
                    st.image(img, caption=f"🖼️ {u_file.name}", width=250)
                else:
                    uploaded_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
                    st.audio(file_bytes, format=mime_type)

    else:
        text_input = st.text_area("講義ノート・文字起こしテキスト入力", height=200, placeholder="ここに講義内容やメモを貼り付けてください...")
        if text_input.strip():
            uploaded_parts.append(text_input)
            recorded_file_meta.append({
                "id": str(uuid.uuid4()),
                "name": "手入力講義テキスト.txt",
                "type": "text/plain",
                "size": format_file_size(len(text_input.encode('utf-8'))),
                "bytes": text_input.encode('utf-8')
            })

    if st.button("🚀 講義を分析してノート＆10問テストを生成", type="primary", use_container_width=True):
        if not uploaded_parts:
            st.error("講義資料ファイルまたはテキストを入力してください。")
        else:
            with st.spinner("Gemini 3.6-flash が講義資料を詳細解析中...（10問テスト＆講師解説生成）"):
                try:
                    result = analyze_lecture_material(client, uploaded_parts)
                    new_id = f"session_{uuid.uuid4().hex[:8]}"
                    
                    new_session = {
                        "id": new_id,
                        "title": result.get("title", "無題の講義"),
                        "folder": selected_folder,
                        "summaryText": result.get("summaryText", ""),
                        "pageExplanations": result.get("pageExplanations", []),
                        "keyPoints": result.get("keyPoints", []),
                        "quizzes": result.get("quizzes", []),
                        "attempts": {},
                        "chatHistory": [],
                        "files": recorded_file_meta
                    }
                    
                    st.session_state.sessions.append(new_session)
                    st.session_state.active_session_id = new_id
                    st.success("講義セッションの作成が完了しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")


# --- SCENARIO 2: Active Session Detail View ---
else:
    # Header & Folder Changer & File Manager Expander
    st.markdown(f"### 📖 {active_session['title']}")
    
    col_meta1, col_meta2, col_meta3 = st.columns([2, 2, 2])
    with col_meta1:
        # Change Folder
        current_folder = active_session.get("folder", "指定なし")
        new_folder_sel = st.selectbox(
            "所属フォルダ", 
            st.session_state.folders, 
            index=st.session_state.folders.index(current_folder) if current_folder in st.session_state.folders else 0,
            key=f"change_folder_{active_session['id']}"
        )
        if new_folder_sel != current_folder:
            active_session["folder"] = new_folder_sel
            st.rerun()

    with col_meta2:
        st.caption(f"📁 フォルダ: **{active_session.get('folder', '指定なし')}**")
        st.caption(f"📄 記録ファイル数: **{len(active_session.get('files', []))} 件**")

    with col_meta3:
        if st.button("🗑️ セッション全体を削除", key="del_sess_btn"):
            st.session_state.sessions = [s for s in st.session_state.sessions if s['id'] != active_session['id']]
            st.session_state.active_session_id = None
            st.rerun()

    # --- Expander for Recorded Uploaded Files ---
    with st.expander("📁 登録済み講義ファイルの確認・追加・削除"):
        files = active_session.get("files", [])
        if not files:
            st.caption("登録されたファイルはありません。")
        else:
            for f in files:
                col_f1, col_f2, col_f3 = st.columns([3, 2, 1])
                with col_f1:
                    st.write(f"📄 **{f['name']}**")
                with col_f2:
                    st.caption(f"{f['type']} ({f['size']})")
                with col_f3:
                    if st.button("🗑️ 削除", key=f"del_file_{f['id']}"):
                        active_session["files"] = [file for file in active_session["files"] if file['id'] != f['id']]
                        st.success(f"ファイル {f['name']} を削除しました。")
                        st.rerun()

        st.divider()
        st.markdown("##### ➕ 追加ファイルをアップロード")
        add_file = st.file_uploader("追加資料（画像/PDF/音声）", type=["png", "jpg", "jpeg", "webp", "pdf", "mp3", "wav", "m4a"], key="add_file_input")
        if add_file:
            if st.button("追加ファイルを記録"):
                b_data = add_file.read()
                new_f_meta = {
                    "id": str(uuid.uuid4()),
                    "name": add_file.name,
                    "type": add_file.type,
                    "size": format_file_size(len(b_data)),
                    "bytes": b_data
                }
                active_session.setdefault("files", []).append(new_f_meta)
                st.success(f"追加ファイル {add_file.name} を記録しました！")
                st.rerun()

    st.divider()

    # Main Workspace Tabs
    tab1, tab2, tab3 = st.tabs(["📝 講義要約・講師解説", "🧪 理解度テスト (デフォルト10問)", "💬 AI講義チューター"])

    # --- TAB 1: Summary (Overall vs Page-by-Page Teacher Mode) ---
    with tab1:
        summary_mode = st.radio(
            "要約表示モードを選択", 
            ["🌐 資料全体の要約", "👨‍🏫 講師風・ページ別詳細解説"], 
            horizontal=True
        )

        st.divider()

        if summary_mode == "🌐 資料全体の要約":
            st.markdown("### 🌐 資料全体の構造化要約")
            st.markdown(active_session.get("summaryText", "要約データがありません。"))
            
            st.divider()
            st.markdown("### 💡 講義の重要ポイント")
            for idx, pt in enumerate(active_session.get("keyPoints", []), 1):
                st.markdown(f"**{idx}.** {pt}")

        else:
            st.markdown("### 👨‍🏫 講師風・ページ別詳細解説")
            page_exps = active_session.get("pageExplanations", [])
            
            if not page_exps:
                st.info("ページ別の解説データがありません。")
            else:
                for p in page_exps:
                    p_num = p.get("page", 1)
                    p_title = p.get("title", f"ページ {p_num}")
                    p_content = p.get("content", "")
                    
                    st.markdown(f"""
                    <div class="page-box">
                        <h4 style="margin-top:0; color:#3730a3;">📖 {p_title}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(p_content)
                    st.divider()

    # --- TAB 2: Quizzes (Default 10 Questions + Additional Generator + Canvas/Handwriting) ---
    with tab2:
        quizzes = active_session.get("quizzes", [])
        st.markdown(f"### 🧪 理解度テスト (全 {len(quizzes)} 問)")

        if not quizzes:
            st.info("テスト問題がありません。")
        else:
            for q_idx, q in enumerate(quizzes, 1):
                st.markdown(f"#### 問題 {q_idx} [{q['type']}]")
                st.markdown(q['question'])
                
                attempt = active_session['attempts'].get(q['id'])
                
                if attempt:
                    # Graded view
                    if attempt.get("isCorrect"):
                        st.success(f"⭕ **正解！** (あなたの解答: {attempt['userAnswer']})")
                    else:
                        st.error(f"❌ **不正解** (あなたの解答: {attempt['userAnswer']})")
                    
                    if attempt.get("recognizedContent"):
                        st.info(f"🔍 **AI手書き認識結果:**\n`{attempt['recognizedContent']}`")

                    st.markdown(f"**模範解答:** {q['correctAnswer']}")
                    if attempt.get("gradedFeedback"):
                        st.info(f"💡 **AI詳細フィードバック:**\n{attempt['gradedFeedback']}")
                    else:
                        st.markdown(f"**解説:** {q['explanation']}")
                    st.divider()

                else:
                    # Input / Handwriting Answer Form
                    with st.form(key=f"form_{q['id']}"):
                        user_ans_text = None
                        handwritten_image = None
                        
                        if q['type'] == 'multiple-choice':
                            user_ans_text = st.radio("選択肢を選んでください", q.get('choices', []), key=f"radio_{q['id']}")
                        
                        else:
                            input_type = st.radio("回答方法", ["📷 手書きノート/計算画像アップロード", "🎨 軽量手書きキャンバス", "⌨️ テキスト入力"], key=f"in_type_{q['id']}")
                            
                            if input_type == "📷 手書きノート/計算画像アップロード":
                                img_file = st.file_uploader("手書きノート・計算画像を添付", type=["png", "jpg", "jpeg"], key=f"img_{q['id']}")
                                if img_file:
                                    handwritten_image = Image.open(img_file)
                                    st.image(handwritten_image, caption="手書き添付画像プレビュー", width=280)

                            elif input_type == "🎨 軽量手書きキャンバス":
                                render_html5_canvas(q['id'])
                                img_file_c = st.file_uploader("キャンバス描画のスクショまたはノート画像（Geminiに送信）", type=["png", "jpg", "jpeg"], key=f"c_img_{q['id']}")
                                if img_file_c:
                                    handwritten_image = Image.open(img_file_c)
                                    st.image(handwritten_image, caption="送信画像プレビュー", width=280)

                            else:
                                user_ans_text = st.text_input("解答を入力（LaTeX数式例: $x^2 + y^2 = 1$ など）", key=f"text_{q['id']}")

                        submit_btn = st.form_submit_button("解答を提出して自動採点")

                        if submit_btn:
                            if not user_ans_text and handwritten_image is None:
                                st.warning("解答を入力するか、画像をアップロードしてください。")
                            else:
                                with st.spinner("Gemini 3.6-flash が手書き認識＆採点中..."):
                                    if q['type'] == 'multiple-choice':
                                        is_correct = (user_ans_text == q['correctAnswer'])
                                        active_session['attempts'][q['id']] = {
                                            "userAnswer": user_ans_text,
                                            "isCorrect": is_correct,
                                            "gradedFeedback": None
                                        }
                                    else:
                                        grade_res = grade_user_answer(
                                            client,
                                            question=q['question'],
                                            correct_answer=q['correctAnswer'],
                                            user_answer_text=user_ans_text,
                                            user_image=handwritten_image
                                        )
                                        active_session['attempts'][q['id']] = {
                                            "userAnswer": user_ans_text or "（手書きノート画像解答）",
                                            "recognizedContent": grade_res.get("recognizedContent", ""),
                                            "isCorrect": grade_res.get("isCorrect", False),
                                            "gradedFeedback": grade_res.get("gradedFeedback", "")
                                        }
                                    st.success("採点が完了しました！")
                                    st.rerun()

        st.divider()

        # Add Additional Questions Button
        st.markdown("#### ➕ テスト問題を追加作成する")
        st.caption("現在の講義内容をもとに、さらに追加で5問の小テストを生成します。")
        
        if st.button("🚀 追加問題（＋5問）を生成", key="add_quizzes_btn"):
            with st.spinner("Gemini 3.6-flash が追加の小テスト5問を作成中..."):
                try:
                    lecture_ctx = f"講義タイトル: {active_session['title']}\n要約: {active_session['summaryText']}\nポイント: {', '.join(active_session['keyPoints'])}"
                    new_q_list = generate_additional_quizzes(client, lecture_ctx, len(quizzes))
                    active_session["quizzes"].extend(new_q_list)
                    st.success(f"新たに {len(new_q_list)} 問のテスト問題を追加しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"追加問題作成エラー: {e}")

    # --- TAB 3: AI Lecture Tutor Chat ---
    with tab3:
        st.markdown("### 💬 AI講義チューター")
        st.caption("講義資料、数式、テスト問題の不明点について質問してください。")
        
        chat_history = active_session.get("chatHistory", [])
        
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if user_prompt := st.chat_input("講義内容についてAIチューターに質問..."):
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            chat_history.append({"role": "user", "content": user_prompt})
            lecture_ctx = f"講義: {active_session['title']}\n要約: {active_session['summaryText']}\nポイント: {', '.join(active_session['keyPoints'])}"
            
            with st.chat_message("assistant"):
                with st.spinner("AIチューター回答中..."):
                    try:
                        bot_reply = generate_chat_response(
                            client,
                            history=chat_history[:-1],
                            new_message=user_prompt,
                            lecture_context=lecture_ctx
                        )
                        st.markdown(bot_reply)
                        chat_history.append({"role": "model", "content": bot_reply})
                        active_session["chatHistory"] = chat_history
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

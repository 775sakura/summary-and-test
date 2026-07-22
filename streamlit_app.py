import os
import json
import io
import streamlit as st
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

# Initialize Gemini Client
def fetch_env_api_key():
    # 1. Streamlit Secretsから取得
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # 2. OS環境変数から取得
    return os.environ.get("GEMINI_API_KEY")

@st.cache_resource
def get_gemini_client(api_key: str = None):
    key = api_key or fetch_env_api_key()
    if not key:
        return None
    return genai.Client(api_key=key)

# Initialize session states
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# Analyze lecture material using Gemini
def analyze_lecture_material(client: genai.Client, parts: list):
    system_instruction = """
あなたは優秀な大学講師兼AI教育スペシャリストです。
アップロードされた講義資料（画像、PDFテキスト、音声など）を分析し、以下のJSON形式で構造化データを生成してください。

【出力要件】：
1. title: 講義資料から推測される適切なタイトル（日本語）
2. summaryText: 講義全体の要約（数式が含まれる場合はLaTeX形式 $...$ または $$...$$ を使用）
3. keyPoints: 重要ポイントの箇条書き（3〜6項目）
4. quizzes: 理解度テスト（3〜5問）。以下の形式を含む：
   - multiple-choice (4択問題)
   - fill-in-the-blank (穴埋め・記述問題)
   - open-ended (思考力・計算過程を問う記述・手書き用問題)

【JSONスキーマ】:
{
  "title": "講義タイトル",
  "summaryText": "講義要約...",
  "keyPoints": ["ポイント1", "ポイント2", "ポイント3"],
  "quizzes": [
    {
      "id": "q1",
      "type": "multiple-choice",
      "question": "問題文...",
      "choices": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
      "correctAnswer": "選択肢1",
      "explanation": "解説文..."
    },
    {
      "id": "q2",
      "type": "fill-in-the-blank",
      "question": "問題文...",
      "correctAnswer": "解答",
      "explanation": "解説文..."
    }
  ]
}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[*parts, "講義資料を分析して指定のJSON形式で出力してください。"],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.3,
        )
    )
    return json.loads(response.text)

# Grade handwritten / image / open-ended response using Gemini
def grade_user_answer(client: genai.Client, question: str, correct_answer: str, user_answer_text: str = None, user_image: Image.Image = None):
    system_instruction = """
あなたは大学の厳格かつ親切な採点AI採点官です。
学生の提出した解答（テキストまたは手書きノート画像）を採点してください。

【評価ルール】：
- 正解と本質的に一致しているか判定（計算過程の正しさ、数学的同値性、考え方の筋道を評価）
- 成績（isCorrect: true/false）
- 採点フィードバック（gradedFeedback）: どこが合っているか、どこで間違えたかをわかりやすく解説
- 理由・誤答分析を丁寧に記述してください。
- 数式は $...$ または $$...$$ のLaTeX形式を使用してください。

【出力JSONフォーマット】:
{
  "isCorrect": true,
  "gradedFeedback": "採点詳細フィードバック..."
}
"""

    contents = [
        f"【問題】: {question}\n【模範解答】: {correct_answer}\n"
    ]
    if user_answer_text:
        contents.append(f"【学生のテキスト解答】: {user_answer_text}")
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
あなたは講義資料に関する質問に答える親切で優秀なAIチューターです。
学生からの質問に、親しみやすく教育的な日本語で回答してください。

【ルール】：
1. 講義コンテキストに基づいた正確な説明を行ってください。
2. 数式や公式は必ずLaTeX形式（インライン: $E=mc^2$, ディスプレイ: $$\\int f(x)dx$$）を使用してください。
3. 分数には \\frac{{a}}{{b}} を使い、下付き文字は $x_1$ などの標準LaTeX記法を守ってください。

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

# ---------------- Streamlit UI ----------------

# Sidebar API Key & Session List
with st.sidebar:
    st.title("🎓 講義アシスタント")
    
    api_key_input = st.text_input("Gemini API Key", type="password", help="GEMINI_API_KEY環境変数がない場合は入力してください")
    client = get_gemini_client(api_key_input)
    
    st.divider()
    st.subheader("📚 講義セッション履歴")
    
    if st.button("➕ 新しい講義を分析", use_container_width=True):
        st.session_state.active_session_id = None
        st.rerun()

    for sess in st.session_state.sessions:
        if st.button(f"📖 {sess['title']}", key=sess['id'], use_container_width=True):
            st.session_state.active_session_id = sess['id']
            st.rerun()

# Main View
st.markdown('<div class="main-header">🎓 AI 講義ノート & 理解度テスト</div>', unsafe_allow_html=True)

# If no active session, show Upload View
active_session = next((s for s in st.session_state.sessions if s['id'] == st.session_state.active_session_id), None)

if not active_session:
    st.markdown('<div class="sub-header">講義スライド、レジュメ画像、PDF、録音音声、または講義ノートテキストをアップロードして分析を開始します。</div>', unsafe_allow_html=True)
    
    if not client:
        st.warning("⚠️ APIキーが見つかりません。サイドバーに GEMINI_API_KEY を入力するか環境変数を設定してください。")
        st.stop()

    input_method = st.radio("入力形式を選択", ["ファイルアップロード (画像/PDF/音声)", "テキストを直接入力"], horizontal=True)
    
    uploaded_parts = []
    
    if input_method == "ファイルアップロード (画像/PDF/音声)":
        uploaded_file = st.file_uploader("講義資料を選択", type=["png", "jpg", "jpeg", "webp", "pdf", "mp3", "wav", "m4a"])
        if uploaded_file:
            file_bytes = uploaded_file.read()
            mime_type = uploaded_file.type
            
            if "pdf" in mime_type:
                pdf_text = extract_text_from_pdf(io.BytesIO(file_bytes))
                uploaded_parts.append(pdf_text)
                st.info(f"PDFテキストの抽出に成功しました（{len(pdf_text)} 文字）")
            elif "image" in mime_type:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption="アップロードされた講義画像", use_container_width=True)
                uploaded_parts.append(image)
            else:
                # Audio / other binary
                uploaded_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
                st.audio(file_bytes, format=mime_type)

    else:
        text_input = st.text_area("講義ノート・文字起こしテキストを入力", height=200, placeholder="ここに講義のメモやテキストを貼り付けてください...")
        if text_input.strip():
            uploaded_parts.append(text_input)

    if st.button("🚀 講義ノートとテストを自動生成", type="primary", use_container_width=True):
        if not uploaded_parts:
            st.error("講義資料またはテキストを入力してください。")
        else:
            with st.spinner("Gemini AI が講義資料を解析して要約と小テストを作成中..."):
                try:
                    result = analyze_lecture_material(client, uploaded_parts)
                    new_id = f"session_{len(st.session_state.sessions) + 1}"
                    new_session = {
                        "id": new_id,
                        "title": result.get("title", "無題の講義"),
                        "summaryText": result.get("summaryText", ""),
                        "keyPoints": result.get("keyPoints", []),
                        "quizzes": result.get("quizzes", []),
                        "attempts": {},
                        "chatHistory": []
                    }
                    st.session_state.sessions.append(new_session)
                    st.session_state.active_session_id = new_id
                    st.success("分析が完了しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"解析エラーが発生しました: {e}")

else:
    # Active Session Header
    st.subheader(f"📖 {active_session['title']}")
    
    tab1, tab2, tab3 = st.tabs(["📝 講義要約・ポイント", "🧪 理解度テスト", "💬 AIチューターと会話"])

    # --- TAB 1: Summary ---
    with tab1:
        st.markdown("### 📌 講義の要約")
        st.markdown(active_session['summaryText'])
        
        st.divider()
        st.markdown("### 💡 重要ポイント")
        for idx, point in enumerate(active_session['keyPoints'], 1):
            st.markdown(f"**{idx}.** {point}")

    # --- TAB 2: Quizzes ---
    with tab2:
        st.markdown("### 🧪 理解度テスト")
        
        quizzes = active_session['quizzes']
        if not quizzes:
            st.info("テスト問題が生成されていません。")
        else:
            for q_idx, q in enumerate(quizzes, 1):
                st.markdown(f"#### 問題 {q_idx} [{q['type']}]")
                st.markdown(q['question'])
                
                attempt = active_session['attempts'].get(q['id'])
                
                if attempt:
                    # Already answered
                    if attempt.get("isCorrect"):
                        st.success(f"⭕ 正解！ (あなたの回答: {attempt['userAnswer']})")
                    else:
                        st.error(f"❌ 不正解 (あなたの回答: {attempt['userAnswer']})")
                    
                    st.markdown(f"**模範解答:** {q['correctAnswer']}")
                    if attempt.get("gradedFeedback"):
                        st.info(f"**AIフィードバック:**\n{attempt['gradedFeedback']}")
                    else:
                        st.markdown(f"**解説:** {q['explanation']}")
                    st.divider()
                else:
                    # Answering form
                    with st.form(key=f"form_{q['id']}"):
                        user_ans_text = None
                        uploaded_handwriting = None
                        
                        if q['type'] == 'multiple-choice':
                            user_ans_text = st.radio("選択肢を選んでください", q.get('choices', []), key=f"radio_{q['id']}")
                        else:
                            answer_method = st.radio("解答方法", ["テキスト入力", "手書き/ノート画像アップロード"], key=f"method_{q['id']}")
                            if answer_method == "テキスト入力":
                                user_ans_text = st.text_input("解答を入力（数式や数値など）", key=f"text_{q['id']}")
                            else:
                                img_file = st.file_uploader("手書きノート画像をアップロード", type=["png", "jpg", "jpeg"], key=f"img_{q['id']}")
                                if img_file:
                                    uploaded_handwriting = Image.open(img_file)
                                    st.image(uploaded_handwriting, caption="提出画像プレビュー", width=300)
                        
                        submit_btn = st.form_submit_button("解答を提出")
                        
                        if submit_btn:
                            if not user_ans_text and not uploaded_handwriting:
                                st.warning("解答を入力または画像を添付してください。")
                            else:
                                with st.spinner("採点中..."):
                                    if q['type'] == 'multiple-choice':
                                        is_correct = (user_ans_text == q['correctAnswer'])
                                        active_session['attempts'][q['id']] = {
                                            "userAnswer": user_ans_text,
                                            "isCorrect": is_correct,
                                            "gradedFeedback": None
                                        }
                                    else:
                                        # Use Gemini AI grading for written/image answers
                                        grade_res = grade_user_answer(
                                            client,
                                            question=q['question'],
                                            correct_answer=q['correctAnswer'],
                                            user_answer_text=user_ans_text,
                                            user_image=uploaded_handwriting
                                        )
                                        active_session['attempts'][q['id']] = {
                                            "userAnswer": user_ans_text or "（画像解答）",
                                            "isCorrect": grade_res.get("isCorrect", False),
                                            "gradedFeedback": grade_res.get("gradedFeedback", "")
                                        }
                                    st.success("採点が完了しました！")
                                    st.rerun()

    # --- TAB 3: AI Chat ---
    with tab3:
        st.markdown("### 💬 AI講義メンター")
        st.caption("講義内容の疑問、公式の証明、補足問題などをAIに質問できます。")
        
        chat_history = active_session.get("chatHistory", [])
        
        # Display past messages
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # User Chat Input
        if user_prompt := st.chat_input("講義内容について質問する..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            chat_history.append({"role": "user", "content": user_prompt})
            
            # Prepare context
            lecture_ctx = f"講義: {active_session['title']}\n要約: {active_session['summaryText']}\nポイント: {', '.join(active_session['keyPoints'])}"
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
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
                        st.error(f"応答生成中にエラーが発生しました: {e}")
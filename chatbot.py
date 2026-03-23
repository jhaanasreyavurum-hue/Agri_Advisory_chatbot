import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from gtts import gTTS
import speech_recognition as sr
from deep_translator import GoogleTranslator
import io
from predict import load_models, predict_image

st.set_page_config(
    page_title="Maize Advisory Chatbot",
    page_icon="🌽",
    layout="wide"
)  

if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "recognized_text" not in st.session_state:
    st.session_state["recognized_text"] = ""
if "detailed_answer" not in st.session_state:
    st.session_state["detailed_answer"] = ""
left_col, right_col = st.columns([1.1, 1.7], gap="large")

with left_col:
    #st.image("images/maize_side.jpg", use_container_width=True)
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            border-radius: 14px;
            min-height: 640px;
            object-fit: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with right_col:
    st.markdown("## 🌽 Maize Advisory Chatbot")
    st.write("Ask anything about maize crops, diseases, pests, or farming practices.")

    language = st.radio(
        "Select Language",
        ["English", "Hindi"],
        key="language_selector",
        horizontal=True
    )

def speak_text(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer.read(), format="audio/mp3")
    except Exception as e:
        st.warning(f"Voice output failed: {e}")

if language == "Hindi":
    common_questions_label   = "सामान्य प्रश्न"
    ask_question_label       = "अपना प्रश्न पूछें"
    answer_label             = "उत्तर"
    speak_answer_label       = "🔊 उत्तर सुनें"
    clear_label              = "साफ करें"
    recognized_question_label = "पहचाना गया प्रश्न"
    upload_label             = "🌿 पत्ती की फोटो अपलोड करें"
    result_label             = "पहचान परिणाम"
else:
    common_questions_label   = "Common Questions"
    ask_question_label       = "Ask your question"
    answer_label             = "Answer"
    speak_answer_label       = "🔊 Speak Answer"
    clear_label              = "Clear"
    recognized_question_label = "Recognized Question"
    upload_label             = "🌿 Upload Maize Leaf Image"
    result_label             = "Detection Result"


@st.cache_resource
def get_models():
    return load_models()

maize_model, disease_model, pest_model = get_models()

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.strip()
    return df

data = load_data()
documents = data["English_Language"].fillna("").tolist()

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sent_model = load_sentence_model()

@st.cache_data
def get_doc_embeddings(docs):
    return sent_model.encode(docs)

doc_embeddings = get_doc_embeddings(documents)

label_display = {
    "blight":         "Blight",
    "grey_leaf_spot": "Grey Leaf Spot",
    "healthy":        "Healthy",
    "rust":           "Rust",
    "aphids":         "Aphids",
    "corn_rootworm":  "Corn Rootworm",
    "fall_army_worm": "Fall Army Worm",
    "stalk_borer":    "Stalk Borer",
}

# ── IMAGE UPLOAD ──────────────────────────────────────
with right_col:
    st.subheader(upload_label)
    uploaded_file = st.file_uploader("Choose a maize leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        result, confidence, category = predict_image(
            image, maize_model, disease_model, pest_model)

    st.subheader(result_label)

    if result == "not_maize":
        st.error("This does not appear to be a maize plant. Please upload a maize leaf image.")
    else:
        display_name = label_display.get(result, result)

        if result == "healthy":
            st.success(f"Result: {display_name} ({category.title()}) — Confidence: {confidence}%")
        else:
            st.warning(f"Result: {display_name} ({category.title()}) — Confidence: {confidence}%")

        if language == "Hindi":
            auto_query = f"मक्का में {display_name} क्या है और इसे कैसे नियंत्रित करें?"
        else:
            auto_query = f"How to control {display_name} in maize?"

        st.info(f"Searching advice for: {auto_query}")
        st.session_state["query"] = auto_query
        st.session_state["recognized_text"] = auto_query

# ── COMMON QUESTIONS ──────────────────────────────────
if language == "Hindi":
    q1 = "मक्का में फॉल आर्मीवर्म को कैसे नियंत्रित करें?"
    q2 = "मक्का की पत्तियाँ पीली क्यों हो रही हैं?"
    q3 = "मक्का की उपज कैसे बढ़ाएँ?"
else:
    q1 = "How to control fall armyworm in maize?"
    q2 = "Why are maize leaves turning yellow?"
    q3 = "How to increase maize yield?"
with right_col:
    st.subheader(common_questions_label)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(q1, key="common_q1_right"):
            st.session_state["query"] = q1
            st.session_state["recognized_text"] = q1
    with col2:
        if st.button(q2, key="common_q2_right"):
            st.session_state["query"] = q2
            st.session_state["recognized_text"] = q2
    with col3:
        if st.button(q3, key="common_q3_right"):
            st.session_state["query"] = q3
            st.session_state["recognized_text"] = q3
with right_col:
    st.subheader(ask_question_label)

    user_input = st.text_input(
        ask_question_label,
        value=st.session_state.get("recognized_text", ""),
        key="main_question_input"
    )
user_input = st.session_state.get("main_question_input", "")

if user_input:
    st.session_state["query"] = user_input.strip()
    st.session_state["recognized_text"] = user_input.strip()

# ── VOICE INPUT ───────────────────────────────────────
with right_col:
    st.subheader("🎤 Speak Your Question")
    audio_value = st.audio_input("Record your question")

if audio_value is not None:
    try:
        recognizer = sr.Recognizer()
        audio_bytes = audio_value.read()
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        if language == "Hindi":
            spoken_text = recognizer.recognize_google(audio_data, language="hi-IN")
        else:
            spoken_text = recognizer.recognize_google(audio_data, language="en-IN")
        st.session_state["query"] = spoken_text
        st.session_state["recognized_text"] = spoken_text
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")

with right_col:
    if st.session_state["recognized_text"]:
        st.subheader(recognized_question_label)
        st.write(st.session_state["recognized_text"])

# ── ANSWER FROM DATASET ───────────────────────────────
if st.session_state["query"]:
    query = st.session_state["query"].strip()

    if language == "Hindi":
        try:
            query_for_search = GoogleTranslator(source="auto", target="en").translate(query)
        except Exception:
            query_for_search = query
    else:
        query_for_search = query

    query_embedding = sent_model.encode([query_for_search])
    similarity = cosine_similarity(query_embedding, doc_embeddings)
    best_index = similarity.argmax()
    best_score = similarity[0][best_index]

    if best_score < 0.45:
        st.session_state["answer"] = (
            "यह प्रश्न डेटासेट में उपलब्ध नहीं है।"
            if language == "Hindi"
            else "This question is not available in the dataset."
        )
    else:
        if language == "Hindi":
            st.session_state["answer"] = str(data.iloc[best_index]["Hindi_Language_Answers"])
        else:
            st.session_state["answer"] = str(data.iloc[best_index]["Answers_from_ChatGpt_in_english"])
with right_col:
    st.subheader(answer_label)
    st.write(st.session_state["answer"])

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(speak_answer_label, key="speak_answer_btn"):
            speak_text(st.session_state["answer"],
                       lang_code="hi" if language == "Hindi" else "en")
    with col_b:
        if st.button(clear_label, key="clear_btn"):
            st.session_state["query"] = ""
            st.session_state["recognized_text"] = ""
            st.session_state["answer"] = ""
            st.session_state["detailed_answer"] = ""
            st.rerun()

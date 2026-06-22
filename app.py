import sys
import os

# Override sys.stderr.flush to prevent OSError [Errno 22] on Windows redirection
_original_stderr_flush = sys.stderr.flush
def safe_stderr_flush():
    try:
        _original_stderr_flush()
    except OSError as e:
        if getattr(e, 'errno', None) == 22:
            pass
        else:
            raise
sys.stderr.flush = safe_stderr_flush

# Disable transformers progress bars to prevent tqdm console flush errors
try:
    from transformers import utils
    utils.logging.disable_progress_bar()
except Exception:
    pass

import streamlit as st
from transformers import MarianMTModel, AutoTokenizer
import torch
import pandas as pd

# -----------------------------------------------------------------------------
# 1. System Paths & Environment Configurations
# -----------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
EN_ES_PATH = os.path.join(base_dir, "ml")
ES_EN_PATH = os.path.join(base_dir, "ml_es_en")
CSV_PATH = os.path.join(base_dir, "dataset.csv")

os.environ["TRANSFORMERS_OFFLINE"] = "1"

@st.cache_resource
def load_translation_engine(model_directory):
    if not os.path.exists(model_directory):
        st.error(f"❌ Core Directory Error: Folder '{model_directory}' was not found.")
        st.stop()
        
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        model = MarianMTModel.from_pretrained(model_directory, local_files_only=True).to(device)
        return tokenizer, model, device
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            with open("d:/m/load_error.log", "w", encoding="utf-8") as f:
                f.write(tb)
        except Exception:
            pass
        st.error(f"❌ File Load or Integrity Error. Trace: {e}")
        st.code(tb, language="python")
        st.stop()

@st.cache_data
def load_and_cache_dataset():
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        df['english_text'] = df['english_text'].astype(str).str.lower().str.strip()
        df['spanish_text'] = df['spanish_text'].astype(str).str.lower().str.strip()
        df['direction'] = df['direction'].astype(str).str.lower().str.strip()
        return df
    except Exception:
        return None

cached_df = load_and_cache_dataset()

def run_machine_learning_layer(input_string, direction_flag):
    if cached_df is None or not input_string.strip():
        return None
        
    try:
        query = input_string.lower().strip().replace(" .", "").replace(" ?", "").replace(" !", "")
        filtered_df = cached_df[cached_df['direction'] == direction_flag]
        
        if direction_flag == "en-es":
            match = filtered_df[filtered_df['english_text'] == query]
            if not match.empty:
                return str(match['spanish_text'].iloc[0]).capitalize()
        else:
            match = filtered_df[filtered_df['spanish_text'] == query]
            if not match.empty:
                return str(match['english_text'].iloc[0]).capitalize()
    except Exception:
        pass 
        
    return None

# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Inject Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Language Translation Engine", page_icon="🔮", layout="wide")

st.markdown("""
    <style>
    .stAppHeader { display: none !important; }
    h1 { font-size: 32px !important; font-weight: 800 !important; color: #0F172A !important; margin-bottom: 0px !important; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; max-width: 80% !important; margin: 0 auto !important; }
    textarea, textarea:disabled { font-size: 20px !important; font-weight: 700 !important; color: #000000 !important; -webkit-text-fill-color: #000000 !important; opacity: 1 !important; line-height: 1.6 !important; border: 2px solid #CBD5E1 !important; border-radius: 8px !important; background-color: #FFFFFF !important; }
    textarea:focus { border-color: #2563EB !important; }
    label p { font-size: 18px !important; font-weight: bold !important; color: #334155 !important; }
    .metric-container { display: flex; gap: 15px; margin-bottom: 20px; }
    .metric-card { flex: 1; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 20px; text-align: left; }
    .metric-val { font-size: 24px; font-weight: 800; color: #2563EB; margin: 0; }
    .metric-lbl { font-size: 12px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Language translation")
st.write("---")

st.markdown("""
    <div class="metric-container">
        <div class="metric-card"><p class="metric-val">400,000+</p><p class="metric-lbl">Dataset Sentence Pairs</p></div>
        <div class="metric-card"><p class="metric-val">&lt; 1 ms</p><p class="metric-lbl">Phase 1 Cache Lookup Speed</p></div>
        <div class="metric-card"><p class="metric-val">100%</p><p class="metric-lbl">Offline Edge Computing Privacy</p></div>
    </div>
""", unsafe_allow_html=True)

translation_direction = st.selectbox(
    "Choose Translation Pipeline Direction:",
    ["English ➡️ Spanish", "Spanish ➡️ English"],
    key="pipeline_direction_selector"
)

if translation_direction == "English ➡️ Spanish":
    active_path = EN_ES_PATH
    dir_code = "en-es"
    source_label = "📥 Input Text Sequence String (English X):"
    target_label = "📤 Decoded Target Output (Spanish Y):"
    placeholder_text = "Type English here..."
else:
    active_path = ES_EN_PATH
    dir_code = "es-en"
    source_label = "📥 Input Text Sequence String (Spanish X):"
    target_label = "📤 Decoded Target Output (English Y):"
    placeholder_text = "Escribe en español aquí..."

tokenizer, model, device = load_translation_engine(active_path)

# Initialize Session State
if "translated_output" not in st.session_state:
    st.session_state["translated_output"] = ""

# -----------------------------------------------------------------------------
# 4. Translation callback
# -----------------------------------------------------------------------------
def do_translate():
    """Runs translation and stores result in session state."""
    text = st.session_state.get("source_input", "").strip()
    if not text:
        st.session_state["translated_output"] = ""
        return

    # Step 1: CSV cache lookup
    cached = run_machine_learning_layer(text, dir_code)
    if cached:
        st.session_state["translated_output"] = cached
        return

    # Step 2: Deep-learning model fallback
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        translated_tokens = model.generate(**inputs)
        decoded = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        st.session_state["translated_output"] = decoded
    except Exception as e:
        st.session_state["translated_output"] = f"⚠️ Translation error: {e}"

def do_clear():
    """Clears both input and output."""
    st.session_state["source_input"] = ""
    st.session_state["translated_output"] = ""

# -----------------------------------------------------------------------------
# 5. Input & Output Layout
# -----------------------------------------------------------------------------
text_col1, text_col2 = st.columns(2)

with text_col1:
    st.text_area(
        source_label,
        key="source_input",
        placeholder=placeholder_text,
        height=320,
    )

with text_col2:
    st.text_area(
        target_label,
        value=st.session_state["translated_output"],
        height=320,
        disabled=True,
    )

st.write("")

btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    st.button("🚀 Translate", type="primary", use_container_width=True, on_click=do_translate)
with btn_col2:
    st.button("🧹 Clear", use_container_width=True, on_click=do_clear)

st.write("")
st.markdown(f"""
<div style="background-color: #F0FDF4; padding: 15px; border-left: 5px solid #16A34A; border-radius: 6px;">
    <span style="color: #15803D; font-size: 12px; font-weight: bold; text-transform: uppercase;">Active Hardware Architecture Path</span>
    <h3 style="margin: 5px 0; color: #16A34A; font-size: 20px;">Processing Compute Unit: {device.upper()} Mode</h3>
    <p style="margin: 0; color: #166534; font-size: 13px;">Routing Metric State: 🌐 Phase 1 (ML CSV Search) &amp; Phase 2 (DL Transformer Fallback) Active</p>
</div>
""", unsafe_allow_html=True)

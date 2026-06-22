# 🔮 Offline Language Translation Engine

A high-performance, privacy-focused offline language translation web application built using **Streamlit** and Hugging Face's **MarianMT** translation models.

## 🚀 Features
- **Phase 1: Ultra-fast Cache Lookup (<1ms)** utilizing a local dataset (`dataset.csv`) containing over 400,000+ cached sentence pairs.
- **Phase 2: Offline Deep Learning Fallback** using locally hosted MarianMT transformer models for high-quality English-to-Spanish and Spanish-to-English translation.
- **100% Privacy & Offline Edge Computing** - No external API calls are made, keeping all input texts completely private.
- **Modern User Interface** with side-by-side translation layout, metrics cards, and clear utilities.

## 🛠️ Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`.

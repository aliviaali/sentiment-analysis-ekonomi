import streamlit as st
import pickle
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.graph_objects as go
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Berita CNBC",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Sastrawi
@st.cache_resource
def load_sastrawi():
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    
    factory_stopword = StopWordRemoverFactory()
    stopwords = factory_stopword.get_stop_words()
    custom_stopwords = ['segini', 'wah', 'catat', 'ini', 'itu']
    stopwords.extend(custom_stopwords)
    
    return stemmer, stopwords

stemmer, stopwords = load_sastrawi()

# Load models
@st.cache_resource
def load_models():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('naive_bayes_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        return vectorizer, nb_model, svm_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Pastikan folder 'models' berisi file model yang diperlukan")
        return None, None, None

vectorizer, nb_model, svm_model = load_models()

# Preprocessing function
def preprocess_text(text):
    """Preprocessing teks berita"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+|@\w+', '', text)
    text = re.sub(r'rp\s*[\d.,]+\s*[tmb]?', 'rupiah', text)
    text = re.sub(r'us\$[\d.,]+', 'dolar', text)
    text = re.sub(r'\d+[.,]?\d*%', 'persen', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [word for word in words if word not in stopwords and len(word) > 2]
    text = ' '.join(words)
    text = stemmer.stem(text)
    
    return text

# Prediction function
def predict_sentiment(text, model_type='svm'):
    """Prediksi sentimen dari teks"""
    if vectorizer is None or nb_model is None or svm_model is None:
        return None, None
    
    cleaned_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    
    if model_type == 'naive_bayes':
        prediction = nb_model.predict(text_tfidf)[0]
        proba = nb_model.predict_proba(text_tfidf)[0]
        labels = nb_model.classes_
    else:
        prediction = svm_model.predict(text_tfidf)[0]
        proba = None
        labels = None
    
    return prediction, proba, labels

# Sentiment color mapping
def get_sentiment_color(sentiment):
    colors = {
        'positif': '#4CAF50',
        'netral': '#2196F3',
        'negatif': '#F44336'
    }
    return colors.get(sentiment, '#757575')

def get_sentiment_emoji(sentiment):
    emojis = {
        'positif': '😊',
        'netral': '😐',
        'negatif': '😞'
    }
    return emojis.get(sentiment, '❓')

# Header
st.markdown('<p class="main-header">📰 Analisis Sentimen Berita</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Klasifikasi Sentimen Berita CNBC Indonesia menggunakan Naive Bayes dan SVM</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=100)
    st.title("⚙️ Pengaturan")
    
    model_choice = st.radio(
        "Pilih Model:",
        ["SVM (Support Vector Machine)", "Naive Bayes"],
        help="Pilih algoritma machine learning untuk prediksi sentimen"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Tentang Model")
    
    if model_choice == "SVM (Support Vector Machine)":
        st.info("""
        **SVM (Linear)**
        - Akurasi tinggi (>85%)
        - Robust untuk text classification
        - Optimal untuk dataset ini
        """)
    else:
        st.info("""
        **Naive Bayes**
        - Cepat dan efisien
        - Probabilistik
        - Cocok untuk baseline
        """)
    
    st.markdown("---")
    
    st.markdown("### 📈 Metrik Performa")
    st.metric("Dataset", "9,819 berita")
    st.metric("Akurasi Target", "> 85%")
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    Dibuat untuk analisis sentimen berita ekonomi dan bisnis Indonesia.
    
    **Tech Stack:**
    - Python 3.8+
    - Scikit-learn
    - Sastrawi
    - Streamlit
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Tunggal", "📊 Prediksi Batch", "ℹ️ Informasi"])

# Tab 1: Single Prediction
with tab1:
    st.header("Analisis Sentimen Teks Tunggal")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Masukkan teks berita:",
            height=150,
            placeholder="Contoh: Saham BCA melonjak 5% dan mencetak rekor tertinggi sepanjang masa...",
            help="Masukkan judul atau isi berita yang ingin dianalisis"
        )
        
        analyze_button = st.button("🔍 Analisis Sentimen", type="primary")
    
    with col2:
        st.markdown("### 💡 Contoh Berita")
        example_texts = {
            "Positif 📈": "Saham BCA melonjak 5% dan mencetak rekor tertinggi!",
            "Negatif 📉": "Ekonomi Indonesia mengalami perlambatan yang mengkhawatirkan",
            "Netral ➡️": "Bank Indonesia memutuskan mempertahankan suku bunga di 6%"
        }
        
        for label, text in example_texts.items():
            if st.button(label, key=label):
                user_input = text
                analyze_button = True
    
    if analyze_button and user_input:
        with st.spinner("🔄 Memproses..."):
            model_type = 'svm' if 'SVM' in model_choice else 'naive_bayes'
            prediction, proba, labels = predict_sentiment(user_input, model_type)
            
            if prediction:
                st.success("✅ Analisis selesai!")
                
                # Display result
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    sentiment_color = get_sentiment_color(prediction)
                    sentiment_emoji = get_sentiment_emoji(prediction)
                    
                    st.markdown(f"""
                    <div style='background-color: {sentiment_color}; padding: 30px; border-radius: 15px; text-align: center;'>
                        <h1 style='color: white; margin: 0;'>{sentiment_emoji}</h1>
                        <h2 style='color: white; margin: 10px 0;'>{prediction.upper()}</h2>
                        <p style='color: white; margin: 0; opacity: 0.9;'>Sentimen Terdeteksi</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Show probability if available (Naive Bayes)
                if proba is not None and labels is not None:
                    st.subheader("📊 Distribusi Probabilitas")
                    
                    prob_df = pd.DataFrame({
                        'Sentimen': [l.capitalize() for l in labels],
                        'Probabilitas': proba * 100
                    }).sort_values('Probabilitas', ascending=False)
                    
                    fig = px.bar(
                        prob_df, 
                        x='Sentimen', 
                        y='Probabilitas',
                        color='Sentimen',
                        color_discrete_map={
                            'Positif': '#4CAF50',
                            'Netral': '#2196F3',
                            'Negatif': '#F44336'
                        },
                        text='Probabilitas'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        yaxis_title="Probabilitas (%)",
                        xaxis_title=""
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Preprocessing info
                with st.expander("🔧 Detail Preprocessing"):
                    cleaned = preprocess_text(user_input)
                    st.markdown(f"**Teks Original:**")
                    st.text(user_input)
                    st.markdown(f"**Teks Setelah Preprocessing:**")
                    st.text(cleaned)
                    st.caption(f"Jumlah kata: {len(user_input.split())} → {len(cleaned.split())}")

# Tab 2: Batch Prediction
with tab2:
    st.header("Analisis Sentimen Batch")
    
    st.info("💡 Upload file CSV dengan kolom 'judul' untuk analisis batch")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=['csv'],
        help="File harus memiliki kolom 'judul' yang berisi teks berita"
    )
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            if 'judul' not in df_batch.columns:
                st.error("❌ File harus memiliki kolom 'judul'")
            else:
                st.success(f"✅ File berhasil diupload! Total data: {len(df_batch)}")
                
                # Preview data
                with st.expander("👀 Preview Data"):
                    st.dataframe(df_batch.head(10))
                
                if st.button("🚀 Mulai Analisis Batch", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    model_type = 'svm' if 'SVM' in model_choice else 'naive_bayes'
                    predictions = []
                    
                    for idx, text in enumerate(df_batch['judul']):
                        prediction, _, _ = predict_sentiment(text, model_type)
                        predictions.append(prediction)
                        
                        # Update progress
                        progress = (idx + 1) / len(df_batch)
                        progress_bar.progress(progress)
                        status_text.text(f"Memproses: {idx + 1}/{len(df_batch)}")
                    
                    df_batch['sentimen_prediksi'] = predictions
                    
                    status_text.text("✅ Analisis selesai!")
                    progress_bar.empty()
                    
                    # Results
                    st.subheader("📊 Hasil Analisis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_counts = df_batch['sentimen_prediksi'].value_counts()
                    
                    with col1:
                        st.metric(
                            "😊 Positif",
                            sentiment_counts.get('positif', 0),
                            f"{sentiment_counts.get('positif', 0) / len(df_batch) * 100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "😐 Netral",
                            sentiment_counts.get('netral', 0),
                            f"{sentiment_counts.get('netral', 0) / len(df_batch) * 100:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "😞 Negatif",
                            sentiment_counts.get('negatif', 0),
                            f"{sentiment_counts.get('negatif', 0) / len(df_batch) * 100:.1f}%"
                        )
                    
                    # Visualization
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Distribusi Sentimen",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positif': '#4CAF50',
                            'netral': '#2196F3',
                            'negatif': '#F44336'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results table
                    st.subheader("📋 Tabel Hasil")
                    st.dataframe(
                        df_batch[['judul', 'sentimen_prediksi']],
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = df_batch.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Hasil (CSV)",
                        data=csv,
                        file_name="hasil_analisis_sentimen.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Tab 3: Information
with tab3:
    st.header("ℹ️ Informasi Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini menggunakan **Machine Learning** untuk mengklasifikasikan sentimen berita 
        ekonomi dan bisnis dari CNBC Indonesia ke dalam tiga kategori:
        
        - **Positif** 😊: Berita dengan sentimen positif
        - **Netral** 😐: Berita dengan sentimen netral
        - **Negatif** 😞: Berita dengan sentimen negatif
        
        ### 🎯 Metode yang Digunakan
        
        1. **Naive Bayes (Multinomial)**
           - Algoritma probabilistik
           - Cepat dan efisien
           - Cocok untuk text classification
        
        2. **SVM (Support Vector Machine)**
           - Linear kernel
           - Akurasi tinggi (>85%)
           - Optimal untuk dataset ini
        
        ### 🔧 Preprocessing
        
        - **Tokenization**: Memecah teks menjadi kata-kata
        - **Stopwords Removal**: Menghapus kata-kata umum
        - **Stemming**: Mengubah kata ke bentuk dasar (Sastrawi)
        - **TF-IDF**: Feature extraction dengan n-gram (1-3)
        """)
    
    with col2:
        st.subheader("📊 Dataset")
        st.markdown("""
        - **Sumber**: Dataset CNBC Indonesia
        - **Total Data**: 9,819 berita
        - **Periode**: 2024
        - **Distribusi**:
          - Netral: 4,356 (44.4%)
          - Positif: 2,887 (29.4%)
          - Negatif: 2,576 (26.2%)
        
        ### 🎓 Performa Model
        
        | Model | Accuracy | Precision | Recall | F1-Score |
        |-------|----------|-----------|--------|----------|
        | Naive Bayes | ~82% | ~81% | ~82% | ~81% |
        | **SVM** | **>85%** | **>84%** | **>85%** | **>84%** |
        
        ### 🚀 Teknologi
        
        - **Python 3.8+**
        - **Scikit-learn**: Machine learning framework
        - **Sastrawi**: Indonesian NLP library
        - **Streamlit**: Web framework
        - **Plotly**: Interactive visualization
        
        ### 📝 Cara Penggunaan
        
        1. Pilih model di sidebar (SVM atau Naive Bayes)
        2. Masukkan teks berita di tab "Prediksi Tunggal"
        3. Klik tombol "Analisis Sentimen"
        4. Lihat hasil prediksi dan probabilitas
        
        Untuk analisis batch, upload file CSV dengan kolom 'judul'.
        """)
    
    st.markdown("---")
    
    st.subheader("📞 Kontak & Kontribusi")
    st.markdown("""
    - **GitHub**: [Repository Project](https://github.com/yourusername/sentiment-analysis)
    - **Issues**: Laporkan bug atau request fitur
    - **Pull Request**: Kontribusi kode sangat diterima!
    
    Dikembangkan dengan ❤️ menggunakan Python dan Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>© 2024 Analisis Sentimen Berita CNBC Indonesia</p>
    <p>Dibuat dengan Streamlit • Python • Machine Learning</p>
</div>
""", unsafe_allow_html=True)

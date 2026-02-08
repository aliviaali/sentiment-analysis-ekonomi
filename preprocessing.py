def preprocess_text(text):
    """
    Fungsi untuk preprocessing teks berita
    
    Steps:
    1. Lowercase
    2. Remove URLs, email, mention
    3. Keep financial terms (Rp, US$, %)
    4. Remove numbers and punctuation
    5. Remove stopwords
    6. Stemming
    """
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email dan mention
    text = re.sub(r'\S+@\S+|@\w+', '', text)
    
    # Preserve financial terms
    text = re.sub(r'rp\s*[\d.,]+\s*[tmb]?', 'rupiah', text)
    text = re.sub(r'us\$[\d.,]+', 'dolar', text)
    text = re.sub(r'\d+[.,]?\d*%', 'persen', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize dan remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Join dan stemming
    text = ' '.join(words)
    text = stemmer.stem(text)
    
    return text

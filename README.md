# Models Directory

Folder ini berisi model machine learning yang sudah di-training.

## File yang Dibutuhkan

Setelah training model di Google Colab atau menggunakan `train.py`, simpan file-file berikut di folder ini:

1. **tfidf_vectorizer.pkl** (~10-50 MB)
   - TF-IDF vectorizer untuk feature extraction
   - Digunakan untuk mengubah teks menjadi numerical features

2. **naive_bayes_model.pkl** (~5-20 MB)
   - Trained Multinomial Naive Bayes model
   - Model probabilistik untuk klasifikasi sentimen

3. **svm_model.pkl** (~10-30 MB)
   - Trained Linear SVM model
   - Model dengan akurasi tinggi (>85%)

## Cara Mendapatkan Model

### Opsi 1: Training di Google Colab (Recommended)

1. Upload `Sentiment_Analysis_NB_SVM.ipynb` ke Google Colab
2. Upload dataset `Dataset-CNBCI-Sentimented.csv`
3. Jalankan semua cell
4. Download 3 file .pkl yang dihasilkan
5. Simpan di folder ini

### Opsi 2: Training Lokal

```bash
# Dari root directory project
python train.py data/Dataset-CNBCI-Sentimented.csv
```

Model akan otomatis disimpan di folder `models/`.

## Catatan Penting

⚠️ **File model TIDAK di-commit ke Git karena ukurannya besar!**

Untuk deployment:
- Gunakan **Git LFS** (Large File Storage)
- Atau simpan di **Google Drive** dan download saat runtime
- Atau gunakan **model compression/quantization**

## Model Information

| Model | Size | Accuracy | Training Time |
|-------|------|----------|---------------|
| TF-IDF Vectorizer | ~20 MB | - | ~2 min |
| Naive Bayes | ~10 MB | ~82% | ~1 min |
| SVM (Linear) | ~15 MB | >85% | ~5 min |

Total: ~45 MB

## Verify Models

Untuk memverifikasi model sudah ada:

```python
import os

models = [
    'tfidf_vectorizer.pkl',
    'naive_bayes_model.pkl',
    'svm_model.pkl'
]

for model in models:
    path = f'models/{model}'
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"✓ {model}: {size:.2f} MB")
    else:
        print(f"✗ {model}: NOT FOUND")
```

## Loading Models

```python
import pickle

# Load vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load Naive Bayes
with open('models/naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# Load SVM
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
```

## Troubleshooting

### Error: File not found
Pastikan file ada di folder `models/` dengan nama yang tepat.

### Error: Can't load pickle
Pastikan scikit-learn version sama dengan saat training.

### Error: Out of memory
File model terlalu besar. Consider model compression.

---

**Need help?** Open an issue on GitHub!

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load Dataset (SOLUSI ANTI-404)
print("📦 Menyiapkan data latihan internal...")
import pandas as pd

# Kita buat dataset sendiri yang cukup banyak supaya model bisa belajar 3 kelas
data_latihan = {
    'ulasan': [
        'barang bagus banget ori', 'puas belanja di sini', 'mantap pengiriman cepat',
        'kualitas jempolan produk original', 'packing rapi aman sentosa', 'rekomendasi banget buat kalian',
        'kecewa barang rusak pas sampai', 'penjual galak respon lama banget', 'barang kw parah menyesal beli',
        'pengiriman telat seminggu lebih', 'nyesel beli di sini barang tidak fungsi', 'produk tidak sesuai foto asli',
        'barang sudah sampai sesuai pesanan', 'biasa saja kualitas standar', 'lumayan buat harga segini',
        'biasa aja sih sesuai harga', 'oke lah sesuai deskripsi produk', 'barang diterima dengan baik tanpa cacat'
    ] * 20, # Kita duplikasi supaya data cukup banyak untuk latihan (360 data)
    'sentimen': [
        'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative', 'negative',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
    ] * 20
}

df = pd.DataFrame(data_latihan)
print(f"✅ Dataset siap dengan {len(df)} baris data!")

# Proses rename (biar aman kalau kodingan lu di bawahnya pake kolom ini)
df = df.rename(columns={'comment': 'ulasan', 'sentiment': 'sentimen'})

# 2. Pre-processing Sederhana (Pembersihan Teks)
def clean_text(text):
    text = text.lower() # Menjadi huruf kecil
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Hapus tanda baca
    text = re.sub(r'\d+', '', text) # Hapus angka
    text = text.strip() # Hapus spasi berlebih
    return text

print("🧹 Melakukan pembersihan data...")
df = df.rename(columns={'comment': 'ulasan', 'sentiment': 'sentimen'})
df = df.dropna(subset=['ulasan', 'sentimen'])
df['ulasan_clean'] = df['ulasan'].apply(clean_text)

# 3. Feature Extraction (TF-IDF)
# Menambahkan stop_words='indonesian' (opsional, perlu list kata)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['ulasan_clean'])
y = df['sentimen']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Definisi Model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='linear', probability=True),
    "Naive Bayes": MultinomialNB()
}

# 5. Pelatihan & Evaluasi
results = {}

print("\n=== 📊 HASIL EVALUASI MODEL (TRAIN VS TEST) ===")
for name, model in models.items():
    # Training
    model.fit(X_train, y_train)

    # Prediksi untuk data testing
    y_pred = model.predict(X_test)

    # Menghitung skor untuk data training dan data testing
    train_acc = model.score(X_train, y_train) # Skor saat latihan
    test_acc = accuracy_score(y_test, y_pred) # Skor saat ujian (data baru)

    results[name] = test_acc

    print(f"\n✅ Model: {name}")
    print(f"Akurasi Training: {train_acc:.2%}")
    print(f"Akurasi Testing : {test_acc:.2%}")

    # Jika selisihnya jauh, berarti Overfit
    selisih = train_acc - test_acc
    if selisih > 0.05:
        print(f"⚠️ Indikasi Overfitting: Selisih {selisih:.2%}")
    else:
        print("✨ Model stabil (Good Fit)")

    print(classification_report(y_test, y_pred, zero_division=0))

# 6. Visualisasi Perbandingan Akurasi (Versi Simple biar gak error)
plt.figure(figsize=(10, 5))
# Hapus bagian 'legend=False' kalau masih error
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Perbandingan Akurasi Model Sentiment Analysis')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.show()

# 7. Simulasi Prediksi ulasan baru
def prediksi_baru(teks):
    teks_clean = clean_text(teks)
    vektor = tfidf.transform([teks_clean])
    print(f"\n--- 📝 Uji Coba Kalimat ---")
    print(f"Teks: '{teks}'")
    for name, model in models.items():
        hasil = model.predict(vektor)
        print(f"[{name}]: {hasil[0]}")

prediksi_baru("kecewa banget, barangnya rusak pas sampe")
prediksi_baru("mantap, kualitas oke punya!")

import joblib
# Simpan hasil latihan biar bisa dipake di website
joblib.dump(models, "semua_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
print("\n💾 MANTAP! File PKL udah jadi. Sekarang waktunya buka website!")
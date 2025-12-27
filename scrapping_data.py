from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

options = Options()
# Memberikan identitas browser asli agar tidak gampang diblokir
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
options.add_argument("--window-size=1920,1080")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.tokopedia.com/sekondhand/review"
driver.get(url)

ulasan_all = []
halaman_sekarang = 1

print("🚀 Script Jalan! Fokus ke jendela Chrome yang terbuka...")
print("💡 TIPS: Jika muncul pop-up login/promo, silakan klik 'X' atau tutup manual di jendela Chrome-nya.")

try:
    while True:
        print(f"📄 Memproses halaman {halaman_sekarang}...")
        
        # Scroll pelan agar ulasan dimuat
        for _ in range(4):
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(1)

        # Ambil ulasan
        reviews = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span[data-testid="lblItemUlasan"]'))
        )
        
        for r in reviews:
            if r.text and r.text not in ulasan_all:
                ulasan_all.append(r.text)

        print(f"✅ Berhasil mengambil {len(ulasan_all)} ulasan.")

        # Cari tombol Next
        try:
            xpath_next = "//button[@aria-label='Laman berikutnya']"
            
            # Tunggu tombol sampai bisa diklik
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xpath_next))
            )
            
            if next_button.get_attribute("disabled") is not None:
                print("⏹️ Mencapai halaman terakhir.")
                break
            
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
            time.sleep(2)
            driver.execute_script("arguments[0].click();", next_button)
            
            halaman_sekarang += 1
            time.sleep(5) # Jeda lebih lama agar tidak dicurigai
            
        except Exception as e:
            print(f"⏹️ Berhenti di halaman {halaman_sekarang}. Cek apakah ada pop-up yang menghalangi?")
            # Jangan langsung quit, beri waktu buat user liat apa yang terjadi
            time.sleep(10) 
            break

except Exception as e:
    print(f"⚠️ Error: {e}")

df_hasil = pd.DataFrame(ulasan_all, columns=['ulasan'])
df_hasil.to_csv("ulasan_tokopedia_full.csv", index=False)
print(f"\n🎉 HASIL AKHIR: {len(df_hasil)} ulasan.")
driver.quit()
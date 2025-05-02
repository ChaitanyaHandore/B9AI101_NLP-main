import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

from user_agents import get_random_headers
# from proxies import get_random_proxy  # Uncomment if using proxies

# ----------------------
# 🔧 Setup & File Paths
# ----------------------
DATA_DIR = r"C:\Users\Autom\PycharmProjects\B9AI101_NLP\data"
os.makedirs(DATA_DIR, exist_ok=True)

attraction_file = os.path.join(DATA_DIR, "tripadvisor_attractions.xlsx")
output_file = os.path.join(DATA_DIR, "full_reviews.xlsx")

df = pd.read_excel(attraction_file)
all_reviews = []

# ----------------------
# 🚗 Chrome Setup
# ----------------------
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={get_random_headers()['User-Agent']}")
    # chrome_options.add_argument(f'--proxy-server={get_random_proxy()}')  # Enable if needed
    return webdriver.Chrome(options=chrome_options)

# ----------------------
# 🔁 Loop through Attractions
# ----------------------
for idx, row in df.iterrows():
    url = row["URL"]
    driver = setup_driver()

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h1[@class='biGQs _P rRtyp']"))
        )
    except Exception as e:
        print(f"❌ Error loading {url}: {e}")
        driver.quit()
        continue

    attraction_name = driver.find_element(By.XPATH, "//h1[@class='biGQs _P rRtyp']").text
    print(f"🚀 Visiting: {attraction_name}")
    time.sleep(2)

    review_count = 0

    while review_count < 100:
        time.sleep(2)
        reviews = driver.find_elements(By.XPATH, "//div[@data-automation='reviewCard']")

        if not reviews:
            print(f"⏳ No reviews found on this page for {attraction_name}.")
            break

        for review in reviews:
            try:
                title = review.find_element(By.XPATH, ".//a[contains(@class, 'FGwzt')]").text
                text = review.find_element(By.XPATH, ".//div[contains(@class, 'KxBGd')]").text
                rating_svg = review.find_element(By.CLASS_NAME, "evwcZ")
                rating_title = rating_svg.get_attribute("aria-labelledby")
                rating = rating_svg.get_attribute("title") or rating_svg.get_attribute("aria-label")
                date = review.find_element(By.XPATH, ".//div[contains(@class, 'ncFvv') and contains(text(),'Written')]").text

                all_reviews.append({
                    "Attraction": attraction_name,
                    "Review Title": title,
                    "Review Text": text,
                    "Rating": rating if rating else rating_title,
                    "Date": date,
                    "URL": url
                })
                review_count += 1

                if review_count >= 100:
                    break

            except Exception as e:
                print(f"⚠️ Review skipped due to error: {e}")
                continue

        # 🔁 Try clicking "Next"
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.ui_button.nav.next.primary"))
            )
            next_button.click()
        except:
            print("🔚 No more pages.")
            break

    print(f"✅ Collected {review_count} reviews from: {attraction_name}")
    driver.quit()
    time.sleep(3)

# ----------------------
# 📥 Save to Excel
# ----------------------
pd.DataFrame(all_reviews).to_excel(output_file, index=False)
print(f"🎯 Done! Reviews saved to: {output_file}")

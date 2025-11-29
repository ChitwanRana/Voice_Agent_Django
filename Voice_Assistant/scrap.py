import json
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import os



# ------------------------------ LOGGING ------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("Patliputra")

load_dotenv()  

BASE_URL = os.getenv("BASE_URL")


# ------------------------------ DRIVER ------------------------------
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


# ------------------------------ SPECIALITIES ------------------------------
def get_specialities(driver):
    driver.get(BASE_URL + "/opd_services")
    time.sleep(2)

    last_height = driver.execute_script("return document.body.scrollHeight")
    specialities_seen = 0

    while True:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Try extracting again
        items = driver.find_elements(By.CSS_SELECTOR, "a.service_heading")

        # If new items loaded → continue scrolling
        if len(items) > specialities_seen:
            specialities_seen = len(items)
            last_height = driver.execute_script("return document.body.scrollHeight")
            continue

        # No new items → stop scrolling
        break

    result = []
    for i in items:
        name = i.text.strip()
        link = i.get_attribute("href")
        if name and link:
            result.append((name, link))

    logger.info(f"FOUND TOTAL SPECIALITIES: {len(result)}")

    return result


# ------------------------------ JS TAB ACTIVATION ------------------------------
def activate_details_tab(driver):
    driver.execute_script("""
        try {
            document.querySelector("#pills-home-tab").click();
        } catch(e) {}
    """)
    time.sleep(1)


def activate_doctors_tab(driver):
    driver.execute_script("""
        try {
            document.querySelector("#pills-profile-tab").click();
        } catch(e) {}
    """)
    time.sleep(1)

# ------------------------------ TREATMENT EXTRACTION ------------------------------
def extract_treatments(driver):
    treatments = []

    # ---- TYPE A: TABLE ----
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    if rows:
        for r in rows:
            cols = r.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 2:
                treatments.append({
                    "english": cols[0].text.strip(),
                    "hindi": cols[1].text.strip()
                })
        if treatments:
            return treatments

    # ---- TYPE B: <ul><li> ----
    lis = driver.find_elements(By.CSS_SELECTOR, ".tf_service_details_text ul li")
    if lis:
        for li in lis:
            text = li.text.strip()
            if text:
                treatments.append({
                    "english": text,
                    "hindi": ""
                })
        if treatments:
            return treatments

    # ---- TYPE C: Bullet/Paragraph ----
    ps = driver.find_elements(By.CSS_SELECTOR, ".tf_service_details_text p")
    for p in ps:
        raw = p.text.strip().replace("\u2022", "").replace("•", "")
        for line in raw.split("\n"):
            line = line.strip()
            if line and "treatment" not in line.lower() and len(line) < 200:
                treatments.append({"english": line, "hindi": ""})

    if treatments:
        return treatments

    # ---- TYPE D: Fallback description ----
    fallback_ps = driver.find_elements(By.CSS_SELECTOR, ".blog-details p")
    fallback_list = [p.text.strip() for p in fallback_ps if len(p.text.strip()) > 10]

    if fallback_list:
        return [{"english": f, "hindi": ""} for f in fallback_list]

    # ---- NOTHING FOUND ----
    return [{"english": "No treatment information available", "hindi": ""}]


# ------------------------------ DOCTORS EXTRACTION ------------------------------
def extract_doctors(driver):
    doctors = []
    cards = driver.find_elements(By.CSS_SELECTOR, ".single_team")

    for c in cards:
        try:
            name = c.find_element(By.CSS_SELECTOR, "h6").text.strip()
            qual = c.find_element(By.CSS_SELECTOR, "p").text.strip()
            doctors.append({"name": name, "qualification": qual})
        except:
            pass

    return doctors


# ------------------------------ SCRAPE EACH DEPARTMENT ------------------------------
def scrape_department(driver, name, url):
    logger.info(f"Scraping: {name}")

    driver.get(url)
    time.sleep(2)

    # ---- Activate details tab ----
    activate_details_tab(driver)
    treatments = extract_treatments(driver)

    # ---- Activate doctors tab ----
    activate_doctors_tab(driver)
    doctors = extract_doctors(driver)

    return {"treatments": treatments, "doctors": doctors}


# ------------------------------ MAIN SCRAPE ALL ------------------------------
def scrape_all():
    driver = init_driver()
    final_data = {}

    try:
        specialities = get_specialities(driver)

        for name, link in specialities:
            final_data[name] = scrape_department(driver, name, link)

    finally:
        driver.quit()

    return final_data


# ------------------------------ RUN ------------------------------
if __name__ == "__main__":
    data = scrape_all()

    with open("patliputra_final.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n\n🟢 Scraping Completed Successfully!")
    print("📁 Output saved in: patliputra_final.json\n")

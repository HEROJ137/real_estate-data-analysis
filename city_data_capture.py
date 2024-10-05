from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
import platform
import shutil
import time
import json
import os

# 크롬 버전 확인 함수
def get_chrome_version(system = platform.system()):
    
    if system == "Windows":
        command = r'reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version'
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        version_line = result.stdout.strip().split('\n')[-1]
        version = version_line.split()[-1]
        
    elif system == "Darwin":  # macOS
        command = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"]
        result = subprocess.run(command, capture_output=True, text=True)
        version = result.stdout.strip().split()[-1]
        
    elif system == "Linux":
        command = ["google-chrome", "--version"]
        result = subprocess.run(command, capture_output=True, text=True)
        version = result.stdout.strip().split()[-1]
        
    else:
        version = "Unknown OS"

    return version

# Selenium driver 설정 및 초기화 함수
def create_driver():
    options = Options()
    options.add_experimental_option("detach", True)

    chrome_driver_path = ChromeDriverManager().install()
    chrome_version = get_chrome_version()
    
    if chrome_version in chrome_driver_path:service = Service(chrome_driver_path)
    else:
        parts = chrome_driver_path.split(os.sep)
        driver_version,driver_version_path,remaining_path = None,None,None

        for i, part in enumerate(parts):
            if chrome_version[:5] in part:
                driver_version = part
                driver_version_path = os.sep.join(parts[:i+1])
                remaining_path = os.sep.join(parts[i+1:])
                break

        if driver_version and driver_version_path and remaining_path:
            new_versioned_path = driver_version_path.replace(driver_version, chrome_version)
            
            if os.path.exists(new_versioned_path):shutil.rmtree(new_versioned_path)
            shutil.copytree(driver_version_path, new_versioned_path)
            shutil.rmtree(driver_version_path)

            service = Service()
    
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_position(0, 0)
    return driver

# 시도명 리스트
city_list = [
    '서울특별시',
    '부산광역시',
    '대구광역시',
    '인천광역시',
    '광주광역시',
    '대전광역시',
    '울산광역시',
    '세종특별자치시',
    '경기도',
    '충청북도',
    '충청남도',
    '전라남도',
    '경상북도',
    '경상남도',
    '제주특별자치도',
    '강원특별자치도',
    '전북특별자치도'
]

def city_data_capture(driver, city_list):
    city_data = {}

    # 국토교통부 실거래가 - 단독/다가구 건물 데이터 url 접속
    driver.get("https://rt.molit.go.kr/pt/gis/gis.do?srhThingSecd=C&mobileAt=")
    driver.implicitly_wait(5)
    time.sleep(1)

    # 주소 선택 클릭
    WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, f'//*[@id="pnuBtn"]'))
    ).click()

    # 시도명 선택
    for city_name in city_list:
        try:
            # 시도명 선택 (시도 선택 시마다 요소를 다시 탐색)
            city_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[2]/select'))
            );time.sleep(0.1)
            city_select = Select(city_element)
            city_select.select_by_visible_text(city_name)

            # 시군구명 선택 (항상 새로운 요소 탐색)
            county_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[3]/select'))
            );time.sleep(0.1)
            county_select = Select(county_element)
            county_all_options = county_select.options

            # 시도명 데이터를 딕셔너리에 추가
            city_data[city_name] = {}

            for county_option in county_all_options:
                county_name = county_option.text
                if county_name == '전체': continue  # '전체' 옵션을 제외

                # 시군구 선택 (항상 새로운 요소 탐색)
                county_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[3]/select'))
                );time.sleep(0.1)
                county_select = Select(county_element)
                county_select.select_by_visible_text(county_name)

                # 읍면동명 선택 (항상 새로운 요소 탐색)
                village_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[4]/div/select'))
                );time.sleep(0.1)
                village_select = Select(village_element)
                village_all_options = village_select.options

                # 시군구명 데이터를 딕셔너리에 추가
                city_data[city_name][county_name] = []

                for village_option in village_all_options:
                    village_name = village_option.text
                    if village_name == '전체': continue  # '전체' 옵션을 제외

                    # 읍면동명을 리스트에 추가
                    city_data[city_name][county_name].append(village_name)

        except Exception as e:
            print(f"Error while processing {city_name}: {e}")

    return city_data

def main():
    # WebDriver 실행
    driver = create_driver()

    # 크롤링 후 JSON 파일로 저장
    with open('location_data.json', 'w', encoding='utf-8') as f:
        json.dump(city_data_capture(driver, city_list), f, ensure_ascii=False, indent=4)

    # WebDriver 종료
    driver.quit()

if __name__ == "__main__":
    main()
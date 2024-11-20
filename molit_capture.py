from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
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

knu = {"강원특별자치도": {"춘천시": ["효자동", "후평동", "석사동", "퇴계동"]}}

# 국토교통부 실거래 데이터를 크롤링하고 CSV 파일로 저장하는 함수
def molit_capture(driver, year):
    # 데이터를 저장할 디렉토리 경로 설정
    directory = 'molit_data'

    # 디렉토리가 존재하지 않으면 새로 생성
    if not os.path.exists(directory): os.makedirs(directory)

    # 국토교통부 실거래가 시스템 URL로 접속
    driver.get("https://rt.molit.go.kr/pt/gis/gis.do?srhThingSecd=C&mobileAt=")
    driver.implicitly_wait(5)  # 페이지 로딩을 대기
    time.sleep(1)  # 안정적인 동작을 위해 추가 대기

    # 기준 연도 선택 버튼 클릭
    WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, f'//*[@id="yearBtn"]'))
    ).click();time.sleep(0.1)  # 안정적인 동작을 위해 추가 대기

    # 기준 연도 선택 (입력받은 year를 기준으로 선택)
    year_element = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, f'/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/select'))
    );time.sleep(0.1)
    Select(year_element).select_by_visible_text(year)

    # 주소 선택 버튼 클릭
    WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, f'//*[@id="pnuBtn"]'))
    ).click();time.sleep(0.1)

    # 시도명 선택 (강원특별자치도)
    city_element = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[2]/select'))
    );time.sleep(0.1)
    Select(city_element).select_by_visible_text("강원특별자치도");time.sleep(0.1)

    # 시군구명 선택 (춘천시)
    county_element = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[3]/select'))
    );time.sleep(0.1)
    Select(county_element).select_by_visible_text("춘천시");time.sleep(0.1)

    # 읍면동명 선택 (효자동, 후평동, 석사동, 퇴계동 반복)
    for knu_village in knu["강원특별자치도"]["춘천시"]:
        # 각 읍면동명을 선택
        village_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[1]/div[1]/div[3]/div[4]/div/select'))
        );time.sleep(0.1)
        Select(village_element).select_by_visible_text(knu_village);time.sleep(0.1)

        # 돋보기 버튼 클릭 (데이터 검색 시작)
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, f'/html/body/div/section/div[1]/div[2]/div[1]/div[1]/ul/li[5]/div'))
        ).click();time.sleep(2)

        # 전월세 버튼 클릭 (전월세 데이터 필터링)
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, f'/html/body/div/section/div[1]/div[2]/div[2]/div[1]/div[2]/div[1]/button[2]'))
        ).click();time.sleep(2)

        # 면적에서 '60제곱미터 이하' 선택
        area_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[2]/div[2]/table/tbody/tr[2]/td[4]/div/select'))
        );time.sleep(0.1)
        Select(area_element).select_by_visible_text('60제곱미터 이하');time.sleep(2)

        # 해당 연도의 데이터를 포함한 월 목록 가져오기
        month_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[2]/div[2]/table/tbody/tr[2]/td[3]/div/select'))
        );time.sleep(2)

        month_select = Select(month_element)
        month_all_options = month_select.options
        month_list = [month_option.text for month_option in month_all_options if month_option.text != '전체']

        # 월별 데이터 반복 수집
        for month in month_list:
            month_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '/html/body/div/section/div[1]/div[2]/div[2]/div[2]/table/tbody/tr[2]/td[3]/div/select'))
            );time.sleep(0.1)
            Select(month_element).select_by_visible_text(month);time.sleep(2)

            # 데이터 HTML 가져오기
            raw_data = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div/section/div[1]/div[2]/div[2]/div[3]/div"))
            );time.sleep(2)

            raw_content = raw_data.get_attribute('outerHTML')
            soup = BeautifulSoup(raw_content, 'lxml')

            # HTML 데이터를 테이블 형태로 파싱
            rows = soup.find_all('tr')
            table_data = []
            row_buffer = None
            for row in rows:
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                if len(cols) == 9: row_buffer = cols    # 첫 번째 줄 데이터 저장
                elif len(cols) == 5:    # 두 번째 줄과 결합하여 완전한 데이터로 저장
                    complete_row = [
                        row_buffer[0], row_buffer[1], cols[0], row_buffer[2], row_buffer[3], cols[1], month + '/' + row_buffer[4],
                        row_buffer[5], cols[2], row_buffer[6], cols[3], row_buffer[7], cols[4], row_buffer[8]
                    ]
                    table_data.append(complete_row)
                    row_buffer = None

            # DataFrame으로 변환
            df = pd.DataFrame(table_data, columns=[
                '법정동', '지번', '도로명', '주택유형', '연면적(㎡)', '계약구분', '계약일',
                '계약기간', '갱신요구권사용', '보증금(만원)', '월세(만원)', '종전보증금(만원)', '종전월세(만원)', '전산공부'])

            # CSV 파일로 저장
            file_path = os.path.join(directory, f'{year}-{month}-{knu_village}.csv')
            df.to_csv(file_path, index=False)
            print(f'파일이 {file_path}에 저장되었습니다.')

        # 데이터 창 닫기
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, f'/html/body/div/section/div[1]/div[2]/div[2]/div[1]/div[1]/a'))
        ).click();time.sleep(0.1)

        # 다시 주소 선택 버튼 클릭 (다음 읍면동 데이터 수집 준비)
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, f'//*[@id="pnuBtn"]'))
        ).click();time.sleep(0.1)

def main():
    # WebDriver 실행
    driver = create_driver()

    for y in range(2014,2024):molit_capture(driver, f"{y}")

    # WebDriver 종료
    driver.quit()

if __name__ == "__main__":
    main()
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

knu_villages = {"효자동":f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/div/div[2]/ul/li[38]', 
       "후평동":f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/div/div[2]/ul/li[39]', 
       "석사동":f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/div/div[2]/ul/li[15]', 
       "퇴계동":f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/div/div[2]/ul/li[37]'
}

def molit_capture(driver, year):
    # CSV 파일을 저장할 디렉토리 경로 설정
    directory = 'naver_data'

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):os.makedirs(directory)

    # 네이버부동산 - 원룸 투룸 매물 url 접속
    driver.get("https://new.land.naver.com/rooms?ms=37.3595704,127.105399,16&a=APT:OPST:ABYG:OBYG:GM:OR:VL:DDDGG:JWJT:SGJT:HOJT&e=RETAIL&aa=SMALLSPCRENT")
    driver.implicitly_wait(5);time.sleep(1)

    # 동일매물 묶기
    checkbox = driver.find_element(By.ID, f'address_group2')
    if not checkbox.is_selected() : checkbox.click();time.sleep(0.5)

    # 주소 선택 팝업 클릭
    WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/a/span[1]'))
    ).click();time.sleep(0.5)
    
    # 강원도 선택
    WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/div/div[2]/ul/li[10]'))
    ).click();time.sleep(0.5)

    # 춘천시 선택
    WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/div/div[2]/ul/li[13]'))
    ).click();time.sleep(0.5)

    # 읍면동명 선택 (효자동, 후평동, 석사동, 퇴계동)
    for knu_village_name, knu_village_xpath in knu_villages:
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, knu_village_xpath))).click()
        time.sleep(0.5)

def main():
    # WebDriver 실행
    driver = create_driver()

    for y in range(2014,2024):molit_capture(driver, f"{y}")

    # WebDriver 종료
    driver.quit()

if __name__ == "__main__":
    main()
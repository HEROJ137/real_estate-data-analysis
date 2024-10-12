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
from datetime import datetime
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

knu_villages = {"효자동":f'area37', "후평동":f'area38', "석사동":f'area14', "퇴계동":f'area36'}

def extract_real_estate_data(raw_data, adress, data):
    # BeautifulSoup을 사용해 매물 정보 HTML 파싱
    raw_content = raw_data.get_attribute('outerHTML')
    soup = BeautifulSoup(raw_content, 'html.parser')

    # 매물 정보가 포함된 모든 'item' 클래스 요소를 찾음
    items = soup.find_all('div', class_='item')

    # 각 매물 정보를 반복하며 데이터 추출
    for item in items:
        try:
            # 매물 유형, 제목, 가격 정보, 면적, 층수, 방향, 태그 및 기타 정보 추출
            property_type = item.find('strong', class_='type').get_text(strip=True) if item.find('strong', class_='type') else None
            title = item.find('span', class_='text').get_text(strip=True) if item.find('span', class_='text') else None
            adress_info = adress
            price = item.find('div', class_='price_line').get_text(strip=True) if item.find('div', class_='price_line') else None
            area = item.find('span', class_='spec').get_text(strip=True) if item.find('span', class_='spec') else None
            direction = area.split(', ')[-1] if area else None
            floors = area.split(', ')[1] if area else None
            tags = ', '.join([tag.get_text(strip=True) for tag in item.find_all('span', class_='tag')])
            confirmed_date = item.find('span', class_='data').get_text(strip=True) if item.find('span', class_='data') else None
            agent_info = item.find_all('a', class_='agent_name')
            agent_platform = agent_info[0].get_text(strip=True) if len(agent_info) > 0 else None
            agent = agent_info[1].get_text(strip=True) if len(agent_info) > 1 else None

            # 매물 정보를 리스트에 추가 (매물 유형, 매물 제목, 보증금/월세, 면적, 층수, 방향, 태그, 중개 플랫폼, 중개사, 등록일)
            data.append([
                property_type, title, adress_info, price, 
                area.split(', ')[0] if area else None, floors, direction,
                tags, agent_platform, agent, confirmed_date
            ])
        except Exception as e:
            print(f"데이터 추출 중 오류 발생: {e}")
            continue

    return data

def naver_capture(driver):
    # CSV 파일을 저장할 디렉토리 경로 설정
    directory = 'naver_data'

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):os.makedirs(directory)

    # 네이버부동산 - 원룸 투룸 매물 url 접속
    driver.get("https://new.land.naver.com/rooms?ms=37.3595704,127.105399,16&a=APT:OPST:ABYG:OBYG:GM:OR:VL:DDDGG:JWJT:SGJT:HOJT&e=RETAIL&aa=SMALLSPCRENT")
    driver.implicitly_wait(5);time.sleep(1)
    driver.execute_script("document.body.style.zoom='75%'")

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
    for knu_village_name, knu_village_id in knu_villages.items():
        # 읍면동명 선택 (순서대로)
        radio_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, knu_village_id)))
        driver.execute_script("arguments[0].click();", radio_button)
        time.sleep(0.5)

        # 매물 아이템 리스트 업데이트를 위해 끝까지 스크롤
        last_height = driver.execute_script("return document.querySelector('.item_list').scrollHeight")
        while True :
            driver.execute_script(f"document.querySelector('.item_list').scrollBy(0, {last_height});");time.sleep(0.5)
            new_height = driver.execute_script("return document.querySelector('.item_list').scrollHeight")
            if new_height == last_height : break
            last_height = new_height

        time.sleep(1)
        driver.execute_script(f"document.querySelector('.item_list').scrollTo(0, 0);");time.sleep(1)

        def select_action(naver_data, item_index, delay = 0.5):
            item_list_raw_data = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, f"/html/body/div[2]/div/section/div[2]/div[1]/div/div[2]/div/div[2]/div/div/div/div[{item_index}]"))
            );time.sleep(0.5)
            
            if "공인중개사협회매물" not in item_list_raw_data.text :
                # 네이버에서 보기 버튼이 있다면 버튼 클릭
                try:
                    WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, f'//*[@id="listContents1"]/div/div/div/div[{item_index}]/div/div[2]/a'))
                    ).click();time.sleep(delay)
                # 없다면 매물 박스 클릭
                except:
                    WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, f'//*[@id="listContents1"]/div/div/div/div[{item_index}]'))
                    ).click();time.sleep(delay)

                # 소재지 정보 가지고 오기
                adress = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, f'//*[@id="detailContents1"]/div[1]/table/tbody/tr[1]/td'))
                ).text

                # 매물 데이터 추출 및 리스트에 추가
                if "춘천시" in adress : 
                    print(f"{knu_village_name} - item_list_{item_index} 소재지 추가 : {adress}")
                    extract_real_estate_data(item_list_raw_data, adress, naver_data)
                else : 
                    print(f"{knu_village_name} - item_list_{item_index} 소재지 없음 : None")
                    extract_real_estate_data(item_list_raw_data, None, naver_data)

                print(f" > 데이터 추출 완료")

                driver.execute_script(f"document.querySelector('.item_list').scrollBy(0, 175);");time.sleep(0.5)
            
            else : 
                driver.execute_script(f"document.querySelector('.item_list').scrollBy(0, 225);");time.sleep(0.5)
                print(f"{knu_village_name} - item_list_{item_index}\n{item_list_raw_data.text}")

        # 매물 데이터 추출
        naver_data = []
        item_index = 2

        select_action(naver_data, 1, delay = 3)
        
        while True:
            item_element = driver.find_elements(By.XPATH, f'//*[@id="listContents1"]/div/div/div/div[{item_index}]')
            if item_element:
                select_action(naver_data, item_index)
                item_index += 1
            else:break


        df = pd.DataFrame(naver_data, columns=[
            '매물 유형', '매물 제목', '소재지', '가격', '면적', '층수', '방향', '태그', '중개 플랫폼', '중개사', '등록일'])
        
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d-%H:%M")
        file_path = os.path.join(directory, f'{formatted_date_time}-{knu_village_name}.csv')
        df.to_csv(file_path, index=False)
        print(f'파일이 {file_path}에 저장되었습니다.')
        time.sleep(0.5)

        # 세부 데이터 팝업 종료 버튼 클릭
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, f'/html/body/div[2]/div/section/div[2]/div[2]/div/button'))
        ).click();time.sleep(0.5)

        # 다음 읍면동명 선택을 위해 팝업 선택
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, f'/html/body/div[2]/div/section/div[2]/div[2]/div[1]/div/div/a/span[3]'))
        ).click();time.sleep(0.5)

def main():
    # WebDriver 실행
    driver = create_driver()

    naver_capture(driver)

    # WebDriver 종료
    driver.quit()

if __name__ == "__main__":
    main()
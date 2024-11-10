import pandas as pd
import glob
import os
import re

# CSV 파일이 있는 디렉토리 경로 지정
directory_path = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data'  # 실제 디렉토리 경로로 수정

# 디렉토리 내 모든 CSV 파일 불러오기
all_files = glob.glob(os.path.join(directory_path, "*.csv"))

def explore_data_to_organize():
    # '매물 제목'의 고유값을 저장할 집합(set) 생성
    unique_titles = set()

    # 모든 CSV 파일을 순회하며 '매물 제목' 열에서 중복 제거된 고유값 추출
    for file in all_files:
        df = pd.read_csv(file)
        unique_titles.update(df['매물 제목'].dropna().unique().tolist())

    # 최종 고유 제목 리스트로 변환
    unique_titles_list = list(unique_titles)
    remove_set = {'도시형생활주택', '다가구', '빌라', '상가주택', '일반원룸', '단독', '기타'}

    unique_titles_list = [unique_title for unique_title in unique_titles_list if unique_title not in remove_set]

    # '동' 앞 숫자를 포함한 텍스트 제거하고 이름만 추출
    titles_cleaned = set(re.split(r'\s+\d+', title)[0].strip() for title in unique_titles_list)

    # 결과 출력
    print("정리된 매물 제목:", titles_cleaned)

    # 결과 출력
    print("매물 제목의 고유값 목록 :")

    for title in titles_cleaned:
        print(title)

def data_cleaning():
    location_dict = {
        '석사주공2단지': '강원도 춘천시 후평동 899',
        '후평봉의': '강원도 춘천시 후평동 542-3',
        '주공5단지': '강원도 춘천시 후평동 481',
        '동보': '강원도 춘천시 효자동 821',
        '메가씨티': '강원도 춘천시 효자동 654-9',
        '크로바': '강원도 춘천시 후평동 511-5',
        '더베네치아스위트(생활숙박시설)': '강원도 춘천시 효자동 651',
        '동산': '강원도 춘천시 후평동 575-4',
        '에리트': '강원도 춘천시 후평동 529-4',
        '세경5차': '강원도 춘천시 석사동 694',
        '퇴계주공2단지': '강원도 춘천시 퇴계동 983',
        '퇴계주공1단지': '강원도 춘천시 퇴계동 967',
        '세경3차': '강원도 춘천시 후평동 845-1',
        '후평주공4단지': '강원도 춘천시 후평동 808-1',
        '근로복지(석사)': '강원도 춘천시 석사동 694-1',
        '신아': '강원도 춘천시 효자동 820',
        '주공7단지': '강원도 춘천시 후평동 477',
        '남춘천역코아루웰라움타워': '강원도 춘천시 퇴계동 369-9'
    }

    # 매물 제목에 따른 소재지 업데이트 함수 정의
    def update_location(row):
        for key, address in location_dict.items():
            if key in row['매물 제목']:
                return address
        return row['소재지']  # 기존 소재지 유지

    # 모든 CSV 파일 업데이트
    for file in all_files:
        df = pd.read_csv(file)  # 파일 읽기
        
        # '소재지' 열 업데이트
        df['소재지'] = df.apply(update_location, axis=1)
        
        # 업데이트된 데이터프레임을 원래 파일에 저장
        df.to_csv(file, index=False)
        print(f"소재지가 업데이트된 파일이 저장되었습니다: {file}")

data_cleaning()
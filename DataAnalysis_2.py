import os
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 디렉토리 경로
naver_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data_updated'
molit_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/molit_data'

# 기준금리 변화 CSV 파일 불러오기
rate_data = pd.read_csv('interest_rate_changes.csv')
rate_data['변경일자'] = pd.to_datetime(rate_data['변경일자'])

# 강원대학교 한빛관 위도 경도
hanbit_lat, hanbit_lon = 37.8687341, 127.7379901

# 기준금리를 가져오는 함수
def get_base_rate(contract_date):
    for _, row in rate_data.iterrows():
        if contract_date >= row['변경일자']:
            return row['기준금리'] / 100
    return rate_data.iloc[-1]['기준금리'] / 100

# 전세를 월세로 환산하는 함수
def convert_to_monthly_rent(jeonse, contract_date):
    base_rate = get_base_rate(contract_date)
    conversion_rate = 0.02 if contract_date.year >= 2021 else 0.035
    monthly_rent = jeonse * (base_rate + conversion_rate) / 12
    return monthly_rent

# 두 좌표 간 거리 계산 (Haversine formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# 전세와 월세를 처리하는 함수
def convert_deposit(price_str):
    price_str = price_str.replace('전세', '').replace(',', '').strip()  # '전세'와 쉼표 제거
    if '억' in price_str:
        parts = price_str.split('억')
        main = int(parts[0]) * 10000
        sub = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return main + sub
    elif price_str.isdigit():
        return int(price_str)
    else:
        raise ValueError(f"Invalid price format: {price_str}")

# 월세 환산 또는 기존 월세 처리
def process_price(row):
    if '전세' in row['가격']:
        deposit = convert_deposit(row['가격'])
        monthly_rent = convert_to_monthly_rent(deposit, datetime(2024, 10, 1))
        return '전세 환산', monthly_rent
    elif '월세' in row['가격']:
        monthly_rent = row['월세']
        return '월세', monthly_rent
    return '기타', 0

# 색상 설정 함수
def assign_color(location):
    if '효자동' in location:
        return 'yellow'
    elif '후평동' in location:
        return 'lightgreen'
    elif '석사동' in location:
        return 'orange'
    elif '퇴계동' in location:
        return 'skyblue'
    else:
        return 'gray'

# 데이터 읽기 및 처리
all_data = []
for file in os.listdir(naver_data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(naver_data_dir, file)
        data = pd.read_csv(file_path)
        data['파일명'] = file_path  # 각 데이터에 파일 경로 추가
        all_data.append(data)

# 모든 데이터를 하나로 합치기
df = pd.concat(all_data, ignore_index=True)

# 데이터 전처리
df['전용면적'] = df['면적'].str.split('/').str[1].str.replace('m²', '').replace('-', np.nan).astype(float)
df['보증금'] = df['가격'].str.extract(r'(\d+[억, ]*\d*)')[0].apply(convert_deposit)
df['월세'] = df['가격'].str.extract(r'/(\d+)')[0].astype(float)
df['색상'] = df['소재지'].apply(assign_color)

# '-'나 NaN인 전용면적은 제외
df = df.dropna(subset=['전용면적'])

# 중복 제거: 모든 컬럼 기준으로 중복된 행 제거
df = df.drop_duplicates()

# 거리와 월세 계산
results = []
for _, row in df.iterrows():
    distance = calculate_distance(hanbit_lat, hanbit_lon, row['위도'], row['경도'])
    rent_type, monthly_rent = process_price(row)
    rent_per_10m2 = (monthly_rent / row['전용면적']) * 10
    results.append({
        '거리': distance,
        '10m²당 월세': rent_per_10m2,
        '월세 유형': rent_type,
        '월세 금액': monthly_rent,
        '전용면적': row['전용면적'],
        '소재지': row['소재지'],
        '파일명': row['파일명'],
        '가격': row['가격']
    })

# 결과를 데이터프레임으로 변환
result_df = pd.DataFrame(results)

# 중복 제거: 모든 컬럼 기준으로 중복된 행 제거
result_df = result_df.drop_duplicates()

# 전세 환산과 월세를 분리
jeonse_df = result_df[result_df['월세 유형'] == '전세 환산']
jeonse_df = jeonse_df.drop_duplicates()

wolse_df = result_df[result_df['월세 유형'] == '월세']
wolse_df = wolse_df.drop_duplicates()

# 결과 시각화 및 추세선 추가
plt.figure(figsize=(12, 6))

# 산점도 그리기
def normalize(series, scale):
    return (series - series.min()) / (series.max() - series.min()) * scale

wolse_sizes = normalize(wolse_df['전용면적'], scale=150)
plt.scatter(wolse_df['거리'], wolse_df['10m²당 월세'], s=wolse_sizes, label='월세', alpha=0.1, color='green')

jeonse_sizes = normalize(jeonse_df['전용면적'], scale=150)
plt.scatter(jeonse_df['거리'], jeonse_df['10m²당 월세'], s=jeonse_sizes, label='전세 환산', alpha=0.1, color='blue')

# 월세 데이터 추세선
wolse_df_clean = wolse_df.dropna(subset=['거리', '10m²당 월세'])
wolse_df_clean = wolse_df_clean[
    ~np.isinf(wolse_df_clean['거리']) & ~np.isinf(wolse_df_clean['10m²당 월세'])
]

if len(wolse_df_clean) > 2:
    z_wolse = np.polyfit(wolse_df_clean['거리'], wolse_df_clean['10m²당 월세'], 2)
    p_wolse = np.poly1d(z_wolse)
    x_wolse = np.linspace(wolse_df_clean['거리'].min(), wolse_df_clean['거리'].max(), 100)
    plt.plot(x_wolse, p_wolse(x_wolse), linestyle='--', color='green', linewidth=2.5, label='월세 추세선')
else:
    print("월세 데이터가 부족하여 추세선을 생성할 수 없습니다.")
    
# 전세 데이터 추세선
jeonse_df_clean = jeonse_df.dropna(subset=['거리', '10m²당 월세'])
jeonse_df_clean = jeonse_df_clean[
    ~np.isinf(jeonse_df_clean['거리']) & ~np.isinf(jeonse_df_clean['10m²당 월세'])
]

if len(jeonse_df_clean) > 2:
    z_jeonse = np.polyfit(jeonse_df_clean['거리'], jeonse_df_clean['10m²당 월세'], 2)
    p_jeonse = np.poly1d(z_jeonse)
    x_jeonse = np.linspace(jeonse_df_clean['거리'].min(), jeonse_df_clean['거리'].max(), 100)
    plt.plot(x_jeonse, p_jeonse(x_jeonse), linestyle='--', color='blue', linewidth=2.5, label='전세 추세선')
else:
    print("전세 데이터가 부족하여 추세선을 생성할 수 없습니다.")

# 그래프 제목 및 축 레이블 설정
plt.title('강원대까지 거리 - 면적당 월세', fontsize=12)
plt.xlabel('강원대까지 거리 (km)', fontsize=10)
plt.ylabel('10m²당 월세 (만원)', fontsize=10)
plt.legend()
plt.grid(True)

# 그래프 보여주기
plt.show()

# 전세 데이터 추세선 다항함수식
if len(jeonse_df_clean) > 2:
    jeonse_equation = np.poly1d(z_jeonse)
    jeonse_coeffs = [f"{coeff:.6f}" for coeff in jeonse_equation.coefficients]
    jeonse_equation_str = " + ".join([f"{coeff}x^{i}" for i, coeff in enumerate(reversed(jeonse_coeffs))])
    print(f"전세 추세선 다항함수식: y = {jeonse_equation_str}")

# 월세 데이터 추세선 다항함수식
if len(wolse_df_clean) > 2:
    wolse_equation = np.poly1d(z_wolse)
    wolse_coeffs = [f"{coeff:.6f}" for coeff in wolse_equation.coefficients]
    wolse_equation_str = " + ".join([f"{coeff}x^{i}" for i, coeff in enumerate(reversed(wolse_coeffs))])
    print(f"월세 추세선 다항함수식: y = {wolse_equation_str}")

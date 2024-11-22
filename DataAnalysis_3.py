import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 문자열을 숫자로 변환하는 함수
def parse_price(price_str):
    # '전세', '월세'가 포함되지 않은 데이터는 무시
    if not (price_str.startswith('전세') or price_str.startswith('월세')):
        return None, None  # 무시
    
    # 전세 처리
    if price_str.startswith('전세'):
        price_str = price_str.replace('전세', '').strip()
        match = re.match(r"(?P<eok>\d+)억\s?(?P<cheon>\d+)?", price_str)
        if match:
            eok = int(match.group('eok')) * 10000  # 억 단위 변환
            cheon = int(match.group('cheon')) if match.group('cheon') else 0  # 천 단위 변환
            return eok + cheon, '전세'
        else:
            return int(price_str.replace(',', '')), '전세'  # 억 없이 단순 숫자
        
    # 월세 처리
    if price_str.startswith('월세'):
        price_str = price_str.replace('월세', '').strip()
        deposit, monthly = price_str.split('/')
        deposit = int(deposit.replace(',', '').strip())  # 보증금
        monthly = int(monthly.strip())  # 월세
        return monthly, '월세'
    
    return None, None  # 처리할 수 없는 경우

# 면적 데이터 파싱 함수
def parse_area(area_str):
    if not area_str: return None
    
    area_str = area_str.strip()
    
    # '공급/전용면적' 형식 처리
    if '/' in area_str:
        supply, exclusive = area_str.split('/')
        supply = re.sub(r'[^\d.]', '', supply)
        exclusive = re.sub(r'[^\d.]', '', exclusive)
        
        if exclusive: return float(exclusive)
        elif supply: return float(supply)
        else: return None
    
    # 전용면적만 있는 경우
    clean_area = re.sub(r'[^\d.]', '', area_str)
    if clean_area: return float(clean_area)
    
    return None

# 기준금리 변화 CSV 파일 불러오기
rate_data = pd.read_csv('interest_rate_changes.csv')

# 기준금리를 가져오는 함수
def get_base_rate(contract_date):
    for _, row in rate_data.iterrows():
        if contract_date >= pd.to_datetime(row['변경일자']):  # 변경일자를 Timestamp로 변환
            return row['기준금리'] / 100
    return rate_data.iloc[-1]['기준금리'] / 100  # 가장 오래된 금리 반환

# 전세에서 월세로 변환
# 전세전환율 계산 방법은 전세보증금 * ( 기준금리(%) + 대통령령에 의거한 월차임전환시산정률(%) ) / 12 
def convert_to_monthly_rent(jeonse, contract_date):
    base_rate = get_base_rate(contract_date)
    conversion_rate = 0.02 if contract_date.year >= 2021 else 0.035
    monthly_rent = jeonse * (base_rate + conversion_rate) / 12
    return monthly_rent

# 위도, 경도를 기반으로 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# 데이터 전처리 함수
def preprocess_naver_data(df):
    # 가격 처리
    prices = df['가격'].apply(parse_price)  # parse_price에서 (값, 유형) 반환
    df['월세'] = prices.apply(lambda x: x[0] if x and x[1] == '월세' else None)  # 월세일 경우 값 저장
    df['전세금'] = prices.apply(lambda x: x[0] if x and x[1] == '전세' else None)  # 전세일 경우 값 저장
    df['전세_여부'] = prices.apply(lambda x: 1 if x and x[1] == '전세' else 0)  # 전세 여부 저장
    
    # 전세를 월세로 환산 (2024년 11월 1일 기준)
    reference_date = datetime(2024, 11, 1)
    df['월세환산'] = df.apply(lambda row: convert_to_monthly_rent(row['전세금'], reference_date) if row['전세_여부'] == 1 else row['월세'], axis=1)
    
    # 면적 처리
    df['면적'] = df['면적'].apply(parse_area)

    # 면적이 60㎡ 이하인 데이터만 유지
    df = df[df['면적'] <= 60]
    
    # 유효 데이터 필터링
    df = df.dropna(subset=['월세환산', '면적'])  # 월세환산, 면적이 없는 데이터 제외
    return df

naver_path = "/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data_updated"
def load_and_process_naver_data(path):
    # naver_data 불러오기
    files = glob(os.path.join(path, "*.csv"))
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df[['소재지', '가격', '면적', '위도', '경도']])
    naver_combined = pd.concat(dfs, ignore_index=True)
    naver_combined = preprocess_naver_data(naver_combined)
    return naver_combined

# 강원대학교와 남춘천역 좌표
knu_coords = (37.8688188, 127.7448121)
namchuncheon_coords = (37.8638766, 127.7239661)

# naver_data 전처리
naver_data = load_and_process_naver_data(naver_path)
naver_data['거리_강원대'] = naver_data.apply(lambda row: haversine(row['위도'], row['경도'], *knu_coords), axis=1)
naver_data['거리_남춘천역'] = naver_data.apply(lambda row: haversine(row['위도'], row['경도'], *namchuncheon_coords), axis=1)

# 면적당 가격 계산
naver_data['면적당 월세'] = naver_data['월세'] / naver_data['면적']  # 월세 면적당 가격
naver_data['면적당 월세환산'] = naver_data['월세환산'] / naver_data['면적']  # 월세환산 면적당 가격

print(naver_data)

# 전세와 월세 데이터를 분리하여 각각 분석 및 시각화
def analyze_and_visualize_polynomial(data, distance_col, degree=2):
    # 전세와 월세 데이터 분리
    jeonse_data = data[data['전세_여부'] == 1]  # 전세
    monthly_data = data[data['전세_여부'] == 0]  # 월세
    
    def analyze_distance_to_price(subset, distance_col, price_col, label, color, degree):
        # 결측값 제거
        subset = subset.dropna(subset=[distance_col, price_col])
        
        # 독립 변수와 종속 변수
        X = subset[[distance_col]].values  # 거리
        y = subset[price_col].values       # 면적당 가격
        
        # 다항식 변환
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # 회귀 모델 학습
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        
        # 결과 출력
        print(f"=== {distance_col}와 {label}의 곡선형 상관 관계 (다항식 차수: {degree}) ===")
        print(f"[{label}] 회귀 계수: {model.coef_}, 절편: {model.intercept_:.4f}, R²: {r2:.4f}")
        
        # 산점도 및 회귀선 시각화
        plt.scatter(subset[distance_col], y, alpha=0.7, label=f"{label} (데이터)", color=color)
        sorted_idx = X[:, 0].argsort()  # 정렬된 거리 값에 따라 회귀선 정렬
        plt.plot(subset[distance_col].iloc[sorted_idx], y_pred[sorted_idx], label=f"{label} (회귀선)", color=color)
        
    # 시각화
    plt.figure(figsize=(12, 8))
    print(f"\n=== {distance_col} 기준 분석 (다항식 차수: {degree}) ===")
    
    # 전세 분석
    analyze_distance_to_price(jeonse_data, distance_col, '면적당 월세환산', '전세 (환산 월세)', 'red', degree)
    # 월세 분석
    analyze_distance_to_price(monthly_data, distance_col, '면적당 월세', '월세', 'blue', degree)
    
    # 그래프 설정
    plt.xlabel("거리 (km)")
    plt.ylabel("면적당 가격 (만원/㎡)")
    plt.title(f"{distance_col}와 면적당 가격 (전세 vs 월세, 곡선형)")
    plt.legend()
    plt.grid()
    plt.show()

# 강원대학교 기준 분석
analyze_and_visualize_polynomial(naver_data, '거리_강원대', degree=3)

# 남춘천역 기준 분석
analyze_and_visualize_polynomial(naver_data, '거리_남춘천역', degree=3)
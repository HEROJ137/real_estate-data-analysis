import os
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
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

# 전세 보증금 문자열을 숫자로 변환하는 함수
def convert_deposit(price_str):
    price_str = price_str.replace('전세', '').replace(',', '').strip()  # '전세'와 쉼표 제거
    if '억' in price_str:
        parts = price_str.split('억')
        main = int(parts[0]) * 10000  # 억 단위를 만 단위로 변환
        sub = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return main + sub
    elif price_str.isdigit():
        return int(price_str)
    else:
        raise ValueError(f"Invalid price format: {price_str}")

# 두 좌표 간 거리 계산 (Haversine formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# 데이터 읽기 및 처리
all_data = []
for file in os.listdir(naver_data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(naver_data_dir, file)
        data = pd.read_csv(file_path)
        all_data.append(data)

# 모든 데이터를 하나로 합치기
df = pd.concat(all_data, ignore_index=True)

# 데이터 전처리
df['전용면적'] = df['면적'].str.split('/').str[1].str.replace('m²', '').replace('-', np.nan).astype(float)
df['월세'] = df['가격'].str.extract(r'/(\d+)')[0].astype(float)
df = df.dropna(subset=['전용면적', '위도', '경도'])

# '-'나 NaN인 전용면적은 제외
df = df.dropna(subset=['전용면적'])

# 중복 제거: 모든 컬럼 기준으로 중복된 행 제거
df = df.drop_duplicates()

# 거리와 월세 계산
results = []
for _, row in df.iterrows():
    distance = calculate_distance(hanbit_lat, hanbit_lon, row['위도'], row['경도'])
    if '전세' in row['가격']:
        deposit = convert_deposit(row['가격'])  # 수정된 부분
        monthly_rent = convert_to_monthly_rent(deposit, datetime(2024, 10, 1))
        rent_type = '전세 환산'
    else:
        monthly_rent = row['월세']
        rent_type = '월세'
    rent_per_10m2 = (monthly_rent / row['전용면적']) * 10
    results.append({
        '거리': distance,
        '10m²당 월세': rent_per_10m2,
        '월세 유형': rent_type,
        '월세 금액': monthly_rent,
        '전용면적': row['전용면적'],
        '위도': row['위도'],
        '경도': row['경도']
    })


# 결과를 데이터프레임으로 변환
result_df = pd.DataFrame(results)

# 중복 제거: 모든 컬럼 기준으로 중복된 행 제거
result_df = result_df.drop_duplicates()

# GeoDataFrame 변환
gdf = gpd.GeoDataFrame(
    result_df, geometry=gpd.points_from_xy(result_df['경도'], result_df['위도']), crs="EPSG:4326"
)

# 좌표계 변환 (Web Mercator)
gdf = gdf.to_crs(epsg=3857)

# 지도 그리기
fig, ax = plt.subplots(figsize=(12, 10))

# 정규화 함수
def normalize(series, scale):
    return (series - series.min()) / (series.max() - series.min()) * scale

# 월세와 전세 환산 구분 시각화
for rent_type, color in [('월세', 'green'), ('전세 환산', 'blue')]:
    subset = gdf[gdf['월세 유형'] == rent_type]
    normalized_size = normalize(subset['10m²당 월세'], scale=200)  # 크기를 0~100으로 정규화
    subset.plot(
        ax=ax,
        color=color,
        alpha=0.2,  # 투명도 0.1
        markersize=normalized_size,  # 정규화된 값으로 마커 크기 조정
        label=rent_type
    )

# 강원대 한빛관 위치 강조
center_x, center_y = gpd.GeoSeries.from_xy([hanbit_lon], [hanbit_lat], crs="EPSG:4326").to_crs(epsg=3857).geometry[0].x, \
                     gpd.GeoSeries.from_xy([hanbit_lon], [hanbit_lat], crs="EPSG:4326").to_crs(epsg=3857).geometry[0].y
ax.scatter(center_x, center_y, color='red', alpha=0.5, label='강원대 한빛관', s=150)

# 폰트 크기 설정
plt.rcParams.update({
    'font.size': 6,  # 기본 폰트 크기
    'axes.titlesize': 12,  # 제목 폰트 크기
    'axes.labelsize': 8,  # 축 레이블 폰트 크기
    'xtick.labelsize': 6,  # x축 눈금 폰트 크기
    'ytick.labelsize': 6,  # y축 눈금 폰트 크기
    'legend.fontsize': 8,  # 범례 폰트 크기
    'figure.titlesize': 8  # 전체 Figure 제목 크기
})

# 배경 지도 추가 (OpenStreetMap)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15)

# 제목 및 범례
plt.title("강원대 주변 매물 지도", fontsize=10)  # 제목 폰트 크기 설정
plt.legend(fontsize=10)  # 범례 폰트 크기 설정
plt.savefig("kangwon_university_map_with_rent_types.png", dpi=300, bbox_inches='tight')
plt.show()

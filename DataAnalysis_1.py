import os
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 디렉토리 경로
naver_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data'
molit_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/molit_data'

# 기준금리 변화 CSV 파일 불러오기
rate_data = pd.read_csv('interest_rate_changes.csv')

# ***************************************************************************************************************************************
# 부동산 시세 변동 추이를 분석
# 특정 기간의 실거래가를 분석하면 월별, 연도별 가격 변화를 확인

# 이를 통해서
#   1. 부동산 시장에서의 계절적 요인(성수기/비수기)이 존재하는지 확인 -> SARIMA모델을 통한 시계열 분석으로 추세선 그리기 
#   2. 과거 시세와 현재 시세를 비교하여 가격의 변동성을 분석, 향후 몇 개월간의 가격 변화를 예측 -> Polynomial Regression을 통한 3차 다항 회귀 분석으로 추세선 그리기
# ***************************************************************************************************************************************


# molit_data의 모든 CSV 파일의 계약일을 YYYY-MM-DD 형식으로 변환하여 묶음. (연면적(㎡) 기준으로 30㎡ 이하와 30㎡ 초과 60㎡ 미만으로 분리)
def load_and_adjust_molit_data(directory, city_name=None, start_year=None, end_year=None):
    dataframes_30_under, dataframes_30_60 = [], []
    
    for file in os.listdir(directory):
        # city_name이 주어진 경우 해당 지명이 포함된 파일만 선택
        if file.endswith('.csv') and (city_name is None or city_name in file):
            year = file.split('-')[0]
            df = pd.read_csv(os.path.join(directory, file))
            df['계약일'] = df['계약일'].apply(lambda x: f"{year}-{x.replace('/', '-')}" if isinstance(x, str) else x)
            df['계약일'] = pd.to_datetime(df['계약일'], errors='coerce')  # 날짜 형식으로 변환
            
            # start_year와 end_year 기준으로 필터링
            if start_year is not None and end_year is not None:
                df = df[(df['계약일'].dt.year >= start_year) & (df['계약일'].dt.year <= end_year)]
            
            # 연면적 기준으로 30㎡ 이하와 30㎡ 초과 60㎡ 미만으로 분리
            df_30_under = df[df['연면적(㎡)'] <= 30]
            df_30_60 = df[(df['연면적(㎡)'] > 30) & (df['연면적(㎡)'] < 60)]
            
            dataframes_30_under.append(df_30_under)
            dataframes_30_60.append(df_30_60)
    
    # 각각의 데이터프레임을 하나로 병합
    combined_df_30_under = pd.concat(dataframes_30_under, ignore_index=True)
    combined_df_30_60 = pd.concat(dataframes_30_60, ignore_index=True)
    
    return combined_df_30_under, combined_df_30_60


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


# 월별로 전세와 월세를 면적당 가격으로 변환한 후, 월별 평균을 계산하여 반환
def calculate_monthly_avg(converted_molit_data):
    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 전세와 월세 데이터 분리
    jeonse_data = converted_molit_data[(converted_molit_data['월세(만원)'] == 0) | (converted_molit_data['월세(만원)'].isna())]
    wolse_data = converted_molit_data[converted_molit_data['월세(만원)'] > 0]

    # 전세를 월세로 환산한 값 추가
    jeonse_data['월세로 환산'] = jeonse_data.apply(lambda row: convert_to_monthly_rent(
        float(row['보증금(만원)'].replace(",", "")), row['계약일']), axis=1)

    # 면적당 월세 계산
    jeonse_data['면적당 월세(전세환산)'] = jeonse_data['월세로 환산'] / jeonse_data['연면적(㎡)']
    wolse_data['면적당 월세'] = wolse_data['월세(만원)'].astype(float) / wolse_data['연면적(㎡)']

    # 계약일의 월 정보 추출
    jeonse_data['계약월'] = jeonse_data['계약일'].dt.to_period('M')
    wolse_data['계약월'] = wolse_data['계약일'].dt.to_period('M')

    # 월별 평균 면적당 월세 계산
    jeonse_monthly_avg = jeonse_data.groupby('계약월')['면적당 월세(전세환산)'].mean() * 10
    wolse_monthly_avg = wolse_data.groupby('계약월')['면적당 월세'].mean() * 10

    # 월별 평균 데이터를 DataFrame으로 변환
    monthly_avg_df = pd.DataFrame({
        '전세 환산 월별 평균 면적당 월세': jeonse_monthly_avg,
        '실제 월세 월별 평균 면적당 월세': wolse_monthly_avg
    }).dropna()  # NaN 값 제거
    return monthly_avg_df

# ***************************************************************************************************************************************
# 국토교통부에서 가지고 온 실거래가 데이터를 바탕으로 월별, 연도별 가격 변화를 확인한다.
# 전세는 월세로 변환하여 계산하고 확인한다.
# ***************************************************************************************************************************************


# YYYY-MM 별 평균값 막대 그래프 ( 평균 면적당 월세 (만원/10㎡) - YYYY-MM )
def DataAnalysis_1A(converted_molit_data, rate_data):
    # 날짜 변환 함수 (최신 기준금리를 기준으로 적용하기 위해 역순 정렬)
    rate_data['변경일자'] = pd.to_datetime(rate_data['변경일자'])
    rate_data = rate_data.sort_values(by='변경일자', ascending=False)

    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 전세와 월세 데이터 분리
    jeonse_data = converted_molit_data[(converted_molit_data['월세(만원)'] == 0) | (converted_molit_data['월세(만원)'].isna())]
    wolse_data = converted_molit_data[converted_molit_data['월세(만원)'] > 0]

    # 전세를 월세로 변환한 값 추가
    jeonse_data['월세로 환산'] = jeonse_data.apply(lambda row: convert_to_monthly_rent(
        float(row['보증금(만원)'].replace(",", "")), row['계약일']), axis=1)

    # 면적당 월세 계산
    jeonse_data['면적당 월세(전세환산)'] = jeonse_data['월세로 환산'] / jeonse_data['연면적(㎡)']
    wolse_data['면적당 월세'] = wolse_data['월세(만원)'].astype(float) / wolse_data['연면적(㎡)']

    # 계약일의 월 정보 추출
    jeonse_data['계약월'] = jeonse_data['계약일'].dt.to_period('M')
    wolse_data['계약월'] = wolse_data['계약일'].dt.to_period('M')

    # 월별 평균 면적당 월세 계산
    jeonse_monthly_avg = jeonse_data.groupby('계약월')['면적당 월세(전세환산)'].mean() * 10
    wolse_monthly_avg = wolse_data.groupby('계약월')['면적당 월세'].mean() * 10

    # 월별 평균 데이터를 DataFrame으로 변환
    monthly_avg_df = pd.DataFrame({
        '전세 환산 월별 평균 면적당 월세': jeonse_monthly_avg,
        '실제 월세 월별 평균 면적당 월세': wolse_monthly_avg
    })

    # NaN 값 제거 (필요할 경우)
    monthly_avg_df = monthly_avg_df.dropna()

    # 시각화
    ax = monthly_avg_df.plot.bar(figsize=(12, 6), alpha=0.6)

    # 그래프 설정
    plt.xlabel('계약 연도')
    plt.ylabel('평균 면적당 월세 (만원/10㎡)')
    plt.title('월 평균 면적당 월세 추이 (전세 환산 및 실제 월세)')
    plt.legend()
    
    # x축을 연도별로 표시
    ax.set_xticks(range(0, len(monthly_avg_df.index), 12))  # 매년 첫 달에 해당하는 인덱스만 표시
    ax.set_xticklabels([date.year for date in monthly_avg_df.index[::12]], rotation=0)  # 연도만 표시하고 회전 없이 설정
    
    plt.show()

# YYYY-MM 별 박스플롯 그래프 ( 면적당 월세 (만원/10㎡) - YYYY-MM )
def DataAnalysis_1B(converted_molit_data, rate_data):
    # 날짜 변환 함수 (최신 기준금리를 기준으로 적용하기 위해 역순 정렬)
    rate_data['변경일자'] = pd.to_datetime(rate_data['변경일자'])
    rate_data = rate_data.sort_values(by='변경일자', ascending=False)

    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 전세와 월세 데이터 분리
    jeonse_data = converted_molit_data[(converted_molit_data['월세(만원)'] == 0) | (converted_molit_data['월세(만원)'].isna())]
    wolse_data = converted_molit_data[converted_molit_data['월세(만원)'] > 0]

    # 전세를 월세로 환산한 값 추가
    jeonse_data['월세로 환산'] = jeonse_data.apply(lambda row: convert_to_monthly_rent(
        float(row['보증금(만원)'].replace(",", "")), row['계약일']), axis=1)

    # 면적당 월세 계산 및 10㎡당 월세로 변환
    jeonse_data['면적당 월세'] = (jeonse_data['월세로 환산'] / jeonse_data['연면적(㎡)']) * 10
    wolse_data['면적당 월세'] = (wolse_data['월세(만원)'].astype(float) / wolse_data['연면적(㎡)']) * 10

    # 계약일의 월 정보 추출 (datetime 형식으로 변환)
    jeonse_data['계약월'] = jeonse_data['계약일'].dt.to_period('M').dt.to_timestamp()
    wolse_data['계약월'] = wolse_data['계약일'].dt.to_period('M').dt.to_timestamp()

    # 10㎡당 월세가 1 이하인 데이터 제외
    jeonse_data = jeonse_data[jeonse_data['면적당 월세'] > 1]
    wolse_data = wolse_data[wolse_data['면적당 월세'] > 1]

    # 전세 데이터 박스플롯
    plt.figure(figsize=(14, 7))
    grouped_jeonse = [group['면적당 월세'].values for _, group in jeonse_data.groupby('계약월')]
    jeonse_labels = jeonse_data['계약월'].dt.strftime('%Y-%m').sort_values().unique()
    box = plt.boxplot(grouped_jeonse, labels=jeonse_labels, showfliers=False, patch_artist=True)
    
    for patch in box['boxes']:
        patch.set_facecolor('lightgrey')
    for median in box['medians']:
        median.set_color('red')
    
    plt.xlabel('계약 월')
    plt.ylabel('면적당 월세 (만원/10㎡)')
    plt.title('<전세> 연도별 면적당 월세 (박스플롯)')
    plt.xticks(ticks=range(0, len(jeonse_labels), 12), labels=jeonse_labels[::12], rotation=0)
    plt.xlim(-1, len(jeonse_labels) + 2)
    plt.tight_layout()
    plt.show()

    # 월세 데이터 박스플롯
    plt.figure(figsize=(14, 7))
    grouped_wolse = [group['면적당 월세'].values for _, group in wolse_data.groupby('계약월')]
    wolse_labels = wolse_data['계약월'].dt.strftime('%Y-%m').sort_values().unique()
    box = plt.boxplot(grouped_wolse, labels=wolse_labels, showfliers=False, patch_artist=True)
    
    for patch in box['boxes']:
        patch.set_facecolor('lightgrey')
    for median in box['medians']:
        median.set_color('red')
    
    plt.xlabel('계약 월')
    plt.ylabel('면적당 월세 (만원/10㎡)')
    plt.title('<월세> 연도별 면적당 월세 (박스플롯)')
    plt.xticks(ticks=range(0, len(wolse_labels), 12), labels=wolse_labels[::12], rotation=0)
    plt.xlim(-1, len(wolse_labels) + 2)
    plt.tight_layout()
    plt.show()

# MM 별 박스플롯 그래프 ( 면적당 월세 (만원/10㎡) - MM )
def DataAnalysis_1C(converted_molit_data):
    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 면적당 월세 계산 (10㎡당 월세)
    converted_molit_data['면적당 월세'] = np.where(
        converted_molit_data['월세(만원)'] > 0,
        converted_molit_data['월세(만원)'].astype(float) / converted_molit_data['연면적(㎡)'] * 10,
        converted_molit_data.apply(lambda row: convert_to_monthly_rent(
            float(row['보증금(만원)'].replace(",", "")), row['계약일']) / row['연면적(㎡)'] * 10, axis=1)
    )

    # 10㎡당 월세가 1 이하인 데이터 제외
    converted_molit_data = converted_molit_data[converted_molit_data['면적당 월세'] > 1]

    # 계약 월 정보 추출 (월 단위)
    converted_molit_data['계약월'] = converted_molit_data['계약일'].dt.month

    # 월별 데이터를 리스트로 변환하여 박스플롯에 사용
    monthly_data = [converted_molit_data[converted_molit_data['계약월'] == month]['면적당 월세'].values for month in range(1, 13)]

    # 박스플롯 시각화
    plt.figure(figsize=(14, 7))
    box = plt.boxplot(monthly_data, labels=[f"{month}월" for month in range(1, 13)], showfliers=False, patch_artist=True)

    # 박스 색상을 라이트그레이로 설정
    for patch in box['boxes']:
        patch.set_facecolor('lightgrey')

    # 중앙값을 빨간색으로 설정
    for median in box['medians']:
        median.set_color('red')

    # 그래프 설정
    plt.xlabel('월')
    plt.ylabel('면적당 월세 (만원/10㎡)')
    plt.title('월별 면적당 월세 (박스플롯)')
    plt.tight_layout()
    plt.show()

# YYYY-MM 별 꺽은선 그래프 그래프 ( 평균 면적당 월세 (만원/10㎡) - YYYY-MM )
def DataAnalysis_1D(converted_molit_data, rate_data):
    # 날짜 변환 함수 (최신 기준금리를 기준으로 적용하기 위해 역순 정렬)
    rate_data['변경일자'] = pd.to_datetime(rate_data['변경일자'])
    rate_data = rate_data.sort_values(by='변경일자', ascending=False)

    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 전세와 월세 데이터 분리
    jeonse_data = converted_molit_data[(converted_molit_data['월세(만원)'] == 0) | (converted_molit_data['월세(만원)'].isna())]
    wolse_data = converted_molit_data[converted_molit_data['월세(만원)'] > 0]

    # 전세를 월세로 환산한 값 추가
    jeonse_data['월세로 환산'] = jeonse_data.apply(lambda row: convert_to_monthly_rent(
        float(row['보증금(만원)'].replace(",", "")), row['계약일']), axis=1)

    # 면적당 월세 계산
    jeonse_data['면적당 월세(전세환산)'] = jeonse_data['월세로 환산'] / jeonse_data['연면적(㎡)']
    wolse_data['면적당 월세'] = wolse_data['월세(만원)'].astype(float) / wolse_data['연면적(㎡)']

    # 계약일의 월 정보 추출
    jeonse_data['계약월'] = jeonse_data['계약일'].dt.to_period('M')
    wolse_data['계약월'] = wolse_data['계약일'].dt.to_period('M')

    # 월별 평균 면적당 월세 계산
    jeonse_monthly_avg = jeonse_data.groupby('계약월')['면적당 월세(전세환산)'].mean() * 10
    wolse_monthly_avg = wolse_data.groupby('계약월')['면적당 월세'].mean() * 10

    # 월별 평균 데이터를 DataFrame으로 변환
    monthly_avg_df = pd.DataFrame({
        '전세 환산 월별 평균 면적당 월세': jeonse_monthly_avg,
        '실제 월세 월별 평균 면적당 월세': wolse_monthly_avg
    })

    # NaN 값 제거 (필요할 경우)
    monthly_avg_df = monthly_avg_df.dropna()

    # x축 범위를 계약월 최솟값과 최댓값으로 설정, 매년 1월만 표시
    min_date = monthly_avg_df.index.min().to_timestamp()
    max_date = monthly_avg_df.index.max().to_timestamp()
    date_range = pd.date_range(start=min_date, end=max_date, freq='AS')  # 매년 1월만 표시
    date_range_with_margin = pd.date_range(start=min_date - pd.DateOffset(months=3), 
                                           end=max_date + pd.DateOffset(months=6), freq='MS')  # 여백 추가

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 실제 데이터 플롯
    ax.plot(monthly_avg_df.index.to_timestamp(), monthly_avg_df['전세 환산 월별 평균 면적당 월세'], label='전세 환산 월별 평균 면적당 월세', marker='o', alpha=0.6)
    ax.plot(monthly_avg_df.index.to_timestamp(), monthly_avg_df['실제 월세 월별 평균 면적당 월세'], label='실제 월세 월별 평균 면적당 월세', marker='o', alpha=0.6)

    # 그래프 설정
    plt.xlabel('계약 연도')
    plt.ylabel('평균 면적당 월세 (만원/10㎡)')
    plt.title('월 평균 면적당 월세 추이 (전세 환산 및 실제 월세)')
    plt.legend()

    # x축 설정: 매년 1월 표시 및 여백 추가
    ax.set_xlim(date_range_with_margin[0], date_range_with_margin[-1])
    ax.set_xticks(date_range)
    ax.set_xticklabels(date_range.strftime('%Y-%m'), rotation=0)

    plt.tight_layout()
    plt.show()

# YYYY-MM 별 매물 갯수 그래프 ( 매물 갯수 - YYYY-MM )
def DataAnalysis_1E(converted_molit_data, rate_data):
    # 날짜 변환 함수 (최신 기준금리를 기준으로 적용하기 위해 역순 정렬)
    rate_data['변경일자'] = pd.to_datetime(rate_data['변경일자'])
    rate_data = rate_data.sort_values(by='변경일자', ascending=False)

    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 전세와 월세 데이터 분리
    jeonse_data = converted_molit_data[(converted_molit_data['월세(만원)'] == 0) | (converted_molit_data['월세(만원)'].isna())]
    wolse_data = converted_molit_data[converted_molit_data['월세(만원)'] > 0]

    # 계약일의 월 정보 추출
    jeonse_data['계약월'] = jeonse_data['계약일'].dt.to_period('M')
    wolse_data['계약월'] = wolse_data['계약일'].dt.to_period('M')

    # 월별 매물 갯수 계산
    jeonse_monthly_count = jeonse_data.groupby('계약월').size()
    wolse_monthly_count = wolse_data.groupby('계약월').size()

    # 월별 매물 갯수를 DataFrame으로 변환
    monthly_count_df = pd.DataFrame({
        '전세 매물 갯수': jeonse_monthly_count,
        '월세 매물 갯수': wolse_monthly_count
    })

    # NaN 값 제거 (필요할 경우)
    monthly_count_df = monthly_count_df.dropna()

    # 시각화
    ax = monthly_count_df.plot.bar(figsize=(12, 6), alpha=0.6)

    # 그래프 설정
    plt.xlabel('계약 연도')
    plt.ylabel('매물 갯수')
    plt.title('월별 매물 갯수 추이 (전세 및 월세)')
    plt.legend()
    
    # x축을 연도별로 표시
    ax.set_xticks(range(0, len(monthly_count_df.index), 12))  # 매년 첫 달에 해당하는 인덱스만 표시
    ax.set_xticklabels([date.year for date in monthly_count_df.index[::12]], rotation=0)  # 연도만 표시하고 회전 없이 설정
    
    plt.show()


# YYYY-MM 별 꺽은선 그래프 그래프 ( 평균 면적당 월세 (만원/10㎡) - YYYY-MM )
# Polynomial Regression을 적용하여 추세선 추가 ( 3차 다항 회귀 분석 )
def DataAnalysis_1Aa(converted_molit_data, rate_data):
    # 날짜 변환 함수 (최신 기준금리를 기준으로 적용하기 위해 역순 정렬)
    rate_data['변경일자'] = pd.to_datetime(rate_data['변경일자'])
    rate_data = rate_data.sort_values(by='변경일자', ascending=False)

    # 계약일을 datetime으로 변환
    converted_molit_data['계약일'] = pd.to_datetime(converted_molit_data['계약일'], errors='coerce')

    # 전세와 월세 데이터 분리
    jeonse_data = converted_molit_data[(converted_molit_data['월세(만원)'] == 0) | (converted_molit_data['월세(만원)'].isna())]
    wolse_data = converted_molit_data[converted_molit_data['월세(만원)'] > 0]

    # 전세를 월세로 환산한 값 추가
    jeonse_data['월세로 환산'] = jeonse_data.apply(lambda row: convert_to_monthly_rent(
        float(row['보증금(만원)'].replace(",", "")), row['계약일']), axis=1)

    # 면적당 월세 계산
    jeonse_data['면적당 월세(전세환산)'] = jeonse_data['월세로 환산'] / jeonse_data['연면적(㎡)']
    wolse_data['면적당 월세'] = wolse_data['월세(만원)'].astype(float) / wolse_data['연면적(㎡)']

    # 계약일의 월 정보 추출
    jeonse_data['계약월'] = jeonse_data['계약일'].dt.to_period('M')
    wolse_data['계약월'] = wolse_data['계약일'].dt.to_period('M')

    # 월별 평균 면적당 월세 계산
    jeonse_monthly_avg = jeonse_data.groupby('계약월')['면적당 월세(전세환산)'].mean() * 10
    wolse_monthly_avg = wolse_data.groupby('계약월')['면적당 월세'].mean() * 10

    # 월별 평균 데이터를 DataFrame으로 변환
    monthly_avg_df = pd.DataFrame({
        '전세 환산 월별 평균 면적당 월세': jeonse_monthly_avg,
        '실제 월세 월별 평균 면적당 월세': wolse_monthly_avg
    })

    # NaN 값 제거 (필요할 경우)
    monthly_avg_df = monthly_avg_df.dropna()

    # x축 범위를 계약월 최솟값과 최댓값으로 설정, 매년 1월만 표시
    min_date = monthly_avg_df.index.min().to_timestamp()
    max_date = monthly_avg_df.index.max().to_timestamp()
    date_range = pd.date_range(start=min_date, end=max_date, freq='AS')  # 매년 1월만 표시
    date_range_with_margin = pd.date_range(start=min_date - pd.DateOffset(months=3), 
                                           end=max_date + pd.DateOffset(months=6), freq='MS')  # 여백 추가

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 실제 데이터 플롯
    ax.plot(monthly_avg_df.index.to_timestamp(), monthly_avg_df['전세 환산 월별 평균 면적당 월세'], label='전세 환산 월별 평균 면적당 월세', marker='o', alpha=0.6)
    ax.plot(monthly_avg_df.index.to_timestamp(), monthly_avg_df['실제 월세 월별 평균 면적당 월세'], label='실제 월세 월별 평균 면적당 월세', marker='o', alpha=0.6)

    # 곡선형 추세선 (Polynomial Trend Line) 추가
    for column in monthly_avg_df.columns:
        x_values = np.arange(len(date_range_with_margin))
        
        # y_values를 날짜 범위와 일치시켜 NaN을 채움
        y_values = monthly_avg_df[column].reindex(date_range_with_margin.to_period("M")).fillna(method="ffill").fillna(method="bfill").values
        
        # Polynomial regression of degree 3
        poly = PolynomialFeatures(degree=3)
        x_poly = poly.fit_transform(x_values.reshape(-1, 1))
        model = LinearRegression().fit(x_poly, y_values)
        trend_line = model.predict(x_poly)

        # 곡선형 추세선 그리기
        ax.plot(date_range_with_margin, trend_line, linestyle='--', label=f'{column} Polynomial Trend Line')

        coefficients = model.coef_
        intercept = model.intercept_

        polynomial_expression = " + ".join([
            f"{coeff:.10f}x^{i}" if abs(coeff) > 1e-10 else ""  # 더 작은 임계값 설정
            for i, coeff in enumerate(coefficients)
        ])

        # 절편 포함 다항식 표현
        polynomial_expression = f"{intercept:.10f}" + (" + " + polynomial_expression if polynomial_expression else "")

        print(f"Polynomial Trend Line Equation: y = {polynomial_expression}")

    # 그래프 설정
    plt.xlabel('계약 연도')
    plt.ylabel('평균 면적당 월세 (만원/10㎡)')
    plt.title('월 평균 면적당 월세 추이 (전세 환산 및 실제 월세) - 곡선형 추세선 포함')
    plt.legend()

    # x축 설정: 매년 1월 표시 및 여백 추가
    ax.set_xlim(date_range_with_margin[0], date_range_with_margin[-1])
    ax.set_xticks(date_range)
    ax.set_xticklabels(date_range.strftime('%Y-%m'), rotation=0)

    plt.tight_layout()
    plt.show()


# SARIMA (Seasonal Autoregressive Integrated Moving Average;시계열 데이터의 계절적 패턴을 반영하여 예측하는 모델) 
# SARIMA는 일반 ARIMA 모델을 확장한 형태로, 계절적 성분을 추가하여 비계절 성분과 계절 성분을 모두 고려하는 방식
def DataAnalysis_1Ab(monthly_avg_df, forecast_period=12):
    # 예측 대상 컬럼
    columns_to_forecast = ['전세 환산 월별 평균 면적당 월세', '실제 월세 월별 평균 면적당 월세']
    scaler = StandardScaler()
    forecast_results = {}

    # SARIMA 모델 파라미터 설정 (p, d, q) x (P, D, Q, s)
    seasonal_order = (1, 1, 1, 12)  # 계절성 주기 12개월(1년)

    # 각 컬럼에 대해 SARIMA 예측 수행
    for column in columns_to_forecast:
        # 시계열 데이터 설정 및 스케일링
        ts_data = monthly_avg_df[column].values.reshape(-1, 1)
        ts_data_scaled = scaler.fit_transform(ts_data).flatten()
        
        try:
            # SARIMA 모델 피팅
            model = SARIMAX(ts_data_scaled, order=(1, 1, 1), seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            
            # 예측
            forecast_scaled = model_fit.forecast(steps=forecast_period)
            
            # 예측치 역변환
            forecast_original = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            
            # 예측 날짜 생성
            last_date = monthly_avg_df.index[-1].to_timestamp()
            forecast_dates = [last_date + timedelta(days=30 * i) for i in range(1, forecast_period + 1)]
            forecast_results[column] = pd.Series(forecast_original, index=forecast_dates)
            
        except Exception as e:
            print(f"Error forecasting {column}: {e}")
            continue

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'전세 환산 월별 평균 면적당 월세': 'tab:blue', '실제 월세 월별 평균 면적당 월세': 'tab:green'}

    # 계절 구분선 추가 (각 계절에 맞는 색상으로 선만 추가)
    start_year = min(monthly_avg_df.index.year.min(), 2013)
    end_year = max(monthly_avg_df.index.year.max(), 2024 + math.ceil(forecast_period / 12))
    seasons = {
        '봄': (3, 5, 'lightpink'),        # 연한 핑크색
        '여름': (6, 8, 'lightgreen'),     # 연한 초록색
        '가을': (9, 11, '#E59171'),        # 단풍색
        '겨울': (12, 2, 'lightblue')      # 연한 파란색
    }
    
    for year in range(start_year, end_year + 1):
        for season, (start_month, _, color) in seasons.items():  # 종료 월은 사용하지 않음
            # 시작 월의 날짜 생성
            start_date = pd.Timestamp(year=year, month=start_month, day=1)
            
            # 계절 시작 월에만 구분선 추가
            ax.axvline(start_date, color=color, linestyle='--', linewidth=0.5)

    # 실제 데이터와 예측 데이터 그리기
    for column in columns_to_forecast:
        # 실제 데이터
        ax.plot(monthly_avg_df.index.to_timestamp(), monthly_avg_df[column], label=f'{column} (Actual)', marker='o', color=colors[column])
        
        # 예측 데이터
        if column in forecast_results:
            ax.plot(forecast_results[column].index, forecast_results[column], label=f'{column} (Forecast)',
                    linestyle='--', marker='o', markerfacecolor='white', color=colors[column], markersize=4)
            
            # 실측치와 예측치의 첫 점을 이어주는 선 추가
            ax.plot(
                [monthly_avg_df.index[-1].to_timestamp(), forecast_results[column].index[0]],
                [monthly_avg_df[column].iloc[-1], forecast_results[column].iloc[0]],
                linestyle='--', color=colors[column], marker='o', markerfacecolor='white', markersize=4
            )

    # 그래프 설정
    plt.xlabel('계약 연도')
    plt.ylabel('평균 면적당 월세 (만원/10㎡)')
    plt.title('월 평균 면적당 월세 예측 (전세 환산 및 실제 월세) - SARIMA 모델')
    plt.legend()
    plt.tight_layout()
    
    # x축 여백 설정
    ax.set_xlim(left=monthly_avg_df.index[0].to_timestamp() - timedelta(days=60),
                right=forecast_results[columns_to_forecast[0]].index[-1] + timedelta(days=60))
    plt.show()


def InterestRateChangeGraph(rate_date):
    df = rate_date

    # 날짜 형식 변환 및 정렬
    df['변경일자'] = pd.to_datetime(df['변경일자'])
    df = df.sort_values(by='변경일자')

    # 막대 그래프 (막대 두께 조정)
    plt.figure(figsize=(12, 6))
    plt.bar(df['변경일자'], df['기준금리'], color='skyblue', alpha=1, width=35, zorder=3)  # 막대 두께 조정
    plt.xlabel('변경일자')
    plt.ylabel('기준금리 (%)')
    plt.title('기준금리 변화 추이')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # molit_data의 모든 CSV 파일을 두개의 데이터프레임으로 나눠서 변환 (연면적(㎡) 기준으로 30㎡ 이하와 30㎡ 초과 60㎡ 미만으로 분리)
    converted_molit_df_1, converted_molit_df_2 = load_and_adjust_molit_data(molit_data_dir)
    # 하나로 합친 데이터프레임
    combined_molit_df = pd.concat([converted_molit_df_1, converted_molit_df_2], axis=0, ignore_index=True)

    #InterestRateChangeGraph(pd.read_csv('interest_rate_changes.csv'))

    DataAnalysis_1E(combined_molit_df, rate_data)
    
    # Polynomial Regression을 적용하여 추세선 추가 ( 3차 다항 회귀 분석 )
    # DataAnalysis_1Aa(combined_molit_df, rate_data)

    # 월별 평균 데이터 계산
    #monthly_avg_df = calculate_monthly_avg(combined_molit_df)

    # SARIMA 예측 수행
    #DataAnalysis_1Ab(monthly_avg_df, forecast_period=18)


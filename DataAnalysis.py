# 두 번째, 강원대학교에서의 거리와 부동산 가격 간의 상관관계를 확인해 보려고 합니다. 
# 강원대학교에서의 거리와 건물 면적, 예상 거래가 등의 정보를 수집하여 3차원 그래프로 시각화하고, 
# 특정 매물이 대학과의 거리에 따라 얼마나 가격 차이를 보이는지 파악해보고자 합니다. 
# 추가로 좋은 매물들이 확인되면 주변 상권과 대중교통 정보를 고려하여 현재 부동산을 거래한다고 했을 때 어떤 매물을 선택하는 것이 좋을지 분석해보고 싶습니다.

# 마지막으로 다중 회귀 분석을 통해 면적, 거리, 건물 연식 등이 부동산 가격에 미치는 영향을 종합적으로 파악하고자 합니다. 
# 국토교통부의 실거래 데이터는 시간이 지남에 따라 건물 연식이 가격에 미치는 영향을 분석할 수 있고, 
# 네이버 부동산 데이터는 현재 시장에서의 가격과 면적의 관계를 반영할 수 있습니다. 
# 두 데이터를 결합하여 가격 대비 더 효율적인 매물을 선택하는 데 필요한 정보를 도출할 수 있을 것 같습니다.

import os
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 디렉토리 경로
naver_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data'
molit_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/molit_data'

# 기준금리 변화 CSV 파일 불러오기
rate_data = pd.read_csv('interest_rate_changes.csv')

# ************************************************************************
# 부동산 시세 변동 추이를 분석
# 특정 기간의 실거래가를 분석하면 월별, 연도별 가격 변화를 확인

# 이를 통해서
#   1. 부동산 시장에서의 계절적 요인(성수기/비수기)이 존재하는지 확인 -> 1년씩 데이터를 뽑고 유사도 산정 후 계절과 연계
#   2. 가격 상승 또는 하락이 주로 발생하는 시기를 파악 -> 위와 동일
#   3. 과거 시세와 현재 시세를 비교하여 가격의 변동성을 분석 -> 
#   4. 향후 몇 개월간의 가격 변화를 예측
# ************************************************************************

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

# 국토교통부에서 가지고 온 실거래가 데이터를 바탕으로 월별, 연도별 가격 변화를 확인한다.
# 전세는 월세로 변환하여 계산하고 확인한다.

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


def DataAnalysis_with_polynomial_trend_line(converted_molit_data, rate_data):
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




if __name__ == "__main__":
    # molit_data의 모든 CSV 파일을 두개의 데이터프레임으로 나눠서 변환 (연면적(㎡) 기준으로 30㎡ 이하와 30㎡ 초과 60㎡ 미만으로 분리)
    converted_molit_df_1, converted_molit_df_2 = load_and_adjust_molit_data(molit_data_dir)
    # 하나로 합친 데이터프레임
    combined_molit_df = pd.concat([converted_molit_df_1, converted_molit_df_2], axis=0, ignore_index=True)

    DataAnalysis_with_polynomial_trend_line(combined_molit_df, rate_data)

    # DataAnalysis_1A(converted_molit_data_1, rate_data)

    # DataAnalysis_1C(converted_molit_data_1)

    pass


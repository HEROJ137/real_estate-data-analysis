# 두 번째, 강원대학교에서의 거리와 부동산 가격 간의 상관관계를 확인해 보려고 합니다. 
# 강원대학교에서의 거리와 건물 면적, 예상 거래가 등의 정보를 수집하여 3차원 그래프로 시각화하고, 
# 특정 매물이 대학과의 거리에 따라 얼마나 가격 차이를 보이는지 파악해보고자 합니다. 
# 추가로 좋은 매물들이 확인되면 주변 상권과 대중교통 정보를 고려하여 현재 부동산을 거래한다고 했을 때 어떤 매물을 선택하는 것이 좋을지 분석해보고 싶습니다.

# 마지막으로 다중 회귀 분석을 통해 면적, 거리, 건물 연식 등이 부동산 가격에 미치는 영향을 종합적으로 파악하고자 합니다. 
# 국토교통부의 실거래 데이터는 시간이 지남에 따라 건물 연식이 가격에 미치는 영향을 분석할 수 있고, 
# 네이버 부동산 데이터는 현재 시장에서의 가격과 면적의 관계를 반영할 수 있습니다. 
# 두 데이터를 결합하여 가격 대비 더 효율적인 매물을 선택하는 데 필요한 정보를 도출할 수 있을 것 같습니다.

import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
#   1. 부동산 시장에서의 계절적 요인(성수기/비수기)이 존재하는지 확인 - (월)
#   2. 가격 상승 또는 하락이 주로 발생하는 시기를 파악
#   3. 과거 시세와 현재 시세를 비교하여 가격의 변동성을 분석
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
        if contract_date >= row['변경일자']: return row['기준금리']/100

    return rate_data.iloc[-1]['기준금리']/100  # 가장 오래된 금리 반환

# 전세에서 월세로 변환
# 전세전환율 계산 방법은 전세보증금 * ( 기준금리(%) + 대통령령에 의거한 월차임전환시산정률(%) ) / 12 
def convert_to_monthly_rent(jeonse, contract_date):
    base_rate = get_base_rate(contract_date)
    conversion_rate = 0.02 if contract_date.year >= 2021 else 0.035
    monthly_rent = jeonse * (base_rate + conversion_rate) / 12
    print(f"jeonse({jeonse}) * (base_rate({base_rate}) + conversion_rate({conversion_rate})) / 12")
    return monthly_rent

# 국토교통부에서 가지고 온 실거래가 데이터를 바탕으로 월별, 연도별 가격 변화를 확인한다.
# 전세는 월세로 변환하여 계산하고 확인한다.
def DataAnalysis_1(converted_molit_data, rate_data):
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

if __name__ == "__main__":
    # molit_data의 모든 CSV 파일을 두개의 데이터프레임으로 나눠서 변환 (연면적(㎡) 기준으로 30㎡ 이하와 30㎡ 초과 60㎡ 미만으로 분리)
    converted_molit_data_1, converted_molit_data_2 = load_and_adjust_molit_data(molit_data_dir)

    DataAnalysis_1(converted_molit_data_2, rate_data)




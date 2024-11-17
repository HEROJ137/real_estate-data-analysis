import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 디렉토리 경로
naver_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data_updated'
molit_data_dir = '/Users/jang-yeong-ung/Documents/real_estate-data-analysis/molit_data'

# 기준금리 변화 CSV 파일 불러오기
rate_data = pd.read_csv('interest_rate_changes.csv')
import pandas as pd
import googlemaps
import time
import json
import os

# 지정된 디렉토리 안의 모든 CSV 파일에서 '중개 플랫폼', '중개사', '등록일' 열을 제거하고 저장.
def clean_csv_files(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):  # CSV 파일만 처리
            file_path = os.path.join(directory_path, filename)
            
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            
            # '중개 플랫폼', '중개사', '등록일' 열 삭제
            df = df.drop(columns=['태그'], errors='ignore')
            
            # 수정된 데이터 저장 (덮어쓰기)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')  # UTF-8로 인코딩 문제 방지

    print("모든 CSV 파일에서 지정된 열을 삭제했습니다.")

# 요청량 관리 데이터를 로드하거나 초기화
def load_or_initialize_usage():
    if os.path.exists(usage_file):
        with open(usage_file, 'r') as file:
            usage_data = json.load(file)
            print(f"기존 요청 데이터 로드 완료: {usage_data}")
    else:
        usage_data = {'request_count': 0, 'date': time.strftime('%Y-%m-%d')}
        print("새로운 요청 데이터 생성.")
    return usage_data

# 날짜가 변경되었으면 요청량 초기화
def reset_usage_if_new_day(usage_data):
    current_date = time.strftime('%Y-%m-%d')
    if usage_data['date'] != current_date:
        usage_data = {'request_count': 0, 'date': current_date}
        print("날짜 변경 확인: 요청 수를 초기화합니다.")
    return usage_data

# 요청량 관리 데이터를 저장
def save_usage(usage_data):
    with open(usage_file, 'w') as file: json.dump(usage_data, file)
    print(f"요청량 데이터 저장 완료: {usage_data}")

# 주소를 위도와 경도로 변환
def get_lat_lon_google(address, usage_data):
    if usage_data['request_count'] >= DAILY_LIMIT:
        print("일일 요청량 초과! 요청 중단.")
        return None, None

    try:
        print(f"주소 처리 중: {address}")
        geocode_result = gmaps.geocode(address)
        usage_data['request_count'] += 1
        time.sleep(REQUEST_DELAY)

        save_usage(usage_data)  # 요청 수 저장

        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            print(f"변환 결과 - 위도: {location['lat']}, 경도: {location['lng']}")
            return location['lat'], location['lng']
        else:
            print("주소를 변환하지 못했습니다.")
            return None, None
    except Exception as e:
        print(f"Error for address {address}: {e}")
        return None, None

# CSV 파일을 처리하여 위도와 경도를 추가
def process_csv_file(input_file_path, output_file_path, usage_data):
    print(f"CSV 파일 처리 시작: {input_file_path}")
    df = pd.read_csv(input_file_path)
    print(f"CSV 파일 로드 완료, 총 {len(df)}개의 행.")

    latitudes = []
    longitudes = []

    for i, address in enumerate(df['소재지']):
        print(f"처리 중 ({i + 1}/{len(df)}): {address}")
        lat, lon = get_lat_lon_google(address, usage_data)

        # 요청량 초과 시 작업 중단
        if usage_data['request_count'] >= DAILY_LIMIT:
            print("일일 요청량 도달! 현재까지의 진행 상태 저장.")
            break

        latitudes.append(lat)
        longitudes.append(lon)

    # 요청량 초과로 중단되었을 경우 남은 데이터를 None으로 채움
    if len(latitudes) < len(df):
        latitudes.extend([None] * (len(df) - len(latitudes)))
        longitudes.extend([None] * (len(df) - len(longitudes)))

    # 데이터프레임 업데이트
    df['위도'] = latitudes
    df['경도'] = longitudes
    print("위도와 경도 열 추가 완료.")

    # CSV 파일 저장
    print(f"업데이트된 CSV 파일 저장 중: {output_file_path}")
    df.to_csv(output_file_path, index=False)
    print(f"파일 저장 완료: {output_file_path}")

# 지정된 디렉토리의 모든 CSV 파일 처리
def process_all_files(input_dir, output_dir):
    usage_data = load_or_initialize_usage()
    usage_data = reset_usage_if_new_day(usage_data)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 디렉토리 내 모든 CSV 파일 처리
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)
            process_csv_file(input_file_path, output_file_path, usage_data)

    # 최종 요청량 저장
    save_usage(usage_data)
    print("모든 파일 처리 완료!")

# 입력된 파일 경로에 해당하는 단일 CSV 파일을 처리하여 위도와 경도를 추가하고 저장.
def process_single_file(file_path):
    # Google Maps API 키 설정 (전역 변수로 활용)
    global gmaps, usage_file, DAILY_LIMIT, REQUEST_DELAY

    # 요청량 관리 데이터 로드
    usage_data = load_or_initialize_usage()
    usage_data = reset_usage_if_new_day(usage_data)

    # 입력 파일 경로에서 출력 파일 경로 생성
    dir_name, file_name = os.path.split(file_path)
    output_file_path = os.path.join(dir_name, f"{os.path.splitext(file_name)[0]}_updated.csv")

    # CSV 파일 처리
    process_csv_file(file_path, output_file_path, usage_data)

    # 요청량 관리 데이터 저장
    save_usage(usage_data)

    print(f"파일 처리 완료. 결과 저장 위치: {output_file_path}")

if __name__ == "__main__":
    clean_csv_files("/Users/jang-yeong-ung/Documents/real_estate-data-analysis/naver_data_updated")
import pandas as pd

def weighted_average_by_date_ranges_and_save(csv_file, date_ranges, output_prefix):
    """
    주어진 여러 날짜 구간에 대해 각 후보자의 지지율을 응답률로 가중평균 계산 후,
    구간별 결과를 time_label과 함께 CSV 파일로 저장

    Parameters:
    csv_file (str): 입력 CSV 파일 경로
    date_ranges (list of tuples): (start_date, end_date) 형식의 날짜 구간 리스트
    output_prefix (str): 저장할 CSV 파일 이름 접두사

    Returns:
    pd.DataFrame: 구간별 가중평균 결과 데이터프레임
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_file, encoding='utf-8-sig')


    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    '''
    # 응답률에서 '%' 기호 제거 후 숫자로 변환 (핵심 수정사항)
    if df['응답률'].dtype == object:
        df['응답률'] = df['응답률'].str.rstrip('%')
        df['응답률'] = pd.to_numeric(df['응답률'], errors='coerce')
    '''
    # 숫자 컬럼들을 numeric 타입으로 변환
    numeric_columns = ['이재명', '김문수', '이준석', '응답률']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # NaN 값이 있는 행 제거
    df = df.dropna(subset=numeric_columns)

    results = []

    for i, (start_date, end_date) in enumerate(date_ranges, start=1):
        # 날짜 구간으로 필터링
        mask = (df['날짜'] >= pd.to_datetime(start_date)) & (df['날짜'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            print(f"구간 {i} ({start_date} ~ {end_date})에 해당하는 데이터가 없습니다.")
            continue

        # 가중평균 계산
        total_weight = filtered_df['응답률'].sum()
        
        if total_weight == 0:
            print(f"구간 {i}의 총 응답률이 0입니다.")
            continue
            
        weighted_avg = {}
        
        for candidate in ['이재명', '김문수', '이준석']:
            if candidate in filtered_df.columns:
                weighted_sum = (filtered_df[candidate] * filtered_df['응답률']).sum()
                weighted_avg[candidate] = weighted_sum / total_weight
            else:
                print(f"'{candidate}' 컬럼이 데이터에 없습니다.")
                weighted_avg[candidate] = None

        # time_label 추가
        weighted_avg['time_label'] = i
        results.append(weighted_avg)

    if not results:
        print("처리할 수 있는 데이터가 없습니다.")
        return pd.DataFrame()

    # 결과를 데이터프레임으로 변환 후 지정된 순서로 컬럼 정렬
    result_df = pd.DataFrame(results)
    result_df = result_df[['이재명', '김문수', '이준석', 'time_label']]

    # CSV 파일로 저장
    output_file = f"{output_prefix}_{date_ranges}.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"{output_file} 파일로 저장되었습니다.")

    return result_df

# 사용 예시
def example_usage():
    # 날짜 구간 설정 예시
    date_ranges = [
        ('2025-05-19', '2025-05-21'),  # (time_label = 1)
        ('2025-05-24', '2025-05-26'),  # (time_label = 2)
        ('2025-05-28', '2025-05-30')  # (time_label = 3)
    ]
    
    # 함수 실행
    result_df = weighted_average_by_date_ranges_and_save(
        csv_file='21대_대선_여론조사_utf8.csv',
        date_ranges=date_ranges,
        output_prefix='weighted_average'
    )
    
    return result_df

# 실행
if __name__ == "__main__":
    result = example_usage()
    print("\n결과:")
    print(result)

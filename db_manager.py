import sqlite3
import pandas as pd
import os
from datetime import datetime

print("db_manager.py 모듈이 성공적으로 로드되었습니다!")

DB_NAME = 'dashboard_data.db' # 데이터베이스 파일 이름
REALTIME_TABLE_NAME = 'realtime_predictions'
HANDOVER_COMMENTS_TABLE = 'handover_comments'
HANDOVER_CHECKLIST_TABLE = 'handover_checklist'

def init_db():
    """데이터베이스 파일이 없으면 생성하고, 필요한 테이블들을 초기화합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 실시간 예측 데이터 테이블
    # 기존 app.py의 numeric_cols와 categorical_cols를 기반으로 컬럼을 정의합니다.
    # 예시 컬럼: 실제 CSV의 모든 컬럼 + 예측 결과, 확률 등을 추가합니다.
    # 모델 예측에 사용되는 다른 숫자/범주형 컬럼들 추가
    # 예: count, working, molten_volume, upper_mold_temp3, lower_mold_temp3, registration_time 등
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {REALTIME_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            n_intervals INTEGER,
            cast_pressure REAL,
            lower_mold_temp2 REAL,
            low_section_speed REAL,
            upper_mold_temp1 REAL,
            upper_mold_temp2 REAL,
            sleeve_temperature REAL,
            lower_mold_temp1 REAL,
            high_section_speed REAL,

            count INTEGER,
            working REAL,
            molten_volume REAL,
            upper_mold_temp3 REAL,
            lower_mold_temp3 REAL,
            registration_time TEXT,
            
            prediction INTEGER,
            probability REAL,

            line TEXT,
            name TEXT,
            mold_name TEXT,
            time TEXT,
            date TEXT,
            passorfail INTEGER -- 원본 데이터의 실제 passorfail (옵션)
        )
    ''')

    # 인수인계 페이지 - 코멘트 테이블
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {HANDOVER_COMMENTS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            comment TEXT
        )
    ''')

    # 인수인계 페이지 - 체크리스트 테이블 (항목별 상태 유지)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {HANDOVER_CHECKLIST_TABLE} (
            item_id TEXT PRIMARY KEY,
            status INTEGER, -- 0: unchecked, 1: checked
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 체크리스트 기본 항목 삽입 (초기 상태 설정)
    # 실제 체크리스트 항목이 변경될 경우 이 부분을 업데이트해야 합니다.
    checklist_items = ['task_1', 'task_2', 'task_3', 'task_4']
    for item in checklist_items:
        cursor.execute(f"INSERT OR IGNORE INTO {HANDOVER_CHECKLIST_TABLE} (item_id, status) VALUES (?, 0)", (item,))

    conn.commit()
    conn.close()
    print(f"✅ 데이터베이스 '{DB_NAME}' 초기화 및 테이블 생성 완료.")

def is_db_empty():
    """REALTIME_TABLE_NAME 테이블이 비어있는지 확인합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {REALTIME_TABLE_NAME}")
    count = cursor.fetchone()[0]
    conn.close()
    return count == 0

def load_all_csvs_to_db(csv_folder_path, df_all_columns):
    """지정된 폴더의 모든 CSV 파일을 읽어 데이터베이스에 적재합니다."""
    init_db() # 테이블이 없으면 생성

    # ✅ 데이터베이스에 이미 데이터가 있는지 확인하고, 있다면 로드를 건너뜁니다.
    if not is_db_empty():
        print("✨ 데이터베이스에 이미 데이터가 존재하여 CSV 로드를 건너뜝니다.")
        return 

    conn = sqlite3.connect(DB_NAME)
    
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    print(f"✨ {len(csv_files)}개의 CSV 파일 로드 시작...")

    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(csv_folder_path, csv_file)
        try:
            # low_memory=False로 설정하여 경고 메시지 방지
            df = pd.read_csv(file_path, low_memory=False)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.isoformat()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.isoformat()
            if 'registration_time' in df.columns:
                df['registration_time'] = pd.to_datetime(df['registration_time'], errors='coerce').dt.isoformat()
            
            # `id` 컬럼이 DB에서 AUTOINCREMENT이므로, DataFrame에 있으면 삭제합니다.
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            
            # CSV 파일의 모든 컬럼을 삽입 시도 (DB 스키마가 정확히 일치해야 함)
            df.to_sql(REALTIME_TABLE_NAME, conn, if_exists='append', index=False)

            if (i + 1) % 1000 == 0: # 1000개 파일마다 진행 상황 출력
                print(f"진행 중: {i + 1}/{len(csv_files)} 파일 로드 완료.")

        except Exception as e:
            print(f"❌ 오류 발생 중 {csv_file} 로드: {e}")
            continue
    conn.close()
    print("✅ 모든 CSV 데이터 데이터베이스 로드 완료.")

def get_last_n_rows_from_db(n):
    """데이터베이스에서 최신 n개의 예측 데이터를 가져옵니다."""
    conn = sqlite3.connect(DB_NAME)
    # ORDER BY n_intervals DESC LIMIT n으로 변경하여 더 정확한 최신 데이터 확보
    query = f"SELECT * FROM {REALTIME_TABLE_NAME} ORDER BY n_intervals DESC LIMIT {n}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.iloc[::-1] # 오래된 데이터부터 보이도록 순서 뒤집기

def add_prediction_to_db(row_data):
    """실시간 예측 결과를 데이터베이스에 추가합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # row_data는 딕셔너리 형태여야 합니다.
    # REALTIME_TABLE_NAME의 모든 컬럼에 맞춰 데이터를 준비합니다.
    columns = ", ".join(row_data.keys())
    placeholders = ", ".join("?" * len(row_data))
    
    try:
        cursor.execute(f"INSERT INTO {REALTIME_TABLE_NAME} ({columns}) VALUES ({placeholders})", tuple(row_data.values()))
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"DB insertion error (likely duplicate n_intervals): {e}")
    finally:
        conn.close()

def get_handover_comments():
    """데이터베이스에서 모든 코멘트를 가져옵니다."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT timestamp, comment FROM {HANDOVER_COMMENTS_TABLE} ORDER BY timestamp DESC", conn)
    conn.close()
    return df.to_dict('records') # [{'timestamp': '...', 'comment': '...'}] 형태로 반환

def add_handover_comment(comment):
    """새로운 코멘트를 데이터베이스에 추가합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(f"INSERT INTO {HANDOVER_COMMENTS_TABLE} (timestamp, comment) VALUES (?, ?)", (timestamp, comment))
    conn.commit()
    conn.close()

def get_checklist_status():
    """데이터베이스에서 체크리스트 상태를 가져옵니다."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT item_id, status FROM {HANDOVER_CHECKLIST_TABLE}", conn)
    conn.close()
    return df.set_index('item_id')['status'].to_dict() # {'task_1': 0, 'task_2': 1, ...} 형태로 반환

def update_checklist_status(item_id, status):
    """체크리스트 항목의 상태를 업데이트합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(f"REPLACE INTO {HANDOVER_CHECKLIST_TABLE} (item_id, status, last_updated) VALUES (?, ?, ?)", (item_id, status, timestamp))
    conn.commit()
    conn.close()
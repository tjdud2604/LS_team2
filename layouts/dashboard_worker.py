# index.py

from dash import html, dcc, Output, Input, State, callback_context, no_update
import pandas as pd
import plotly.graph_objs as go
import joblib
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
import os # os 모듈 추가
import traceback # 상세 트레이스백 출력을 위해 추가

# --- 파일 경로 및 데이터/모델 로드 설정 ---
# 현재 스크립트(index.py)의 디렉토리 경로를 가져옵니다.
# Render 환경에서 파일 경로를 안정적으로 찾기 위해 사용됩니다.
current_dir = os.path.dirname(__file__)

# data 폴더가 index.py와 같은 레벨(프로젝트 루트)에 있다고 가정하고 경로를 설정합니다.
DATA_FOLDER = os.path.join(current_dir, "data")

MODEL_PATH = os.path.join(DATA_FOLDER, "model_voting.pkl")
DATA_PATH = os.path.join(DATA_FOLDER, "train.csv") # 실제 사용하는 데이터 파일명으로 변경하세요.

# 모델 예측에 필요 없는 컬럼들 정의
DROP_COLS = ['tryshot_signal', 'heating_furnace']
EXCLUDE_COLS = ['id', 'passorfail', 'timestamp', 'time', 'date', 'name', 'line', 'mold_name']

# --- 전역 변수로 모델과 데이터프레임 초기화 ---
# 로딩 실패 시 앱이 완전히 중단되지 않도록 빈 값으로 미리 설정
tuned_vote = None
df_all = pd.DataFrame()
numeric_cols = []
categorical_cols = []

# --- 모델과 데이터 로드 (시작 시점에 실행) ---
# 이 부분의 print문은 Render 로그에 출력되므로, 로딩 성공/실패 여부 파악에 유용합니다.
try:
    # 모델 로드 시도
    tuned_vote = joblib.load(MODEL_PATH)
    print(f"✅ 모델 로드 성공: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    print("경로를 확인하거나, 'data' 폴더가 Git에 커밋되었는지 확인하세요.")
except Exception as e:
    print(f"❌ 오류: 모델 로드 중 예외 발생: {e}")
    print(traceback.format_exc())

try:
    # 데이터 로드 시도
    df_all = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"✅ 데이터 로드 성공: {DATA_PATH}, Shape: {df_all.shape}")
    print(f"데이터 프레임이 비어있나요? {df_all.empty}")

    # 데이터 로드 성공 후 컬럼 정의 (df_all이 로드된 후에만 수행)
    numeric_cols = df_all.select_dtypes(include=['int64', 'float64']).columns.difference(EXCLUDE_COLS + DROP_COLS).tolist()
    categorical_cols = df_all.select_dtypes(include=['object']).columns.difference(EXCLUDE_COLS + DROP_COLS).tolist()

    print(f"수치형 컬럼 수: {len(numeric_cols)}, 범주형 컬럼 수: {len(categorical_cols)}")
    if not numeric_cols:
        print("⚠️ 경고: numeric_cols 리스트가 비어있습니다. 예측에 문제가 있을 수 있습니다.")
    if not categorical_cols:
        print("⚠️ 경고: categorical_cols 리스트가 비어있습니다. 예측에 문제가 있을 수 있습니다.")

except FileNotFoundError:
    print(f"❌ 오류: 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    print("경로를 확인하거나, 'data' 폴더가 Git에 커밋되었는지 확인하세요.")
    df_all = pd.DataFrame() # 파일이 없으면 빈 DataFrame으로 초기화하여 앱 크래시 방지
except Exception as e:
    print(f"❌ 오류: 데이터 로드 중 예외 발생: {e}")
    print(traceback.format_exc())
    df_all = pd.DataFrame() # 오류 발생 시 빈 DataFrame으로 초기화

# --- 워크스페이스 레이아웃 정의 ---
def wo_layout():
    return html.Div(
        className="glass-card",
        style={"marginTop": "20px", "padding": "2rem", "position": "relative", "minHeight": "90vh"},
        children=[
            dcc.Store(id='fault-history-store', data=[]),  # 불량 기록 저장용 Dcc.Store
            html.H2("📡 실시간 센서 데이터", style={"marginBottom": "20px"}, className='shiny-text'),
            dcc.Interval(id="interval", interval=2_000, n_intervals=0), # 2초마다 콜백 트리거

            # Row Container
            html.Div(
                style={"display": "flex", "gap": "20px"},
                children=[
                    # 왼쪽 카드: 결과 및 확률 표시
                    html.Div(
                        id="result-prob-card",
                        style={
                            "flex": "1", "marginBottom": "10px", "backgroundColor": "rgba(255, 255, 255, 0.3)",
                            "borderRadius": "10px", "padding": "15px", "color": "#000",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)", "minHeight": "100px",
                            "display": "flex", "flexDirection": "column", "justifyContent": "center",
                        },
                        className="glass-card"
                    ),

                    # 중간 카드: 불량 기록 표시
                    html.Div(
                        id="fault-record",
                        style={
                            "flex": "1", "height": "60vh", "overflowY": "auto", "padding": "10px",
                            "backgroundColor": "rgba(255, 255, 255, 0.3)", "borderRadius": "10px",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                        },
                        className='glass-card'
                    ),

                    # 오른쪽 카드: 실시간 확률 그래프
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H5("📈 실시간 양불 확률 그래프"),
                            dcc.Graph(id="prob-graph", style={"height": "60vh"})
                        ]
                    )
                ]
            )
        ]
    )

# --- 콜백 함수 등록 ---
# 이 함수는 app 객체가 생성된 후 `app.layout`이 설정되기 전에 호출되어야 합니다.
def register_callbacks(app):
    # 확률 이력을 저장할 리스트 (함수 내에서 `nonlocal` 키워드를 사용해야 함)
    prob_history = []

    @app.callback(
        Output("result-prob-card", "children"),
        Output("fault-record", "children"),
        Output("prob-graph", "figure"),
        Output("fault-history-store", "data"),
        Input("interval", "n_intervals"), # 2초마다 트리거
        Input({'type': 'delete-fault-btn', 'index': ALL}, 'n_clicks'), # 삭제 버튼 클릭 시 트리거
        State("fault-history-store", "data"), # 불량 기록 데이터 (수정 시킬 때 필요)
        prevent_initial_call=False # 초기 로드 시에도 콜백 실행 허용
    )
    def update_dashboard_or_delete(n_intervals, delete_clicks, fault_history):
        # prob_history 리스트를 이 함수 내에서 변경하기 위해 nonlocal 선언
        nonlocal prob_history
        ctx = callback_context

        # fault_history가 None일 경우 빈 리스트로 초기화 (초기 로드 시 발생 가능)
        if fault_history is None:
            fault_history = []

        # 어떤 Input도 트리거되지 않았다면 업데이트 방지
        if not ctx.triggered:
            raise PreventUpdate

        # 어떤 Input이 콜백을 트리거했는지 확인
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # --- 삭제 버튼 클릭 시 처리 ---
        if trigger_id.startswith("{"):
            try:
                # 클릭된 삭제 버튼의 인덱스 파싱
                triggered_index = eval(trigger_id)['index']
                # 실제 fault_history에서 삭제할 인덱스 계산 (최근 20개만 보여주므로)
                start_index = max(len(fault_history) - 20, 0)
                actual_index = start_index + triggered_index

                # 유효한 인덱스 범위 내에서만 삭제
                if 0 <= actual_index < len(fault_history):
                    fault_history = fault_history.copy()  # 원본 데이터 변경 방지를 위해 복사
                    fault_history.pop(actual_index) # 해당 항목 삭제

                # 삭제 시에는 다른 UI는 업데이트하지 않고, 불량 기록만 업데이트
                return no_update, no_update, no_update, fault_history
            except Exception as e:
                print(f"[❌ 삭제 버튼 콜백 오류] {e}")
                print(traceback.format_exc())
                return no_update, no_update, no_update, no_update

        # --- Interval 업데이트 시 처리 (실시간 데이터 예측) ---
        elif trigger_id == "interval":
            # 데이터나 모델이 로드되지 않았다면 업데이트 중지하고 로그 출력
            if df_all.empty or tuned_vote is None:
                print("⚠️ 경고: df_all(데이터) 또는 모델이 로드되지 않아 데이터 업데이트를 건너뜀.")
                return no_update, no_update, no_update, no_update

            # Interval이 데이터 길이를 초과하면 업데이트 중지
            if n_intervals >= len(df_all):
                print(f"⚠️ 경고: n_intervals({n_intervals})가 데이터 길이를 초과({len(df_all)})했습니다. 더 이상 업데이트하지 않습니다.")
                return no_update, no_update, no_update, no_update

            try:
                # 현재 Interval에 해당하는 데이터 행 추출
                row = df_all.iloc[[n_intervals]].copy()
                # 불필요한 컬럼 제거
                row = row.drop(columns=DROP_COLS, errors='ignore')

                # 'time'과 'date' 컬럼이 존재하면 'timestamp' 컬럼 생성
                if 'time' in row.columns and 'date' in row.columns:
                    row['timestamp'] = pd.to_datetime(row['time'] + " " + row['date'], errors='coerce')
                else:
                    print("⚠️ 경고: 'time' 또는 'date' 컬럼이 없어 timestamp를 생성할 수 없습니다.")
                    row['timestamp'] = pd.NaT # Not a Time (결측 처리)

                # 추출된 'row'가 비어있을 경우 처리 중지
                if row.empty:
                    print(f"⚠️ 경고: n_intervals={n_intervals} 에 해당하는 row 데이터가 비어있습니다. 업데이트 중지.")
                    raise PreventUpdate

                # 숫자형 컬럼 결측치 처리 (원본 df_all의 중앙값 사용)
                for col in numeric_cols:
                    if col in row.columns:
                        if not df_all[col].empty and not df_all[col].isnull().all(): # df_all의 해당 컬럼이 비어있지 않고 모두 NaN이 아닐 때
                            median_val = df_all[col].median()
                            if pd.isna(median_val): # 중앙값 자체가 NaN인 경우 (예: 해당 컬럼이 모두 NaN)
                                median_val = 0.0 # 안전한 기본값으로 대체
                            row[col] = row[col].fillna(median_val)
                        else:
                            print(f"⚠️ 경고: df_all['{col}']가 비어있거나 모든 값이 NaN이어서 숫자형 컬럼의 중앙값을 계산할 수 없습니다. 0.0으로 대체.")
                            row[col] = row[col].fillna(0.0) # df_all 해당 컬럼이 비었을 때를 대비

                # 범주형 컬럼 결측치 처리 (원본 df_all의 최빈값 사용)
                for col in categorical_cols:
                    if col in row.columns:
                        if not df_all[col].empty and not df_all[col].mode().empty: # df_all의 해당 컬럼이 비어있지 않고 최빈값이 존재할 때
                            mode_val = df_all[col].mode()[0]
                            row[col] = row[col].fillna(mode_val)
                        else:
                            print(f"⚠️ 경고: df_all['{col}']가 비어있어 범주형 컬럼의 최빈값을 계산할 수 없습니다. 'unknown'으로 대체.")
                            row[col] = row[col].fillna('unknown') # df_all 해당 컬럼이 비었을 때를 대비

                # 예측에 사용할 전체 피처 컬럼 리스트 생성
                all_feature_cols = numeric_cols + categorical_cols
                if not all_feature_cols:
                    print("❌ 오류: 예측할 피처 컬럼(numeric_cols + categorical_cols)이 비어있습니다. 모델 예측 불가.")
                    raise PreventUpdate # 더 이상 진행 불가

                # X_row를 생성하기 전에, 현재 row에 모든 필요한 컬럼이 있는지 최종 확인
                missing_cols_in_row = [col for col in all_feature_cols if col not in row.columns]
                if missing_cols_in_row:
                    print(f"❌ 오류: 현재 행(row)에 예측에 필요한 컬럼이 부족합니다: {missing_cols_in_row}")
                    raise PreventUpdate

                # 모델 예측을 위한 DataFrame 생성
                X_row = pd.DataFrame(row[all_feature_cols].values, columns=all_feature_cols)

                # 최종적으로 X_row가 비어있을 경우 방지
                if X_row.empty:
                    print("❌ 오류: 모델 예측을 위한 X_row가 비어있습니다. 예측 불가.")
                    raise PreventUpdate

                # 모델 예측 수행
                pred = tuned_vote.predict(X_row)[0]
                prob = tuned_vote.predict_proba(X_row)[0, 1]

                # --- 결과 카드 내용 구성 ---
                result_card_children = [
                    html.Div(
                        f"{'❌ 불량' if pred == 1 else '✅ 양품'} (Index: {n_intervals})",
                        style={"fontWeight": "bold", "fontSize": "20px", "marginBottom": "10px"}
                    ),
                    html.P(f"예측 결과: {'불량품' if pred == 1 else '정상품'}"),
                    html.P(f"🔍 불량 확률: {prob:.4f}")
                ]

                # --- 불량 기록 업데이트 및 표시 ---
                if pred == 1: # 불량으로 예측된 경우에만 기록
                    fault_history = fault_history.copy() # 원본 데이터 변경 방지를 위해 복사
                    fault_history.append(f"Index {n_intervals} - 불량 발생 (확률: {prob:.4f})")

                fault_count = len(fault_history[-20:]) # 불량 기록은 최근 20개만 표시

                fault_items = []
                for i, rec in enumerate(fault_history[-20:]): # 최근 20개만 UI에 표시
                    fault_items.append(
                        html.Div(
                            [
                                html.Span(rec, style={"marginRight": "10px"}),
                                html.Button(
                                    "삭제",
                                    id={'type': 'delete-fault-btn', 'index': i}, # 동적 ID 생성
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#dc3545", "color": "white", "border": "none",
                                        "borderRadius": "4px", "padding": "2px 8px", "cursor": "pointer",
                                    }
                                ),
                            ],
                            style={
                                "marginBottom": "0px", "padding": "4px", "display": "flex",
                                "justifyContent": "space-between", "alignItems": "center",
                                "backgroundColor": "rgba(255,255,255,0.2)", "borderRadius": "5px"
                            }
                        )
                    )

                fault_display_children = [
                    html.H5(
                        "📋불량 기록" + (" ⚠️불량 5건 이상" if fault_count >= 5 else ""), # 5건 이상일 때 경고
                        style={"marginBottom": "10px"}
                    ),
                    *fault_items # 리스트 언팩킹
                ]

                fault_display_style = {
                    "backgroundColor": "rgba(255, 255, 255, 0.3)", "borderRadius": "10px", "padding": "15px",
                    "color": "#000", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)", "maxHeight": "60vh",
                    "overflowY": "auto",
                    "border": "2px solid red" if fault_count >= 5 else "1px solid rgba(0,0,0,0.1)" # 5건 이상일 때 테두리 색상 변경
                }

                fault_display = html.Div(
                    fault_display_children,
                    style=fault_display_style,
                    className='glass-card'
                )

                # --- 확률 그래프 업데이트 ---
                prob_history.append(prob) # 현재 확률을 이력에 추가
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=prob_history.copy(), # 이력 복사 (Plotly 내부적으로 필요)
                    mode="lines+markers",
                    name="불량 확률",
                    line=dict(color="red")
                ))
                fig.update_layout(
                    yaxis=dict(title="불량 확률", range=[0, 1]), # Y축 범위 0~1 고정
                    xaxis=dict(title="시간 흐름 (Interval)"),
                    margin=dict(t=30, l=30, r=30, b=30),
                    paper_bgcolor='rgba(0,0,0,0)', # 그래프 배경 투명하게
                    plot_bgcolor='rgba(0,0,0,0)' # 플롯 영역 배경 투명하게
                )

                # 모든 Output 업데이트
                return result_card_children, fault_display, fig, fault_history

            except PreventUpdate:
                # PreventUpdate는 에러가 아니므로 로그에 출력하지 않음
                return no_update, no_update, no_update, no_update
            except Exception as e:
                # 콜백 내부에서 예상치 못한 오류 발생 시 로그에 상세 정보 출력
                print(f"[❌ 콜백 오류] {e}")
                print(traceback.format_exc()) # 상세 트레이스백 출력 (Render 로그에서 확인)
                return no_update, no_update, no_update, no_update # 오류 발생 시 UI 업데이트 방지

        else:
            # interval 또는 삭제 버튼이 아닌 다른 Input이 트리거된 경우 (현재는 해당 없음)
            return no_update, no_update, no_update, no_update


# --- analytics_layout 함수 (변경 없음) ---
def analytics_layout():
    return html.Div([
        html.H3("인수인계"),
        html.P("여기는 실시간 데이터 분석 페이지입니다."),
    ])
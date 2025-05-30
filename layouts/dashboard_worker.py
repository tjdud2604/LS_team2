# app.py

from dash import html, dcc, Output, Input, State, callback_context, no_update
import pandas as pd
import plotly.graph_objs as go
import joblib
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
from datetime import datetime
import os

# --- Celery 및 Redis 설정 (비동기 콜백을 사용하려면 이 부분이 필요합니다) ---
# 이 부분은 현재 주석 처리되어 있습니다.
# Render의 무료 플랜에서 Celery 워커를 실행하기 어렵고, CeleryManager 임포트 오류가 발생했기 때문입니다.
# 만약 비동기 콜백을 사용하려면, 아래 주석을 풀고 Redis 설정을 완료해야 합니다.
# from dash.long_callback import CeleryManager
# from celery import Celery 

# REDIS_URL = os.environ.get('REDIS_URL')
# if REDIS_URL:
#     print(f"Connecting to Redis at: {REDIS_URL}")
#     celery_app = Celery(__name__, broker=REDIS_URL, backend=REDIS_URL)
#     long_callback_manager = CeleryManager(celery_app)
# else:
#     # REDIS_URL이 설정되지 않은 경우, 비동기 콜백은 사용할 수 없습니다.
#     # 이 경우, long_callback을 사용할 수 없으므로 해당 데코레이터를 제거하거나
#     # DiskcacheManager를 사용하도록 폴백 로직을 구현해야 합니다.
#     raise ValueError("REDIS_URL 환경 변수가 설정되지 않았습니다. 비동기 콜백을 위해 필요합니다.")

# long_callback_manager를 여기에 전달합니다. (위에서 주석처리된 Celery 설정 후)
app = dash.Dash(__name__) # long_callback_manager=long_callback_manager 주석처리

server = app.server

# === 파일 경로 및 데이터 로드 ===
MODEL_PATH = "./data/model_voting.pkl" # 모델 경로 확인 (voting.pkl 또는 voting1.pkl)
DATA_PATH = "./data/train.csv"

# === EXCLUDE_COLS 수정 ===
# 모델 예측에 필요한 컬럼들을 더 이상 EXCLUDE_COLS에서 제외하지 않도록 수정합니다.
# 'passorfail', 'tryshot_signal', 'heating_furnace'는 여전히 제외합니다.
EXCLUDE_COLS = [
    'id', 'line', 'name', 'mold_name', 'time', 'date',
    'passorfail', # 예측 목표 변수이므로 제외
    'tryshot_signal', 'heating_furnace' # 원본 코드에서 의도적으로 제외된 컬럼
    # 'count', 'working', 'molten_volume', 'upper_mold_temp3', 'lower_mold_temp3', 'registration_time'
    # 이 컬럼들은 이제 제외되지 않고 모델 입력에 포함됩니다.
]

# 센서 특성 및 임계값
sensor_features = [
    "cast_pressure", "lower_mold_temp2", "low_section_speed", "upper_mold_temp1",
    "upper_mold_temp2", "sleeve_temperature", "lower_mold_temp1", "high_section_speed"
]
sensor_thresholds = {
    "upper_mold_temp1": [(99,104),(154, 999)],
    "upper_mold_temp2": (154, 999),
    "lower_mold_temp1": [(124,134),(159,309)],
    "lower_mold_temp2": [(139,149),(169,999)],
    "low_section_speed": [105, 110, 115],
    "high_section_speed": [(101.5, 102.5),(109,113),(117,118)],
    "cast_pressure": [(311,315),(318,334)],
    "sleeve_temperature": [(160, 210), (460, 530)]
}

# 모델 및 데이터 로드
try:
    tuned_vote = joblib.load(MODEL_PATH)
    print(f"✅ 모델 로드 성공: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    class DummyModel:
        def predict(self, X): return [0]
        def predict_proba(self, X): return [[0.9, 0.1]]
    tuned_vote = DummyModel()

try:
    df_all = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"✅ 데이터 로드 성공: {DATA_PATH}, Shape: {df_all.shape}")
except FileNotFoundError:
    print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    df_all = pd.DataFrame()

# 특징 컬럼 정의 (수정된 EXCLUDE_COLS를 반영)
numeric_cols = df_all.select_dtypes(include=['int64', 'float64']).columns.difference(EXCLUDE_COLS).tolist()
categorical_cols = df_all.select_dtypes(include=['object']).columns.difference(EXCLUDE_COLS).tolist()

# === 헬퍼 함수 ===
def get_card_color(feature, value):
    if pd.isna(value):
        return "#d4edda"
    
    threshold = sensor_thresholds.get(feature)
    if threshold is None:
        return ""
    
    if isinstance(threshold, tuple):
        if threshold[0] <= value <= threshold[1]:
            return ""
        else:
            return "#f8d7da"
    elif isinstance(threshold, list):
        is_normal = False
        if all(isinstance(r, tuple) for r in threshold):
            for r in threshold:
                if r[0] <= value <= r[1]:
                    is_normal = True
                    break
        else:
            if value in threshold:
                is_normal = True
        
        return "" if is_normal else "#f8d7da"
    return ""

# === 메인 레이아웃 ===
def wo_layout():
    return html.Div(
        className="glass-card",
        style={"marginTop": "20px", "padding": "2rem", "position": "relative", "minHeight": "90vh"},
        children=[
            dcc.Store(id='fault-history-store', data=[]),
            dcc.Store(id='counter-store', data={
                'total_count': 0,
                'fail_count': 0,
                'timestamps': [],
                'failure_rates': [],
                'defect_logs': []
            }),
            html.H2("📡 실시간 센서 데이터", style={"marginBottom": "20px"}, className='shiny-text'),
            dcc.Interval(id="interval", interval=2_000, n_intervals=0),

            html.Div([
                html.H4("센서 데이터 확인", style={"marginTop": "40px", "marginBottom": "10px"}, className='shiny-text'),
                html.Div(
                    id="sensor-card-container",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "20px"
                    }
                )
            ]),

            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginTop": "20px"},
                children=[
                    html.Div(
                        id="result-prob-card",
                        style={
                            "flex": "1",
                            "minWidth": "250px",
                            "marginBottom": "10px",
                            "backgroundColor": "rgba(255, 255, 255, 0.3)",
                            "borderRadius": "10px",
                            "padding": "15px",
                            "color": "#000",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                            "minHeight": "100px",
                            "display": "flex",
                            "flexDirection": "column",
                            "justifyContent": "center",
                        },
                        className="glass-card"
                    ),

                    html.Div(
                        id="fault-record",
                        style={
                            "flex": "1",
                            "minWidth": "300px",
                            "maxHeight": "60vh",
                            "overflowY": "auto",
                            "padding": "10px",
                            "backgroundColor": "rgba(255, 255, 255, 0.3)",
                            "borderRadius": "10px",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                        },
                        className='glass-card'
                    ),

                    html.Div(
                        style={"flex": "2", "minWidth": "400px"},
                        children=[
                            html.H5("실시간 양불 확률 그래프", style={"textAlign": "center"}),
                            dcc.Graph(id="prob-graph", style={"height": "60vh"})
                        ]
                    )
                ]
            )
        ]
    )

# === 콜백 등록 ===
def register_callbacks(app):

    @app.callback(
        Output("result-prob-card", "children"),
        Output("fault-record", "children"),
        Output("prob-graph", "figure"),
        Output("fault-history-store", "data"),
        Output("sensor-card-container", "children"),
        Output("counter-store", "data"),
        Input("interval", "n_intervals"),
        Input({'type': 'delete-fault-btn', 'index': ALL}, 'n_clicks'), 
        State("fault-history-store", "data"),
        State("counter-store", "data"),
        prevent_initial_call=False
    )
    def update_dashboard_and_delete(n_intervals, delete_clicks_list, fault_history, counter_data):
        ctx = callback_context

        if fault_history is None:
            fault_history = []
        if counter_data is None:
            counter_data = {
                'total_count': 0,
                'fail_count': 0,
                'timestamps': [],
                'failure_rates': [],
                'defect_logs': []
            }

        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # --- 1. 삭제 버튼 클릭 시 처리 ---
        if "delete-fault-btn" in trigger_id:
            try:
                triggered_index = eval(trigger_id)['index'] 
            except (SyntaxError, NameError):
                raise PreventUpdate
            
            if 0 <= triggered_index < len(fault_history):
                fault_history = fault_history.copy()
                deleted_log = fault_history.pop(triggered_index)
                print(f"삭제됨: {deleted_log}, 남은 기록 수: {len(fault_history)}")
                
                fault_items = []
                for i, rec in enumerate(fault_history): 
                    fault_items.append(
                        html.Div(
                            [
                                html.Span(rec, style={"marginRight": "10px"}),
                                html.Button(
                                    "삭제",
                                    id={'type': 'delete-fault-btn', 'index': i},
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#dc3545",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "2px 8px",
                                        "cursor": "pointer",
                                    }
                                ),
                            ],
                            style={
                                "marginBottom": "0px",
                                "padding": "4px",
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "backgroundColor": "rgba(255,255,255,0.2)",
                                "borderRadius": "5px",
                                "marginTop": "5px"
                            }
                        )
                    )
                
                fault_count = len(fault_history)
                fault_display_children = [
                    html.H5(
                        "불량 기록" + (" ⚠️불량 5건 이상" if fault_count >= 5 else ""),
                        style={"marginBottom": "10px"}
                    ),
                    *fault_items
                ]
                
                fault_display_style = {
                    "backgroundColor": "rgba(255, 255, 255, 0.3)",
                    "borderRadius": "10px",
                    "padding": "15px",
                    "color": "#000",
                    "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                    "maxHeight": "60vh",
                    "overflowY": "auto",
                    "border": "2px solid red" if fault_count >= 5 else "1px solid rgba(0,0,0,0.1)"
                }
                
                fault_display = html.Div(
                    fault_display_children,
                    style=fault_display_style,
                    className='glass-card'
                )
                
                return no_update, fault_display, no_update, fault_history, no_update, no_update
            else:
                raise PreventUpdate
        
        # --- 2. interval 업데이트 시 처리 ---
        elif trigger_id == "interval":
            if df_all.empty or n_intervals >= len(df_all):
                print(f"데이터 끝 또는 로드 안됨. n_intervals: {n_intervals}, df_all_len: {len(df_all)}")
                raise PreventUpdate

            try:
                row = df_all.iloc[[n_intervals]].copy()
                
                # 데이터 전처리: 모델 예측에 필요한 컬럼 포함
                # 이전에 EXCLUDE_COLS에 있던 컬럼(count, working 등)들이 이제 포함됩니다.
                for col in numeric_cols:
                    if col in row.columns:
                        row[col] = row[col].fillna(df_all[col].median() if not df_all.empty else 0)
                for col in categorical_cols:
                    if col in row.columns:
                        row[col] = row[col].fillna(df_all[col].mode()[0] if not df_all.empty else 'N/A')

                # 모델 예측에 사용될 X_row 생성 시, numeric_cols와 categorical_cols에
                # 이제 문제가 되었던 컬럼들이 포함되어 있을 것입니다.
                X_row = pd.DataFrame(row[numeric_cols + categorical_cols].values, columns=numeric_cols + categorical_cols)

                # 모델 예측
                pred = tuned_vote.predict(X_row)[0]
                prob = tuned_vote.predict_proba(X_row)[0, 1]

                # --- 카운터 및 불량률 업데이트 ---
                counter_data['total_count'] += 1
                if pred == 1:
                    counter_data['fail_count'] += 1
                
                failure_rate = counter_data['fail_count'] / counter_data['total_count'] if counter_data['total_count'] > 0 else 0
                now_time = datetime.now().strftime("%H:%M:%S")
                counter_data['timestamps'].append(now_time)
                counter_data['failure_rates'].append(failure_rate)

                # --- 결과 카드 내용 구성 ---
                result_card_children = [
                    html.Div(
                        f"{'불량' if pred == 1 else '양품'} (Index: {n_intervals})",
                        style={"fontWeight": "bold", "fontSize": "20px", "marginBottom": "10px"}
                    ),
                    html.P(f"예측 결과: {'불량품' if pred == 1 else '정상품'}"),
                    html.P(f"불량 확률: {prob:.4f}")
                ]

                # --- 불량 기록 업데이트 ---
                if pred == 1:
                    fault_history = fault_history.copy()
                    fault_history.append(f"Index {n_intervals} - 불량 발생 (확률: {prob:.4f})")
                
                fault_items = []
                for i, rec in enumerate(fault_history[-20:]):
                    fault_items.append(
                        html.Div(
                            [
                                html.Span(rec, style={"marginRight": "10px"}),
                                html.Button(
                                    "삭제",
                                    id={'type': 'delete-fault-btn', 'index': i},
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#dc3545",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "2px 8px",
                                        "cursor": "pointer",
                                    }
                                ),
                            ],
                            style={
                                "marginBottom": "0px",
                                "padding": "4px",
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "backgroundColor": "rgba(255,255,255,0.2)",
                                "borderRadius": "5px",
                                "marginTop": "5px"
                            }
                        )
                    )

                fault_count = len(fault_history[-20:])
                fault_display_children = [
                    html.H5(
                        "불량 기록" + (" ⚠️불량 5건 이상" if fault_count >= 5 else ""),
                        style={"marginBottom": "10px"}
                    ),
                    *fault_items
                ]

                fault_display_style = {
                    "backgroundColor": "rgba(255, 255, 255, 0.3)",
                    "borderRadius": "10px",
                    "padding": "15px",
                    "color": "#000",
                    "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                    "maxHeight": "60vh",
                    "overflowY": "auto",
                    "border": "2px solid red" if fault_count >= 5 else "1px solid rgba(0,0,0,0.1)"
                }

                fault_display = html.Div(
                    fault_display_children,
                    style=fault_display_style,
                    className='glass-card'
                )

                # --- 불량률 그래프 업데이트 ---
                recent_times = counter_data['timestamps'][-60:]
                recent_rates = counter_data['failure_rates'][-60:]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recent_times,
                    y=recent_rates,
                    mode='lines+markers',
                    name='최근 불량률',
                    line=dict(color='red', width=3),
                    marker=dict(color='red', size=10, symbol='circle')
                ))
                fig.update_layout(
                    title='📉 최근 60개 예측 기준 누적 불량률',
                    title_font=dict(size=22, family='Arial', color='black'),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        title='시간 (실시간)', 
                        title_font=dict(size=14),
                        showgrid=True,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        title='누적 불량률', 
                        title_font=dict(size=14), 
                        range=[0, 0.1],
                        showgrid=True,
                        gridcolor='lightgray'
                    ),
                    margin=dict(l=50, r=30, t=50, b=40),
                    font=dict(size=13)
                )

                # --- 센서 카드 업데이트 ---
                sensor_cards = []
                for col in sensor_features:
                    val = row[col].values[0] if col in row.columns else None
                    display_val = f"값: {val:.2f}" if pd.notna(val) else "값: N/A"
                    bg_color = get_card_color(col, val)
                    card_style = {
                        "backgroundColor": bg_color if bg_color else "rgba(255, 255, 255, 0.3)",
                        "padding": "1rem",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
                        "textAlign": "center",
                    }
                    sensor_cards.append(
                        html.Div(
                            className="sensor-card glass-card",
                            style=card_style,
                            children=[
                                html.H6(col, style={"marginBottom": "10px"}),
                                html.Div(display_val, style={"fontSize": "1.5rem", "fontWeight": "bold"})
                            ]
                        )
                    )

                return result_card_children, fault_display, fig, fault_history, sensor_cards, counter_data

            except Exception as e:
                print(f"[❌ Error in interval update] {e}")
                return no_update, no_update, no_update, no_update, no_update, no_update
        
        return no_update, no_update, no_update, no_update, no_update, no_update

################# 인수인계 페이지 

def analytics_layout():
    return html.Div(
        className="glass-card",
        style={"marginTop": "20px", "padding": "2rem", "minHeight": "90vh"},
        children=[
            # 인수인계 데이터를 저장할 dcc.Store
            dcc.Store(id='handover-data-store', data={
                'comments': [], # 사용자가 작성할 코멘트
                'checklist_status': {} # 체크리스트 항목별 상태
            }),

            html.H2("📄 인수인계 페이지", style={"marginBottom": "20px"}, className='shiny-text'),

            # 이전 작업자 코멘트
            html.Div(
                style={"backgroundColor": "rgba(255, 255, 255, 0.3)", "padding": "20px", "borderRadius": "10px", "marginBottom": "30px"},
                children=[
                    html.H3("이전 작업자 코멘트", style={"borderBottom": "1px solid #ccc", "paddingBottom": "10px", "marginBottom": "15px"}),
                    html.P("1. 주간 보고서 제출 시, '생산량', '불량률', '주요 이슈'를 필수로 포함.", style={"marginBottom": "5px"}),
                    html.P("2. 야간 근무 시, 2시간마다 장비 온도를 수동으로 확인 후 기록.", style={"marginBottom": "5px"}),
                    html.P("3. 문제가 발생하면 즉시 담당 엔지니어에게 연락바람."),
                ]
            ),

            # 지시사항 (체크리스트)
            html.Div(
                style={"backgroundColor": "rgba(255, 255, 255, 0.3)", "padding": "20px", "borderRadius": "10px", "marginBottom": "30px"},
                children=[
                    html.H3("오늘의 지시사항 (체크리스트)", style={"borderBottom": "1px solid #ccc", "paddingBottom": "10px", "marginBottom": "15px"}),
                    # 체크리스트 항목들. 실제로는 데이터베이스나 파일에서 로드하는 것이 좋습니다.
                    # 여기서는 예시로 하드코딩합니다.
                    html.Div(id='checklist-container', children=[
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'checklist-item', 'index': 0},
                                options=[{'label': '1. 안전 점검 완료', 'value': 'task_1'}],
                                value=[], # 초기 상태는 체크되지 않음
                                inline=True
                            )
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'checklist-item', 'index': 1},
                                options=[{'label': '2. 샘플링 및 품질 검사 진행', 'value': 'task_2'}],
                                value=[],
                                inline=True
                            )
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'checklist-item', 'index': 2},
                                options=[{'label': '3. 생산량 현황 확인', 'value': 'task_3'}],
                                value=[],
                                inline=True
                            )
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            dcc.Checklist(
                                id={'type': 'checklist-item', 'index': 3},
                                options=[{'label': '4. 불량품 발생 기록', 'value': 'task_4'}],
                                value=[],
                                inline=True
                            )
                        ], style={"marginBottom": "10px"})
                    ])
                ]
            ),

            # 실시간 사용자 코멘트
            html.Div(
                style={"backgroundColor": "rgba(255, 255, 255, 0.3)", "padding": "20px", "borderRadius": "10px"},
                children=[
                    html.H3("사용자 코멘트", style={"borderBottom": "1px solid #ccc", "paddingBottom": "10px", "marginBottom": "15px"}),
                    html.Div(id='user-comments-display', style={"minHeight": "100px", "marginBottom": "20px", "border": "1px solid #eee", "padding": "10px", "borderRadius": "5px", "backgroundColor": "white"}),
                    dcc.Textarea(
                        id='new-comment-input',
                        placeholder='새로운 코멘트를 입력하세요...',
                        style={'width': '100%', 'height': 80, 'marginBottom': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                    ),
                    html.Button('코멘트 작성', id='submit-comment-button', n_clicks=0,
                                style={
                                    "backgroundColor": "#007bff",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "padding": "8px 15px",
                                    "cursor": "pointer"
                                })
                ]
            )
        ]
    )

# ... (기존 register_callbacks 함수 생략) ...
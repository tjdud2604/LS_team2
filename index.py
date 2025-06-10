# index.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import json

from layouts import login, dashboard_worker, dashboard_admin
from components.sidebar import side_layout
from layouts.dashboard_worker import preload_initial_data
from feature_engineer import FeatureEngineer
import pandas as pd

from layouts.dashboard_admin import compute_valid_ranges, DEFAULT_THRESHOLDS_BY_MOLD
pd.set_option('future.no_silent_downcasting', True)

# 예시 사용자 정보
USERS = {
    "1": {"password": "1", "role": "admin"},
    "2": {"password": "2", "role": "worker"}
}

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
server = app.server

# 애플리케이션의 메인 레이아웃을 정의하는 함수
def serve_layout():
    # 초기 데이터 사전 로딩 (train.csv에서)
    initial_counter_data, initial_fault_history, initial_monitoring_data, production_monitoring, mold_stats = preload_initial_data()

    train_df = pd.read_csv("./data/sampled_train.csv", low_memory=False) ### 경로 바꾸깅
    initial_valid_ranges = compute_valid_ranges(train_df, DEFAULT_THRESHOLDS_BY_MOLD)
    return html.Div([
        dcc.Location(id="url", refresh=False),
        
        # ✅ 초기 데이터로 store 설정
        dcc.Store(id='fault-history-store', data=initial_fault_history),
        dcc.Store(id='counter-store', data=initial_counter_data),
        dcc.Store(id='realtime-monitoring-store', data=initial_monitoring_data),
        dcc.Store(id="valid-ranges-store", data=initial_valid_ranges),
        dcc.Store(id="production-monitoring-store",data=production_monitoring),
        dcc.Store(id="mold-stats-store",data=mold_stats),
        
        dcc.Interval(id="interval", interval=2_000, n_intervals=0),
        dcc.Store(id="user-role"),
        dcc.Store(id="username-store"),
        dcc.Store(id="password-store"),
        

        html.Div(id="sidebar-container"),
        html.Button("Logout", id="logout-button", style={"display": "none"}),
        html.Button(id="login-trigger-button", style={"display": "none"}),

        html.Div(id="page-content", children=login.layout()),
        html.Div(id="login-message", style={"color": "red", "textAlign": "center", "marginTop": "10px"})
    ])

app.layout = serve_layout

# --- 로그인 입력 값 dcc.Store에 저장 ---
# submit-login-button 클릭 시, input-username과 input-password의 값을 dcc.Store에 저장

@app.callback(
    Output("username-store", "data"),
    Output("password-store", "data"),
    Output("login-trigger-button", "n_clicks"), # 로그인 트리거 버튼 클릭 수 업데이트
    Input("submit-login-button", "n_clicks"),
    State("input-username", "value"),
    State("input-password", "value"),
    State("login-trigger-button", "n_clicks"), # 현재 트리거 버튼 n_clicks
    prevent_initial_call=True
)
def store_login_credentials_and_trigger(submit_clicks, username_val, password_val, trigger_clicks):
    if submit_clicks is None or submit_clicks == 0:
        raise PreventUpdate
    
    # Debugging:
    print(f"Store callback triggered. Username: {username_val}, Password: {password_val}")

    # n_clicks가 None일 경우 0으로 초기화
    new_trigger_clicks = (trigger_clicks or 0) + 1 
    
    return username_val, password_val, new_trigger_clicks


# --- 로그인/로그아웃 처리 콜백 ---
# 로그인 트리거 버튼 또는 로그아웃 버튼 클릭 시 사용자 역할 및 경로 업데이트
@app.callback(
    Output("user-role", "data"),
    Output("url", "pathname"),
    Output("login-message", "children"),
    Input("login-trigger-button", "n_clicks"), # 이제 이 버튼이 로그인 로직을 트리거합니다.
    Input("logout-button", "n_clicks"),
    State("username-store", "data"), # dcc.Store에서 사용자 이름 가져오기
    State("password-store", "data"), # dcc.Store에서 비밀번호 가져오기
    State("user-role", "data"), # 현재 저장된 사용자 역할
    prevent_initial_call=True
)
def login_logout_callback(login_trigger_clicks, logout_clicks, username, password, current_role):
    triggered_id = ctx.triggered_id

    # Debugging print statements
    print(f"Login/Logout callback triggered by: {triggered_id}")
    if triggered_id == "login-trigger-button":
        print(f"Username from Store: {username}")
        print(f"Password from Store: {password}")
    
    # 로그인 트리거 버튼이 클릭된 경우
    if triggered_id == "login-trigger-button":
        # username이나 password가 None이거나 비어있으면 오류 메시지 반환
        if username is None or password is None or not username or not password:
            return no_update, no_update, "아이디와 비밀번호를 모두 입력하세요."
        
        # 사용자 인증 로직
        if username in USERS and USERS[username]["password"] == password:
            role = USERS[username]["role"]
            path = "/worker/analytics" if role == "worker" else "/admin/model" if role == "admin" else "/"
            return role, path, "" # 역할, 경로, 메시지 초기화
        else:
            return no_update, no_update, "아이디 또는 비밀번호가 잘못되었습니다."
    
    # 로그아웃 버튼이 클릭된 경우
    elif triggered_id == "logout-button":
        # 로그아웃 시에는 사용자 역할, 경로, 메시지 모두 초기화
        return None, "/", ""
    
    return no_update, no_update, no_update

# --- 통합 레이아웃 업데이트 콜백 ---
# 사용자 역할 또는 URL 경로 변경 시 사이드바, 로그아웃 버튼, 페이지 콘텐츠 업데이트
@app.callback(
    Output("sidebar-container", "style"),
    Output("sidebar-container", "children"),
    Output("logout-button", "style"),
    Output("page-content", "children"),
    Input("user-role", "data"), # dcc.Store에 저장된 사용자 역할
    Input("url", "pathname") # 현재 URL 경로
)
def unified_layout(role, pathname):
    sidebar_style = {"display": "none"}
    sidebar_children = ""
    logout_style = {"display": "none"}
    page_layout = login.layout() # 기본은 로그인 페이지

    if not role or pathname == "/":
        return sidebar_style, sidebar_children, logout_style, login.layout()

    if role:
        sidebar_style = {"display": "block"}
        sidebar_children = side_layout(role)
        logout_style = {"display": "inline-block"}

        if role == "worker":
            if pathname == "/worker":
                page_layout = dashboard_worker.wo_layout()
            elif pathname == "/worker/analytics":
                page_layout = dashboard_worker.analytics_layout()
            elif pathname == "/worker/analytics":
                page_layout = dashboard_worker.mold_layout()
            else:
                page_layout = dashboard_worker.wo_layout()
        elif role == "admin":
            if pathname == "/admin":
                page_layout = dashboard_admin.ad_layout()
            else:
                page_layout = dashboard_admin.ad_layout()

    return sidebar_style, sidebar_children, logout_style, page_layout

# 대시보드 워커 콜백 등록 (layouts/dashboard_worker.py에 정의된 콜백)
dashboard_worker.register_callbacks(app)
dashboard_worker.register_monitoring_callbacks(app)
dashboard_worker.register_mold_callbacks(app) 
# dashboard_worker.register_mold_time_callbacks(app)
dashboard_admin.register_callbacks(app)

# if __name__ == "__main__":
#     app.run(debug=True, port=8050)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Render에서 제공하는 포트 사용
    app.run(host="0.0.0.0", port=port, debug=True)  # 외부 접속 가능한 호스트로 설정


# import os

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8050))  # 환경변수에서 포트 읽기
#     app.run_server(host="0.0.0.0", port=port, debug=True)

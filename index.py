# index.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, no_update, callback_context, ALL
from dash.exceptions import PreventUpdate
from datetime import datetime

# layouts 폴더에서 login, dashboard_worker, dashboard_admin 모듈 임포트
from layouts import login, dashboard_worker, dashboard_admin
# components 폴더에서 sidebar 모듈 임포트
from components.sidebar import side_layout
import os
from flask import Flask


app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
print(os.environ.get("PORT"))
server = app.server

# 예시 사용자 정보
USERS = {
    "1": {"password": "1", "role": "admin"},
    "2": {"password": "2", "role": "worker"}
}

# 애플리케이션의 메인 레이아웃을 정의하는 함수
def serve_layout():
    return html.Div([
        dcc.Location(id="url", refresh=False), # URL 변경 감지
        dcc.Store(id="user-role"), # 사용자 역할 저장
        dcc.Store(id="username-store"), # 사용자 이름을 임시 저장할 dcc.Store 추가
        dcc.Store(id="password-store"), # 비밀번호를 임시 저장할 dcc.Store 추가
        
        html.Div(id="sidebar-container"), # 사이드바 컨테이너
        html.Button("Logout", id="logout-button", style={"display": "none"}), # 로그아웃 버튼 (초기 숨김)
        html.Button(id="login-trigger-button", style={"display": "none"}), # 로그인 로직을 트리거할 숨겨진 버튼 (이전 login-button 대체)
        html.Div(id="page-content", children=login.layout()), # 페이지 내용 컨테이너 (초기 로그인 페이지)
        html.Div(id="login-message", style={"color": "red", "textAlign": "center", "marginTop": "10px"})
    ])

app.layout = serve_layout()

# --- 로그인 입력 값 dcc.Store에 저장 ---
# submit-login-button 클릭 시, input-username과 input-password의 값을 dcc.Store에 저장
# 이 콜백은 login.layout()이 화면에 있을 때만 작동하므로 안전합니다.
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
            else:
                page_layout = dashboard_worker.wo_layout()
        elif role == "admin":
            if pathname == "/admin":
                page_layout = dashboard_admin.ad_layout()
            elif pathname == "/admin/model":
                page_layout = dashboard_admin.model_layout()
            elif pathname == "/admin/final":
                page_layout = dashboard_admin.final_layout()
            else:
                page_layout = dashboard_admin.ad_layout()

    return sidebar_style, sidebar_children, logout_style, page_layout



# app.py

# ... (기존 register_callbacks 함수 내부의 update_dashboard_and_delete 콜백 생략) ...


    # --- 사용자 코멘트 업데이트 콜백 ---
@app.callback(
    Output('user-comments-display', 'children'),
    Output('new-comment-input', 'value'),
    Output('handover-data-store', 'data'),
    Input('submit-comment-button', 'n_clicks'),
    Input('handover-data-store', 'data'), # 새로고침 시 기존 코멘트 로드용
    State('new-comment-input', 'value'),
    prevent_initial_call=False # 초기 로딩 시에도 댓글 표시
)
def update_user_comments(n_clicks, stored_data, new_comment):
        # 콜백이 실행될 때 stored_data에서 최신 코멘트를 가져옴
    comments = stored_data.get('comments', [])
        
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    if trigger_id == 'submit-comment-button' and n_clicks > 0 and new_comment:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment_entry = f"[{timestamp}] 사용자: {new_comment}"
        comments.append(comment_entry)
            
        # dcc.Store 데이터 업데이트
        stored_data['comments'] = comments
            
        # 화면에 표시될 코멘트 HTML 생성
        comment_elements = [html.P(comment, style={"marginBottom": "5px"}) for comment in comments]
        return comment_elements, "", stored_data # 입력창 초기화, 스토어 업데이트
        
    elif trigger_id == 'handover-data-store' or trigger_id == 'initial_load':
        # 페이지 로드 또는 스토어 업데이트 시 기존 코멘트 표시
        comment_elements = [html.P(comment, style={"marginBottom": "5px"}) for comment in comments]
        return comment_elements, no_update, no_update # 입력창 유지, 스토어 유지
        
    return no_update, no_update, no_update


    # --- 체크리스트 상태 업데이트 콜백 ---
@app.callback(
    Output('handover-data-store', 'data', allow_duplicate=True), # allow_duplicate=True 필요
    Input({'type': 'checklist-item', 'index': ALL}, 'value'), # 동적으로 생성된 체크리스트 아이템
    State('handover-data-store', 'data'),
    prevent_initial_call=True # 초기 로딩 시에는 실행되지 않도록
)

def update_checklist_status(checklist_values, stored_data):
        # checklist_values는 각 체크리스트 항목의 현재 값(체크된 항목) 리스트의 리스트입니다.
        # 예: [[], ['task_2'], [], ['task_4']]
        
    checklist_status = stored_data.get('checklist_status', {})
    
    # 어떤 체크리스트 항목이 변경되었는지 정확히 파악하기 위해 ctx.triggered 사용
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
            
    for input_item in ctx.triggered:
        prop_id = input_item['prop_id'] # 예: '{"index":0,"type":"checklist-item"}.value'
        new_value = input_item['value'] # 예: ['task_1'] 또는 []
            
        # Dynamic ID에서 index와 type 추출
        if 'checklist-item' in prop_id:
            try:
                # '{"index":0,"type":"checklist-item"}.value' 에서 index 추출
                idx_str = prop_id.split('"index":')[1].split(',')[0]
                index = int(idx_str)
                
                # 해당 체크리스트 항목의 ID (예: 'task_1') 가져오기
                # options[0]['value']를 사용하는 이유는, 각 checklist-item은 단일 옵션만 가지기 때문입니다.
                # 만약 여러 옵션을 가질 수 있다면, 더 복잡한 로직이 필요합니다.
                task_id = f'task_{index + 1}' # 예시로 task_0, task_1 대신 task_1, task_2로 매핑
                    
                # 체크 여부 업데이트 (체크되면 True, 아니면 False)
                checklist_status[task_id] = bool(new_value)
            except Exception as e:
                print(f"Error parsing checklist item ID: {e}")

    stored_data['checklist_status'] = checklist_status
    return stored_data

    # --- 체크리스트 초기 로딩 시 상태 복원 콜백 ---
    # 이 콜백은 페이지가 로드될 때 stored_data에 저장된 checklist_status를 바탕으로
    # 체크리스트 항목들의 'value'를 설정하여 이전에 체크된 상태를 복원합니다.
@app.callback(
    Output({'type': 'checklist-item', 'index': ALL}, 'value'),
    Input('handover-data-store', 'data'),
    State({'type': 'checklist-item', 'index': ALL}, 'id'), # 각 체크리스트 아이템의 id를 가져옴
    prevent_initial_call=False
)
def restore_checklist_status(stored_data, checklist_ids):
    if not stored_data or 'checklist_status' not in stored_data:
        raise PreventUpdate
            
    checklist_status = stored_data['checklist_status']
    output_values = []

    # checklist_ids는 다음과 같은 형태의 리스트:
    # [{'index': 0, 'type': 'checklist-item'}, {'index': 1, 'type': 'checklist-item'}, ...]
    
    for item_id_dict in checklist_ids:
        index = item_id_dict['index']
        task_id = f'task_{index + 1}' # task_1, task_2 등으로 매핑
        
        # 저장된 상태에 따라 체크 여부 설정
        if checklist_status.get(task_id, False):
            output_values.append([task_id]) # 체크된 상태는 리스트에 value를 담아서 반환
        else:
            output_values.append([]) # 체크되지 않은 상태는 빈 리스트 반환
        
    return output_values

# ... (나머지 코드 생략) ...


# 대시보드 워커 콜백 등록 (layouts/dashboard_worker.py에 정의된 콜백)
dashboard_worker.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8050))  # 환경변수에서 포트 읽기
#     app.run_server(host="0.0.0.0", port=port)

from dash import html, dcc

def side_layout(role):
    if role == "admin":
        links = [
            dcc.Link("관리자용 페이지", href="/admin", className="sidebar-link")
        ]
    elif role == "worker":
        links = [
            dcc.Link("공정 실시간 모니터링", href="/worker", className="sidebar-link"),
            dcc.Link("공정 누적 모니터링", href="/worker/analytics", className="sidebar-link")
        ]
    else:
        # 비로그인 상태 또는 role이 없음
        links = [
            dcc.Link("관리자용 대시보드", href="/admin", className="sidebar-link"),
            dcc.Link("작업자용 대시보드", href="/worker", className="sidebar-link"),
        ]

    return html.Div([
        html.H2("Menu", className="sidebar-title"),
        *links,
    ], className="sidebar")

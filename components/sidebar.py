from dash import html, dcc

def side_layout(role):
    if role == "admin":
        links = [
            dcc.Link("변수별 기여도", href="/admin", className="sidebar-link"),
            dcc.Link("변수별 인사이트 요약", href="/admin/model", className="sidebar-link"),
            dcc.Link("신규 필독서", href="/admin/final", className="sidebar-link"),
        ]
    elif role == "worker":
        links = [
            dcc.Link("실시간 데이터", href="/worker", className="sidebar-link"),
            dcc.Link("인수인계", href="/worker/analytics", className="sidebar-link")
        ]
    else:
        # 비로그인 상태 또는 role이 없음
        links = [
            dcc.Link("관리자용 대시보드", href="/admin", className="sidebar-link"),
            dcc.Link("작업자용 대시보드", href="/worker", className="sidebar-link"),
        ]

    return html.Div([
        html.H2("📋 Menu", className="sidebar-title"),
        *links,
    ], className="sidebar")

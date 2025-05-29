
from dash import html
import dash_bootstrap_components as dbc


def ad_layout():
    return html.Div([
        html.H3("관리자 메인 대시보드"),
        html.P("여기는 관리자용 기본 화면입니다."),
    ])

def model_layout():
    return html.Div([
        html.H3("예측 모델 페이지"),
        html.P("예측 모델 관련 내용이 표시됩니다."),
    ])

def final_layout():
    return html.Div([
        html.H3("결론 페이지"),
        html.P("프로젝트 결론 및 요약 내용입니다."),
    ])

def register_callbacks(app):
    # 콜백 등록 내용
    pass

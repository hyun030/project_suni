# -*- coding: utf-8 -*-
# 모든 Plotly 차트 생성 함수들을 관리합니다.

import pandas as pd
import streamlit as st
from .table import get_company_color

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
def create_sk_bar_chart(chart_df: pd.DataFrame):
    """SK에너지 강조 막대 차트"""
    if chart_df.empty or not PLOTLY_AVAILABLE:
        return None
    
    companies = chart_df['회사'].unique() if '회사' in chart_df.columns else []
    color_discrete_map = {company: get_company_color(company, companies) for company in companies}
    
    fig = px.bar(
        chart_df, x='지표', y='수치', color='회사',
        title="💼 SK에너지 vs 경쟁사 수익성 지표 비교",
        height=450, text='수치',
        color_discrete_map=color_discrete_map, barmode='group'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', textfont=dict(size=12))
    fig.update_layout(
        yaxis=dict(title="수치 (%)", title_font_size=14),
        xaxis=dict(title="재무 지표", title_font_size=14),
        legend_title="회사"
    )
    return fig

def create_sk_radar_chart(chart_df):
    """SK에너지 중심 레이더 차트 (지표별 Min-Max 정규화 적용)"""
    if chart_df.empty or not PLOTLY_AVAILABLE:
        return None
    
    companies = chart_df['회사'].unique() if '회사' in chart_df.columns else []
    metrics = chart_df['지표'].unique() if '지표' in chart_df.columns else []
    
    # 지표별 최소, 최대값 계산
    min_max = {}
    for metric in metrics:
        values = chart_df.loc[chart_df['지표'] == metric, '수치']
        min_val = values.min()
        max_val = values.max()
        # 최소 최대값이 같으면 max_val = min_val + 1로 설정(0 나누기 방지)
        if min_val == max_val:
            max_val = min_val + 1
        min_max[metric] = (min_val, max_val)
    
    fig = go.Figure()
    
    for i, company in enumerate(companies):
        company_data = chart_df[chart_df['회사'] == company] if '회사' in chart_df.columns else chart_df
        normalized_values = []
        for metric in metrics:
            raw_value = company_data.loc[company_data['지표'] == metric, '수치'].values
            if len(raw_value) == 0:
                norm_value = 0
            else:
                val = raw_value[0]
                min_val, max_val = min_max[metric]
                norm_value = (val - min_val) / (max_val - min_val)
            normalized_values.append(norm_value)
        
        # 닫힌 도형을 위해 첫 값 반복
        normalized_values.append(normalized_values[0])
        theta_labels = list(metrics) + [metrics[0]] if len(metrics) > 0 else ['지표1']
        
        # 색상
        color = get_company_color(company, companies)
        
        # SK에너지 스타일 강조
        if 'SK' in company:
            line_width = 5
            marker_size = 12
            name_style = f"**{company}**"
        else:
            line_width = 3
            marker_size = 8
            name_style = company
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=theta_labels,
            fill='toself',
            name=name_style,
            line=dict(width=line_width, color=color),
            marker=dict(size=marker_size, color=color)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],  # 정규화 했으니 0~1 범위
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=14)
            ),
            angularaxis=dict(
                tickfont=dict(size=16)
            )
        ),
        title="🎯 SK에너지 vs 경쟁사 수익성 지표 비교 (정규화)",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        ),
        title_font_size=20,
        font=dict(size=14)
    )
    
    return fig

def create_quarterly_trend_chart(quarterly_df: pd.DataFrame):
    """[복원] 회사별 분기별 추이 꺾은선 그래프"""
    if quarterly_df.empty or not PLOTLY_AVAILABLE: return None
    
    fig = go.Figure()
    companies = quarterly_df['회사'].unique()
    
    for company in companies:
        company_data = quarterly_df[quarterly_df['회사'] == company]
        line_color = get_company_color(company, companies)
        
        line_width, name_style = (4, f"**{company}**") if 'SK' in company else (2, company)
        
        # 영업이익률 추이
        if '영업이익률' in company_data.columns:
            fig.add_trace(go.Scatter(
                x=company_data['분기'], y=company_data['영업이익률'],
                mode='lines+markers', name=f"{name_style} 영업이익률(%)",
                line=dict(color=line_color, width=line_width)
            ))
    
    fig.update_layout(
        title="📈 분기별 영업이익률 추이",
        xaxis_title="분기", yaxis_title="영업이익률 (%)",
        height=500, hovermode='x unified'
    )
    return fig

def create_gap_trend_chart(quarterly_df: pd.DataFrame, target_metric='영업이익률'):
    """[개선] SK에너지와 경쟁사 간의 특정 지표 '갭' 추이를 시각화합니다."""
    if quarterly_df.empty or target_metric not in quarterly_df.columns: 
        st.warning(f"갭 분석을 위한 '{target_metric}' 데이터가 없습니다.")
        return None

    try:
        # SK에너지 데이터 추출
        sk_data = quarterly_df[quarterly_df['회사'].str.contains("SK", na=False, case=False)]
        if sk_data.empty:
            st.warning("SK에너지 데이터를 찾을 수 없습니다.")
            return None
            
        # 경쟁사 데이터 추출 및 평균 계산
        competitor_data = quarterly_df[~quarterly_df['회사'].str.contains("SK", na=False, case=False)]
        if competitor_data.empty:
            st.warning("경쟁사 데이터를 찾을 수 없습니다.")
            return None
            
        competitor_avg = competitor_data.groupby('분기')[target_metric].mean().reset_index()
        
        # 데이터 병합
        sk_selected = sk_data[['분기', target_metric]].rename(columns={target_metric: f'{target_metric}_sk'})
        gap_df = pd.merge(sk_selected, competitor_avg, on='분기', suffixes=('_sk', '_comp'))
        
        if gap_df.empty:
            st.warning("갭 분석을 위한 데이터 병합에 실패했습니다.")
            return None
            
        # 갭 계산
        gap_df['갭'] = gap_df[f'{target_metric}_sk'] - gap_df[f'{target_metric}_comp']
        
        # 차트 생성
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gap_df['분기'], 
            y=gap_df['갭'],
            mode='lines+markers',
            name=f'{target_metric} 갭',
            line=dict(color='#E31E24', width=3),
            marker=dict(size=8)
        ))
        
        # 0선 추가 (손익분기점)
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="손익분기점", annotation_position="top right")
        
        fig.update_layout(
            title=f"📊 SK에너지 vs 경쟁사 '{target_metric}' 갭(Gap) 추이 분석",
            xaxis_title="분기",
            yaxis_title=f"갭 크기 (SK - 경쟁사 평균)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"갭 분석 차트 생성 중 오류: {e}")
        return None

# -*- coding: utf-8 -*-
# 데이터를 특정 목적(차트, 테이블)에 맞게 분석하거나 준비하는 함수들을 관리합니다.

import pandas as pd
from data.loader import DartAPICollector

def prepare_chart_data(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()
    
    ratio_data = df[df['구분'].str.contains('%|점|억원', na=False)].copy()
    companies = [col for col in ratio_data.columns if col != '구분' and not col.endswith('_원시값')]
    
    chart_data_list = []
    for _, row in ratio_data.iterrows():
        for company in companies:
            value_str = str(row.get(company, '0')).replace('%', '').replace('점', '').replace('억원', '')
            try:
                value = float(value_str)
                chart_data_list.append({'지표': row['구분'], '회사': company, '수치': value})
            except (ValueError, TypeError):
                continue
    return pd.DataFrame(chart_data_list)

def create_dart_source_table(dart_collector: DartAPICollector, collected_companies: list, analysis_year: str):
    """DART 출처 정보 테이블 생성 (클릭 가능한 링크)"""
    if not hasattr(dart_collector, 'source_tracking') or not dart_collector.source_tracking:
        return pd.DataFrame()
    
    source_data = []
    for company, info in dart_collector.source_tracking.items():
        if company in collected_companies:
            # 유효한 DART 링크 생성
            rcept_no = info.get('rcept_no', 'N/A')
            if rcept_no and rcept_no != 'N/A':
                dart_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
            else:
                dart_url = "https://dart.fss.or.kr"
            
            source_data.append({
                '회사명': company,
                '보고서 유형': info.get('report_type', '재무제표'),
                '연도': info.get('year', analysis_year),
                '회사코드': info.get('company_code', 'N/A'),
                'DART 바로가기': dart_url,
                '접수번호': rcept_no
            })
    
    return pd.DataFrame(source_data)
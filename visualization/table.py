# -*- coding: utf-8 -*-
# 시각화(차트, 테이블)에 사용되는 헬퍼 함수들을 관리합니다.

from config import SK_COLORS

def get_company_color(company_name: str, all_companies: list) -> str:
    """회사 이름에 따라 고유 색상을 반환합니다. (오류 방지 기능 추가)"""
    
    # --- ▼▼▼ 오류 방지 코드 추가 ▼▼▼ ---
    # company_name이 문자열이 아니거나 비어있는 경우, 기본 색상을 반환하여 오류를 방지.
    if not isinstance(company_name, str) or not company_name:
        return SK_COLORS.get('competitor', '#6C757D') # 기본 경쟁사 색상

    if 'SK' in company_name:
        return SK_COLORS['primary']
    else:
        # 경쟁사들에게 서로 다른 파스텔 색상 할당
        competitor_colors = [
            SK_COLORS['competitor_1'],
            SK_COLORS['competitor_2'],
            SK_COLORS['competitor_3'],
            SK_COLORS['competitor_4'],
            SK_COLORS['competitor_5']
        ]
        
     # SK가 아닌 회사들의 인덱스 계산
        non_sk_companies = [comp for comp in all_companies if isinstance(comp, str) and 'SK' not in comp]
        try:
            index = non_sk_companies.index(company_name)
            return competitor_colors[index % len(competitor_colors)]
        except ValueError:
            return SK_COLORS.get('competitor', '#6C757D')
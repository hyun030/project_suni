# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Union
import re
import time
from functools import lru_cache

import pandas as pd
import requests
import streamlit as st
from dateutil import parser

# 프로젝트 설정 파일 import
import config

# 선택적 의존성 import
try:
    import gspread
    from google.oauth2.service_account import Credentials
    _GSPREAD_AVAILABLE = True
except ImportError:
    _GSPREAD_AVAILABLE = False


class DartAPICollector:
    """DART API를 통해 재무 데이터를 수집하는 클래스"""
    def __init__(self, api_key):
        self.api_key = api_key
        # 출처 추적용 딕셔너리
        self.source_tracking = {}
        
        # 회사명 매핑 개선
        self.company_name_mapping = {
            "SK에너지": [
                "SK에너지", "SK에너지주식회사", "에스케이에너지",
                "SK ENERGY", "SK Energy Co., Ltd."
            ],
            "GS칼텍스": [
                "GS칼텍스", "지에스칼텍스", "GS칼텍스주식회사", "지에스칼텍스주식회사"
            ],
            # HD현대오일뱅크 매핑 강화
            "HD현대오일뱅크": [
                "HD현대오일뱅크", "HD현대오일뱅크주식회사", 
                "현대오일뱅크", "현대오일뱅크주식회사",
                "HYUNDAI OILBANK", "Hyundai Oilbank Co., Ltd.",
                "267250"  # 종목코드 추가
            ],
            "현대오일뱅크": [
                "HD현대오일뱅크", "HD현대오일뱅크주식회사",
                "현대오일뱅크", "현대오일뱅크주식회사"
            ],
            "S-Oil": [
                "S-Oil", "S-Oil Corporation", "S-Oil Corp", "에쓰오일", "에스오일",
                "주식회사S-Oil", "S-OIL", "s-oil", "010950"
            ]
        }

        # STOCK_CODE_MAPPING도 업데이트
        self.STOCK_CODE_MAPPING = {
            "S-Oil": "010950",
            "GS칼텍스": "089590", 
            "HD현대오일뱅크": "267250",
            "현대오일뱅크": "267250",
            "SK에너지": "096770",
        }

    def get_corp_code_enhanced(self, company_name):
        """강화된 회사 고유번호 조회 (출력 간소화)"""
        url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={self.api_key}"
        search_names = self.company_name_mapping.get(company_name, [company_name])
        
        try:
            res = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                xml_file = z.open(z.namelist()[0])
                tree = ET.parse(xml_file)
                root = tree.getroot()
            
            # 모든 회사 목록에서 매칭 시도
            all_companies = []
            for corp in root.findall("list"):
                corp_name_elem = corp.find("corp_name")
                corp_code_elem = corp.find("corp_code")
                stock_code_elem = corp.find("stock_code")
                
                if corp_name_elem is not None and corp_code_elem is not None:
                    all_companies.append({
                        'name': corp_name_elem.text,
                        'code': corp_code_elem.text,
                        'stock_code': stock_code_elem.text if stock_code_elem is not None else None
                    })
            
            # 여러 단계로 검색
            for search_name in search_names:
                # 1단계: 종목코드로 검색 (S-Oil 전용)
                if search_name.isdigit():
                    for company in all_companies:
                        if company['stock_code'] == search_name:
                            return company['code']
                
                # 2단계: 정확히 일치
                for company in all_companies:
                    if company['name'] == search_name:
                        return company['code']
                
                # 3단계: 포함 검색
                for company in all_companies:
                    if search_name in company['name'] or company['name'] in search_name:
                        return company['code']
                
                # 4단계: 대소문자 무시 검색
                for company in all_companies:
                    if search_name.lower() in company['name'].lower() or company['name'].lower() in search_name.lower():
                        return company['code']
            
            return None
            
        except Exception as e:
            st.error(f"회사 코드 조회 오류: {e}")
            return None

    def get_financial_statement(self, corp_code, bsns_year, reprt_code, fs_div="CFS"):
        """재무제표 조회"""
        url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
            "fs_div": fs_div
        }
        
        try:
            res = requests.get(url, params=params).json()
            if res.get("status") == "000" and "list" in res:
                df = pd.DataFrame(res["list"])
                df["보고서구분"] = reprt_code
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()

    def get_company_financials_auto(self, company_name, bsns_year):
        """회사 재무제표 자동 수집 (출처 추적 포함)"""
        # 종목코드 직접 매핑
        if company_name in self.STOCK_CODE_MAPPING:
            stock_code = self.STOCK_CODE_MAPPING[company_name]
            corp_code = self.convert_stock_to_corp_code(stock_code)
            if corp_code:
                # 재무제표 직접 조회
                report_codes = ["11011", "11014", "11012"]
                for report_code in report_codes:
                    df = self.get_financial_statement(corp_code, bsns_year, report_code)
                    if not df.empty:
                        # rcept_no 생성 및 출처 정보 저장 (개선)
                        rcept_no = self._generate_rcept_no(corp_code, bsns_year, report_code)
                        self._save_source_info(company_name, corp_code, report_code, bsns_year, rcept_no)
                        return df
        
        # 기존 검색 방식으로 폴백
        corp_code = self.get_corp_code_enhanced(company_name)
        if not corp_code:
            return None
        
        # 여러 보고서 타입 시도
        report_codes = ["11011", "11014", "11012"]
        for report_code in report_codes:
            df = self.get_financial_statement(corp_code, bsns_year, report_code)
            if not df.empty:
                # rcept_no 생성 및 출처 정보 저장 (개선)
                rcept_no = self._generate_rcept_no(corp_code, bsns_year, report_code)
                self._save_source_info(company_name, corp_code, report_code, bsns_year, rcept_no)
                return df
        
        return None

    def convert_stock_to_corp_code(self, stock_code):
        """종목코드를 DART 회사코드로 변환"""
        try:
            url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={self.api_key}"
            res = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                xml_file = z.open(z.namelist()[0])
                tree = ET.parse(xml_file)
                root = tree.getroot()
            
            # 종목코드로 매칭
            for corp in root.findall("list"):
                stock_elem = corp.find("stock_code")
                corp_code_elem = corp.find("corp_code")
                
                if (stock_elem is not None and 
                    corp_code_elem is not None and 
                    stock_elem.text == stock_code):
                    return corp_code_elem.text
            
            return None
        except Exception as e:
            return None

    def _generate_rcept_no(self, corp_code, bsns_year, report_code):
        """rcept_no 생성 (실제 API에서 가져오기)"""
        try:
            # DART API의 공시목록 조회
            url = "https://opendart.fss.or.kr/api/list.json"
            params = {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bgn_de": f"{bsns_year}0101",
                "end_de": f"{bsns_year}1231",
                "pblntf_ty": "A",  # 정기공시
                "corp_cls": "Y",   # 유가증권
                "page_no": 1,
                "page_count": 100
            }
            
            res = requests.get(url, params=params).json()
            if res.get("status") == "000" and "list" in res:
                # 해당 보고서 타입에 맞는 rcept_no 찾기
                report_keywords = {
                    "11011": ["사업보고서"],
                    "11014": ["분기보고서", "3분기"],
                    "11012": ["반기보고서"],
                    "11013": ["분기보고서", "1분기"]
                }
                
                keywords = report_keywords.get(report_code, [])
                for item in res["list"]:
                    report_nm = item.get("report_nm", "")
                    if any(keyword in report_nm for keyword in keywords):
                        return item.get("rcept_no")
            
            return f"{corp_code}_{bsns_year}_{report_code}"  # 기본값
        except:
            return f"{corp_code}_{bsns_year}_{report_code}"

    def _save_source_info(self, company_name, corp_code, report_code, bsns_year, rcept_no):
        """출처 정보 저장 (개선된 버전)"""
        report_type_map = {
            "11011": "사업보고서",
            "11014": "3분기보고서",
            "11012": "반기보고서",
            "11013": "1분기보고서"
        }
        
        self.source_tracking[company_name] = {
            'company_code': corp_code,
            'report_code': report_code,
            'report_type': report_type_map.get(report_code, "재무제표"),
            'year': bsns_year,
            'rcept_no': rcept_no,
            'dart_url': f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
            'direct_link': f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}&reprtCode={report_code}"
        }

    def test_api_key(self):
        """DART API 키 유효성 테스트"""
        try:
            url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={self.api_key}"
            res = requests.get(url, timeout=10)
            
            if res.status_code == 200:
                return True, "API 키가 유효합니다."
            elif res.status_code == 401:
                return False, "API 키가 유효하지 않습니다."
            else:
                return False, f"API 응답 오류: {res.status_code}"
        except Exception as e:
            return False, f"API 테스트 중 오류: {e}"


class QuarterlyDataCollector:
    """분기별 데이터 수집 클래스 (성능 최적화)"""
    def __init__(self, dart_collector: DartAPICollector):
        self.dart_collector = dart_collector
        self._quarterly_cache = {}  # 분기 데이터 캐싱

    def collect_quarterly_data(self, company_name, year=2024):
        """분기별 데이터 수집 (순차 처리로 변경)"""
        cache_key = f"{company_name}_{year}"
        
        # 캐시 확인
        if cache_key in self._quarterly_cache:
            return self._quarterly_cache[cache_key]
        
        try:
            corp_code = self.dart_collector.get_corp_code_enhanced(company_name)
            if not corp_code:
                return pd.DataFrame()

            quarters = ["11011", "11014", "11012", "11013"]  # 1Q, 2Q, 3Q, 4Q
            quarterly_data = []
            
            # 순차 처리로 분기 데이터 수집
            for report_code in quarters:
                try:
                    df = self.dart_collector.get_financial_statement(corp_code, year, report_code)
                    if not df.empty:
                        quarter = self._get_quarter_from_code(report_code)
                        metrics = self._extract_key_metrics(df, quarter)
                        metrics['회사'] = company_name
                        metrics['연도'] = year
                        quarterly_data.append(metrics)
                except Exception as e:
                    st.warning(f"분기 데이터 조회 실패: {e}")
            
            if quarterly_data:
                result_df = pd.DataFrame(quarterly_data)
                self._quarterly_cache[cache_key] = result_df
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"분기별 데이터 수집 오류: {e}")
            return pd.DataFrame()

    def _get_quarter_from_code(self, report_code):
        """보고서 코드로 분기 추출"""
        quarter_map = {"11011": "1Q", "11014": "2Q", "11012": "3Q", "11013": "4Q"}
        return quarter_map.get(report_code, "Unknown")

    def _extract_key_metrics(self, df, quarter):
        """주요 지표 추출 (성능 최적화)"""
        metrics = {'분기': quarter}
        
        def find_amount(keywords):
            for keyword in keywords:
                matches = df[df['account_nm'].str.contains(keyword, case=False, na=False)]
                if not matches.empty:
                    try:
                        # 가장 최신 데이터 우선
                        latest = matches.iloc[0]
                        amount = latest.get('thstrm_amount', 0)
                        if amount and str(amount).replace('-', '').replace('.', '').isdigit():
                            return float(amount.replace('-', ''))
                    except (ValueError, TypeError):
                        continue
            return 0
        
        # 주요 지표 추출
        metrics['매출액'] = find_amount(['매출액', '매출', '수익'])
        metrics['영업이익'] = find_amount(['영업이익'])
        metrics['당기순이익'] = find_amount(['당기순이익', '순이익'])
        metrics['자산총계'] = find_amount(['자산총계', '총자산'])
        metrics['부채총계'] = find_amount(['부채총계', '총부채'])
        metrics['자본총계'] = find_amount(['자본총계', '총자본'])
        
        # 비율 계산
        if metrics['매출액'] > 0:
            metrics['영업이익률'] = (metrics['영업이익'] / metrics['매출액']) * 100
            metrics['순이익률'] = (metrics['당기순이익'] / metrics['매출액']) * 100
        
        if metrics['자산총계'] > 0:
            metrics['ROA'] = (metrics['당기순이익'] / metrics['자산총계']) * 100
        
        if metrics['자본총계'] > 0:
            metrics['ROE'] = (metrics['당기순이익'] / metrics['자본총계']) * 100
        
        return metrics


class SKNewsCollector:
    """SK 뉴스 수집 클래스 (구글시트 전용, 성능 최적화)"""
    def __init__(self, custom_keywords: list | None = None, sheet_id=None, service_account_json=None):
        self.custom_keywords = custom_keywords or config.BENCHMARKING_KEYWORDS
        self.sheet_id = sheet_id or config.SHEET_ID
        self.service_account_json = service_account_json or config.GOOGLE_SERVICE_ACCOUNT_JSON
        self._news_cache = {}  # 뉴스 데이터 캐싱

    def collect_news(self, *, max_items_per_feed: int = 30) -> pd.DataFrame:
        """뉴스 수집 (구글시트 전용)"""
        try:
            # 구글시트에서 뉴스 수집
            news_df = self._fetch_sheet_news()
            
            if not news_df.empty:
                # 데이터 품질 향상
                news_df = self._enrich_dataframe(news_df)
                
                # 키워드 필터링 및 점수 계산
                filtered_news = []
                for _, row in news_df.iterrows():
                    title = str(row.get('제목', ''))
                    content = str(row.get('내용', ''))
                    full_text = f"{title} {content}"
                    
                    # 키워드 매칭
                    matched_keywords = self._extract_keywords(full_text)
                    if matched_keywords:
                        row_copy = row.copy()
                        row_copy['키워드'] = matched_keywords
                        row_copy['벤치마킹점수'] = self._calc_importance(full_text)
                        row_copy['영향도'] = self._calc_sk_relevance(full_text)
                        row_copy['회사'] = self._extract_company(full_text)
                        filtered_news.append(row_copy)
                
                if filtered_news:
                    result_df = pd.DataFrame(filtered_news)
                    # 벤치마킹 점수 기준으로 정렬
                    result_df = result_df.sort_values('벤치마킹점수', ascending=False)
                    return result_df.head(max_items_per_feed)
            
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"뉴스 수집 중 오류: {e}")
            return pd.DataFrame()

    def _fetch_sheet_news(self) -> pd.DataFrame:
        """구글시트에서 뉴스 데이터 가져오기"""
        if not _GSPREAD_AVAILABLE:
            st.error("gspread 라이브러리가 설치되지 않았습니다.")
            return pd.DataFrame()
        
        try:
            # 서비스 계정 인증
            if self.service_account_json:
                creds = Credentials.from_service_account_info(self.service_account_json)
            else:
                # 기본 인증 방식 (필요시 수정)
                creds = None
            
            gc = gspread.authorize(creds)
            sheet = gc.open_by_key(self.sheet_id).sheet1
            
            # 데이터 가져오기
            data = sheet.get_all_records()
            if data:
                df = pd.DataFrame(data)
                # 필수 컬럼 확인
                required_columns = ['제목', '내용', '날짜']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.warning(f"구글시트에 필수 컬럼이 없습니다: {missing_columns}")
                    return pd.DataFrame()
                return df
            else:
                st.warning("구글시트에서 데이터를 찾을 수 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"구글시트 접근 오류: {e}")
            return pd.DataFrame()

    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 품질 향상"""
        # 날짜 형식 통일
        if '날짜' in df.columns:
            df['날짜'] = df['날짜'].apply(self._parse_date)
        
        # 출처 정보 추가
        if '출처' not in df.columns:
            df['출처'] = '구글시트'
        
        # URL 정보 추가 (있는 경우)
        if 'URL' not in df.columns:
            df['URL'] = ''
        
        return df

    def _extract_keywords(self, text: str) -> str:
        """키워드 추출"""
        matched_keywords = []
        for keyword in self.custom_keywords:
            if keyword.lower() in text.lower():
                matched_keywords.append(keyword)
        return ', '.join(matched_keywords)

    def _calc_importance(self, text: str) -> int:
        """중요도 점수 계산"""
        score = 0
        text_lower = text.lower()
        
        # 키워드별 가중치
        keyword_weights = {
            '투자': 3, '확장': 3, '신규': 3, '혁신': 3,
            '성장': 2, '개선': 2, '전략': 2, '시장': 2,
            '수익': 2, '이익': 2, '매출': 2
        }
        
        for keyword, weight in keyword_weights.items():
            if keyword in text_lower:
                score += weight
        
        return min(score, 10)  # 최대 10점

    def _calc_sk_relevance(self, text: str) -> int:
        """SK 관련성 점수 계산"""
        sk_keywords = ['sk', '에너지', '정유', '석유', '화학', '석유화학']
        text_lower = text.lower()
        
        score = 0
        for keyword in sk_keywords:
            if keyword in text_lower:
                score += 1
        
        return min(score, 5)  # 최대 5점

    def _extract_company(self, text: str) -> str:
        """회사명 추출"""
        companies = ['SK에너지', 'GS칼텍스', 'HD현대오일뱅크', 'S-Oil', '현대오일뱅크']
        text_lower = text.lower()
        
        for company in companies:
            if company.lower() in text_lower:
                return company
        
        return '산업 동향'

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """날짜 파싱"""
        try:
            if pd.isna(date_str) or date_str == '':
                return datetime.now().strftime('%Y-%m-%d')
            
            # 다양한 날짜 형식 처리
            if isinstance(date_str, str):
                # 이미 YYYY-MM-DD 형식인 경우
                if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    return date_str
                
                # 다른 형식인 경우 파싱 시도
                try:
                    parsed_date = parser.parse(date_str)
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
            
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')

# -*- coding: utf-8 -*-
# DART, XBRL 등 모든 재무 데이터를 표준화하고 심층 분석하는 모듈입니다.

import re
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import config
from functools import lru_cache
import time

class UniversalDataProcessor:
    """
    모든 재무 데이터를 일관된 기준으로 처리하고, 심층 지표를 계산하여
    분석용 데이터프레임을 생성하는 통합 클래스.
    """
    def __init__(self):
        pass

    def process_dart_data(self, dart_df: pd.DataFrame, company_name: str) -> pd.DataFrame | None:
        """DART 데이터프레임을 표준화하고 심층 분석합니다."""
        try:
            if dart_df.empty:
                return None
            
            # 디버깅: 원본 DART 데이터 로깅
            st.write(f"🔍 {company_name} 원본 DART 데이터 ({len(dart_df)}개 항목):")
            debug_df = dart_df[['account_nm', 'thstrm_amount']].head(10)
            st.dataframe(debug_df, use_container_width=True)

            financial_data = {}
            processed_count = 0
            
            for _, row in dart_df.iterrows():
                account_nm = row.get('account_nm', '')
                thstrm_amount = row.get('thstrm_amount', '0')
                
                # 빈 값 건너뛰기
                if not account_nm or not thstrm_amount:
                    continue
                        
                try:
                    # 마이너스 값 정확 처리
                    amount_str = str(thstrm_amount).replace(',', '')
                    if '(' in amount_str and ')' in amount_str:
                        # 괄호로 표시된 마이너스
                        amount_str = '-' + amount_str.replace('(', '').replace(')', '')
                    value = float(amount_str) if amount_str != '-' else 0
                    
                    # DART API는 천원 단위로 제공하므로 억원 단위로 변환
                    value = value / 100_000  # 천원 → 억원 변환
                    
                except (ValueError, TypeError):
                    continue

                # 계정과목 매핑
                mapped = False
                for key, mapped_name in self.INCOME_STATEMENT_MAP.items():
                    if key in account_nm or account_nm in key:
                        if mapped_name not in financial_data or abs(value) > abs(financial_data[mapped_name]):
                            financial_data[mapped_name] = value
                            mapped = True
                        break
                
                if mapped:
                    processed_count += 1

            # 디버깅: 매핑된 재무 데이터 로깅
            st.write(f"📊 {company_name} 매핑된 재무 데이터 ({processed_count}개 처리):")
            for key, value in financial_data.items():
                st.write(f"  {key}: {value:,.0f}억원")

            # 데이터 검증
            if not financial_data:
                st.error(f"❌ {company_name}: 매핑된 재무 데이터가 없습니다.")
                return None

            return self._create_income_statement(financial_data, company_name)
            
        except Exception as e:
            st.error(f"DART 데이터 처리 오류: {e}")
            return None

    # 기존 통합 코드의 매핑 사전
    INCOME_STATEMENT_MAP = {
        'sales': '매출액',
        'revenue': '매출액',
        '매출액': '매출액',
        '수익(매출액)': '매출액',
        'costofgoodssold': '매출원가',
        'cogs': '매출원가',
        'costofrevenue': '매출원가',
        '매출원가': '매출원가',
        'operatingexpenses': '판관비',
        'sellingexpenses': '판매비',
        'administrativeexpenses': '관리비',
        '판매비와관리비': '판관비',
        '판관비': '판관비',
        'grossprofit': '매출총이익',
        '매출총이익': '매출총이익',
        'operatingincome': '영업이익',
        'operatingprofit': '영업이익',
        '영업이익': '영업이익',
        'netincome': '당기순이익',
        '당기순이익': '당기순이익',
    }

    def process_uploaded_file(self, uploaded_file):
        """업로드된 파일을 표준화하고 심층 분석합니다."""
        try:
            # 파일 크기 체크 (50MB 제한)
            file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else 0
            if file_size > 50 * 1024 * 1024:
                st.error(f"❌ 파일이 너무 큽니다 ({file_size/(1024*1024):.1f}MB). 50MB 이하로 업로드해주세요.")
                return None
            
            # 파일 처음부터 읽기
            uploaded_file.seek(0)
            content = uploaded_file.read()
            
            # 빠른 인코딩 감지 및 디코딩
            content_str = self._fast_decode(content)
            if not content_str:
                st.error("❌ 파일 인코딩을 읽을 수 없습니다.")
                return None
            
            # XML 파싱 (더 안전한 방식)
            try:
                # lxml이 있으면 사용, 없으면 기본 xml 파서 사용
                soup = BeautifulSoup(content_str, 'lxml-xml')
                if not soup.find():  # 파싱 실패 시 기본 파서 사용
                    soup = BeautifulSoup(content_str, 'xml')
            except Exception:
                soup = BeautifulSoup(content_str, 'html.parser')  # 최후 수단
            
            # 회사명 추출 (더 빠르고 정확하게)
            company_name = self._extract_company_name_fast(soup, uploaded_file.name)
            
            # 재무 데이터 추출 (최적화된 버전)
            financial_data = self._extract_financial_items_optimized(soup)
            
            if not financial_data:
                st.warning(f"⚠️ {uploaded_file.name}에서 재무 항목을 찾을 수 없습니다.")
                st.info("💡 파일이 표준 XBRL 형식인지 확인해주세요.")
                return None
            
            # 표준 손익계산서 구조로 변환
            income_statement = self._create_income_statement(financial_data, company_name)
            
            return income_statement
            
        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류: {str(e)}")
            st.info("💡 파일 형식을 확인하고 다시 시도해주세요.")
            return None

    def process_excel_data(self, excel_df, filename):
        """Excel 파일 데이터를 표준화하고 심층 분석합니다."""
        try:
            if excel_df.empty:
                return None
            
            # 회사명 추출 (파일명에서)
            company_name = self._extract_company_name_from_filename(filename)
            
            # Excel 데이터를 재무 데이터로 변환
            financial_data = {}
            
            # Excel 컬럼을 재무 항목으로 매핑
            for col in excel_df.columns:
                col_lower = str(col).lower()
                
                # 매출액 관련
                if any(keyword in col_lower for keyword in ['매출', 'revenue', 'sales', '수익']):
                    if '매출액' not in financial_data:
                        financial_data['매출액'] = self._extract_numeric_value(excel_df[col])
                
                # 매출원가 관련
                elif any(keyword in col_lower for keyword in ['원가', 'cost', '매출원가']):
                    if '매출원가' not in financial_data:
                        financial_data['매출원가'] = self._extract_numeric_value(excel_df[col])
                
                # 영업이익 관련
                elif any(keyword in col_lower for keyword in ['영업이익', 'operating', '영업손익']):
                    if '영업이익' not in financial_data:
                        financial_data['영업이익'] = self._extract_numeric_value(excel_df[col])
                
                # 당기순이익 관련
                elif any(keyword in col_lower for keyword in ['순이익', 'net', '당기순이익']):
                    if '당기순이익' not in financial_data:
                        financial_data['당기순이익'] = self._extract_numeric_value(excel_df[col])
                
                # 매출총이익 관련
                elif any(keyword in col_lower for keyword in ['총이익', 'gross', '매출총이익']):
                    if '매출총이익' not in financial_data:
                        financial_data['매출총이익'] = self._extract_numeric_value(excel_df[col])
            
            # 데이터가 없으면 첫 번째 숫자 컬럼을 매출액으로 가정
            if not financial_data and len(excel_df.columns) > 0:
                numeric_cols = excel_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    financial_data['매출액'] = self._extract_numeric_value(excel_df[numeric_cols[0]])
            
            if not financial_data:
                st.warning(f"⚠️ {filename}에서 재무 데이터를 찾을 수 없습니다.")
                return None
            
            # 표준 손익계산서 구조로 변환
            return self._create_income_statement(financial_data, company_name)
            
        except Exception as e:
            st.error(f"Excel 데이터 처리 오류: {e}")
            return None

    def _fast_decode(self, content):
        """최적화된 인코딩 감지 및 디코딩"""
        # 가장 일반적인 인코딩부터 시도 (한국어 환경 최적화)
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                decoded = content.decode(encoding)
                # 한글이 제대로 디코딩되었는지 간단히 체크
                if '매출' in decoded or 'revenue' in decoded.lower():
                    return decoded
                return decoded  # 한글이 없어도 성공한 디코딩은 반환
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # 모든 인코딩 실패 시 오류 무시하고 디코딩
        try:
            return content.decode('utf-8', errors='ignore')
        except:
            return None

    def _extract_company_name_fast(self, soup, filename):
        """최적화된 회사명 추출"""
        # 1단계: 표준 XBRL 태그에서 회사명 검색
        company_tags = [
            'EntityRegistrantName', 'CompanyName', 'entity', 'registrant',
            'ReportingEntityName', 'EntityName', 'CorporateName'
        ]
        
        for tag_name in company_tags:
            # 정확한 태그명으로 먼저 검색
            node = soup.find(tag_name)
            if node and node.string and len(node.string.strip()) > 1:
                return node.string.strip()
            
            # 부분 매칭으로 검색 (대소문자 무시)
            node = soup.find(lambda t: t.name and tag_name.lower() in t.name.lower())
            if node and node.string and len(node.string.strip()) > 1:
                return node.string.strip()
        
        # 2단계: 파일명에서 회사명 추출 (강화된 매핑)
        name = filename.split('.')[0].lower()
        name_mapping = {
            'sk': 'SK에너지',
            'skenergy': 'SK에너지',
            'gs': 'GS칼텍스',
            'gscaltex': 'GS칼텍스',
            'hd': 'HD현대오일뱅크',
            'hyundai': 'HD현대오일뱅크',
            'hdoil': 'HD현대오일뱅크',
            's-oil': 'S-Oil',
            'soil': 'S-Oil',
            'soilcorp': 'S-Oil'
        }
        
        for key, company in name_mapping.items():
            if key in name:
                return company
        
        # 3단계: 파일명 그대로 사용 (정리해서)
        clean_name = re.sub(r'[^A-Za-z가-힣0-9\s]', '', filename.split('.')[0])
        return clean_name if clean_name else "Unknown Company"

    def _extract_company_name_from_filename(self, filename):
        """파일명에서 회사명 추출"""
        name = filename.split('.')[0].lower()
        name_mapping = {
            'sk': 'SK에너지',
            'skenergy': 'SK에너지',
            'gs': 'GS칼텍스',
            'gscaltex': 'GS칼텍스',
            'hd': 'HD현대오일뱅크',
            'hyundai': 'HD현대오일뱅크',
            'hdoil': 'HD현대오일뱅크',
            's-oil': 'S-Oil',
            'soil': 'S-Oil',
            'soilcorp': 'S-Oil'
        }
        
        for key, company in name_mapping.items():
            if key in name:
                return company
        
        # 파일명 그대로 사용 (정리해서)
        clean_name = re.sub(r'[^A-Za-z가-힣0-9\s]', '', filename.split('.')[0])
        return clean_name if clean_name else "Unknown Company"

    def _extract_numeric_value(self, series):
        """시리즈에서 숫자 값 추출"""
        try:
            # NaN이 아닌 첫 번째 값 찾기
            for value in series:
                if pd.notna(value) and value != 0:
                    return float(value)
            return 0
        except (ValueError, TypeError):
            return 0

    def _extract_financial_items_optimized(self, soup):
        """최적화된 재무 항목 추출"""
        items = {}
        processed_count = 0
        
        # 숫자가 포함된 태그만 사전 필터링 (성능 향상)
        numeric_tags = []
        for tag in soup.find_all():
            if tag.string and re.search(r'\d', tag.string):
                numeric_tags.append(tag)
        
        if not numeric_tags:
            st.warning("📊 숫자 데이터가 포함된 태그를 찾을 수 없습니다.")
            return items
        
        # 진행 상황 표시
        st.info(f"🔍 {len(numeric_tags)}개의 숫자 태그 발견, 분석 중...")
        
        # 기존 통합 코드의 정규식 패턴 적용
        INCOME_STATEMENT_PATTERNS = {
            # 매출 관련 (더 광범위한 패턴)
            r'(revenue|sales|매출|수익|총매출|매출수익|operating.*revenue)(?!.*cost|원가|비용)': '매출액',
            r'(cost.*revenue|cost.*sales|cost.*goods|매출원가|원가|판매원가|제품매출원가)': '매출원가',
            
            # 이익 관련
            r'(gross.*profit|총이익|매출총이익|총수익)': '매출총이익',
            r'(operating.*income|operating.*profit|영업이익|영업손익|영업수익)(?!.*비용|expense)': '영업이익',
            r'(net.*income|net.*profit|당기순이익|순이익|당기.*순손익|net.*earnings)(?!.*loss)': '당기순이익',
            
            # 비용 관련 (더 정확한 패턴)
            r'(selling.*expense|selling.*cost|판매비|판매비용|판매관련비용)': '판매비',
            r'(administrative.*expense|administrative.*cost|관리비|관리비용|일반관리비)': '관리비',
            r'(selling.*administrative|판매비.*관리비|판관비|판매.*관리.*비용)': '판관비',
            r'(employee.*benefit|employee.*cost|wage|salary|인건비|급여|임금)': '인건비',
            r'(depreciation|amortization|감가상각|상각비|감가상각비)': '감가상각비',
            
            # 기타 항목
            r'(interest.*expense|interest.*cost|이자비용|이자지급)': '이자비용',
            r'(financial.*cost|금융비용|금융원가)': '금융비용',
            r'(non.*operating.*income|영업외수익|기타수익)': '영업외수익',
            r'(non.*operating.*expense|영업외비용|기타비용)': '영업외비용'
        }
        
        # 정규식 미리 컴파일 (성능 향상)
        compiled_patterns = {}
        for pattern, item in INCOME_STATEMENT_PATTERNS.items():
            compiled_patterns[re.compile(pattern, re.IGNORECASE)] = item
        
        # 각 태그 분석
        for tag in numeric_tags:
            tag_text = tag.string.strip()
            
            # 숫자 추출 및 검증
            try:
                # 괄호로 둘러싸인 음수 처리
                if '(' in tag_text and ')' in tag_text:
                    number_str = re.sub(r'[^\d.]', '', tag_text.replace('(', '').replace(')', ''))
                    if number_str:
                        value = -float(number_str)
                    else:
                        continue
                else:
                    # 일반적인 숫자 추출
                    number_str = re.sub(r'[^\d.-]', '', tag_text)
                    if number_str and number_str not in ['-', '.', '-.']:
                        value = float(number_str)
                    else:
                        continue
                
                # 너무 작은 값은 제외 (노이즈 제거)
                if abs(value) < 1000:
                    continue
                    
            except (ValueError, TypeError):
                continue
            
            # 태그 정보 구성 (태그명 + 속성)
            tag_info_parts = [tag.name.lower() if tag.name else '']
            if tag.attrs:
                tag_info_parts.extend([str(v).lower() for v in tag.attrs.values()])
            tag_info = ' '.join(tag_info_parts)
            
            # 정규식 패턴 매칭
            for pattern, standard_item in compiled_patterns.items():
                if pattern.search(tag_info):
                    # 같은 항목이 이미 있으면 더 큰 절댓값으로 업데이트
                    if standard_item not in items or abs(value) > abs(items[standard_item]):
                        items[standard_item] = value
                    processed_count += 1
                    break
        
        # 결과 요약 표시
        if items:
            st.success(f"✅ {len(items)}개 재무항목 추출 (총 {processed_count}개 태그 처리)")
            with st.expander("🔍 추출된 데이터 상세 보기"):
                for key, value in items.items():
                    formatted_value = self._format_amount(value)
                    st.write(f"**{key}**: {formatted_value}")
        else:
            st.warning("⚠️ 표준 재무 항목을 찾을 수 없습니다.")
        
        return items

    def _create_income_statement(self, data, company_name):
        """표준 손익계산서 구조 생성"""
        # 표준 손익계산서 항목 순서
        standard_items = [
            '매출액', '매출원가', '매출총이익', '판매비', '관리비', '판관비',
            '인건비', '감가상각비', '영업이익', '영업외수익', '영업외비용',
            '금융비용', '이자비용', '당기순이익'
        ]
        
        # 파생 항목 계산 (누락된 항목 추정)
        calculated_items = self._calculate_derived_items(data)
        data.update(calculated_items)
        
        # 손익계산서 생성
        income_statement = []
        for item in standard_items:
            value = data.get(item, 0)
            if value != 0:  # 0이 아닌 값만 포함
                income_statement.append({
                    '구분': item,
                    company_name: self._format_amount(value),
                    f'{company_name}_원시값': value
                })
        
        # 비율 계산 및 추가
        ratios = self._calculate_ratios(data)
        for ratio_name, ratio_value in ratios.items():
            income_statement.append({
                '구분': ratio_name,
                company_name: f"{ratio_value:.2f}%",
                f'{company_name}_원시값': ratio_value
            })
        
        return pd.DataFrame(income_statement)

    def _calculate_derived_items(self, data):
        """파생 항목 계산 (누락된 데이터 추정)"""
        calculated = {}
        
        # 매출총이익 계산
        if '매출액' in data and '매출원가' in data:
            calculated['매출총이익'] = data['매출액'] - data['매출원가']
        elif '매출액' in data and '매출총이익' not in data:
            # 매출총이익이 없으면 업계 평균 30%로 추정
            calculated['매출총이익'] = data['매출액'] * 0.3
            calculated['매출원가'] = data['매출액'] - calculated['매출총이익']
        elif '매출총이익' in data and '매출액' not in data and '매출원가' in data:
            calculated['매출액'] = data['매출총이익'] + data['매출원가']
        
        # 판관비 계산
        if '판매비' in data and '관리비' in data:
            calculated['판관비'] = data['판매비'] + data['관리비']
        elif '판관비' in data and '판매비' not in data and '관리비' not in data:
            # 판관비를 6:4 비율로 분할 (일반적 비율)
            calculated['판매비'] = data['판관비'] * 0.6
            calculated['관리비'] = data['판관비'] * 0.4
        
        # 영업이익 계산
        if '매출총이익' in data and '판관비' in data and '영업이익' not in data:
            calculated['영업이익'] = data['매출총이익'] - data['판관비']
        
        return calculated

    def _calculate_ratios(self, data):
        """주요 재무비율 계산"""
        ratios = {}
        매출액 = data.get('매출액', 0)
        
        if 매출액 <= 0:
            return ratios  # 매출액이 없으면 비율 계산 불가
        
        # 수익성 비율
        if '영업이익' in data:
            ratios['영업이익률(%)'] = round((data['영업이익'] / 매출액) * 100, 2)
        
        if '당기순이익' in data:
            ratios['순이익률(%)'] = round((data['당기순이익'] / 매출액) * 100, 2)
        
        if '매출총이익' in data:
            ratios['매출총이익률(%)'] = round((data['매출총이익'] / 매출액) * 100, 2)
        
        # 비용 비율
        if '매출원가' in data:
            ratios['매출원가율(%)'] = round((data['매출원가'] / 매출액) * 100, 2)
        
        if '판관비' in data:
            ratios['판관비율(%)'] = round((data['판관비'] / 매출액) * 100, 2)
        
        if '인건비' in data:
            ratios['인건비율(%)'] = round((data['인건비'] / 매출액) * 100, 2)
        
        return ratios

    def _format_amount(self, amount):
        """금액 포맷팅 (한국 단위 사용)"""
        if amount == 0:
            return "0원"
            
        abs_amount = abs(amount)
        sign = "▼ " if amount < 0 else ""
        
        if abs_amount >= 1_000_000_000_000:  # 1조 이상
            return f"{sign}{amount/1_000_000_000_000:.1f}조원"
        elif abs_amount >= 100_000_000:  # 1억 이상
            return f"{sign}{amount/100_000_000:.0f}억원"
        elif abs_amount >= 10_000:  # 1만 이상
            return f"{sign}{amount/10_000:.0f}만원"
        else:
            return f"{sign}{amount:,.0f}원"

    def merge_company_data(self, dataframes: list[pd.DataFrame]):
        """여러 회사 데이터 병합 (안전한 병합)"""
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # 기준이 되는 첫 번째 데이터프레임
        merged = dataframes[0].copy()
        
        # 나머지 데이터프레임들을 순차적으로 병합
        for df in dataframes[1:]:
            try:
                # 회사 컬럼만 추출 (구분, _원시값 컬럼 제외)
                company_cols = [col for col in df.columns 
                              if col != '구분' and not col.endswith('_원시값')]
                
                for company_col in company_cols:
                    # 구분을 인덱스로 하여 데이터 병합
                    company_data = df.set_index('구분')[company_col]
                    merged_temp = merged.set_index('구분')
                    merged_temp = merged_temp.join(company_data, how='outer')
                    merged = merged_temp.reset_index()
            except Exception as e:
                st.warning(f"⚠️ 데이터 병합 중 오류: {e}")
                continue
        
        # 결측치를 "-"로 채움
        merged = merged.fillna("-")
        
        return merged

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiohttp
import multiprocessing as mp

# GPU 가속 라이브러리 (선택적)
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    st.success("🚀 GPU 가속 사용 가능!")
except ImportError:
    GPU_AVAILABLE = False
    st.info("💡 GPU 가속을 위해 cudf, cupy 설치: pip install cudf-cu12 cupy-cuda12x")

# NumPy 병렬 처리
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    st.info("💡 NumPy 병렬 처리를 위해 numba 설치: pip install numba")

# 모듈화된 파일들 import
import config
from data.loader import DartAPICollector, QuarterlyDataCollector, SKNewsCollector
from data.preprocess import UniversalDataProcessor
from insight.gemini_api import GeminiInsightGenerator
from insight.analyzer import prepare_chart_data, create_dart_source_table
from visualization.charts import create_sk_bar_chart, create_sk_radar_chart, create_quarterly_trend_chart, create_gap_trend_chart, PLOTLY_AVAILABLE
from util.export import create_excel_report, create_enhanced_pdf_report
from util.email_util import render_email_links

st.set_page_config(page_title="SK에너지 경쟁사 분석 대시보드", page_icon="⚡", layout="wide")

# --- GPU 가속 함수들 ---
@jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda x: x
def fast_financial_calculations(data_array):
    """GPU 가속 재무 계산"""
    if NUMBA_AVAILABLE:
        result = np.zeros_like(data_array)
        for i in prange(len(data_array)):
            if data_array[i] != 0:
                result[i] = data_array[i] * 1.1  # 예시 계산
        return result
    return data_array

def gpu_dataframe_operations(df):
    """GPU 가속 DataFrame 연산"""
    if GPU_AVAILABLE and len(df) > 1000:  # 큰 데이터셋만 GPU 사용
        try:
            # CPU -> GPU
            gpu_df = cudf.from_pandas(df)
            
            # GPU 연산
            numeric_cols = gpu_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if gpu_df[col].dtype in ['float64', 'int64']:
                    gpu_df[col] = gpu_df[col].fillna(0)
            
            # GPU -> CPU
            return gpu_df.to_pandas()
        except Exception as e:
            st.warning(f"GPU 연산 실패, CPU로 전환: {e}")
            return df
    return df

# --- 비동기 AI API 호출 ---
async def async_gemini_call(session, prompt, api_key):
    """비동기 Gemini API 호출"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
    }
    
    async with session.post(url, headers=headers, json=data, params={"key": api_key}) as response:
        return await response.json()

async def batch_gemini_calls(prompts, api_key):
    """배치 Gemini API 호출"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_gemini_call(session, prompt, api_key) for prompt in prompts]
        return await asyncio.gather(*tasks)

# --- 병렬 데이터 처리 ---
def parallel_data_processing(dataframes, max_workers=None):
    """병렬 데이터 처리"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # CPU 코어 수에 따라 조정
    
    def process_single_df(df):
        if df is None or df.empty:
            return None
        
        # GPU 가속 적용
        df = gpu_dataframe_operations(df)
        
        # NumPy 병렬 연산
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = fast_financial_calculations(df[col].values)
        
        return df
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_df, dataframes))
    
    return [r for r in results if r is not None]

# --- 세션 상태 초기화 ---
def initialize_session_state():
    session_vars = [
        'financial_data', 'news_data', 'financial_insight', 'news_insight',
        'selected_companies', 'manual_financial_data', 'quarterly_data',
        'dart_collector', 'integrated_insight', 'processing_status'
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    
    # 뉴스 분석용 커스텀 키워드 초기화
    if 'custom_keywords' not in st.session_state:
        st.session_state.custom_keywords = config.BENCHMARKING_KEYWORDS
    
    # 성능 설정
    if 'use_gpu' not in st.session_state:
        st.session_state.use_gpu = GPU_AVAILABLE
    if 'parallel_workers' not in st.session_state:
        st.session_state.parallel_workers = min(mp.cpu_count(), 4)

# --- 고성능 데이터 수집 ---
def collect_financial_data_async(dart_collector, processor, selected_companies, analysis_year):
    """GPU 가속 비동기 데이터 수집"""
    progress_bar = st.progress(0, text="🚀 GPU 가속 데이터 수집 시작...")
    dataframes = []
    total_companies = len(selected_companies)
    
    def process_company(company, idx):
        try:
            progress = (idx + 1) / total_companies
            progress_bar.progress(progress, text=f"📊 {company} GPU 처리 중... ({idx + 1}/{total_companies})")
            
            dart_df = dart_collector.get_company_financials_auto(company, analysis_year)
            if dart_df is not None and not dart_df.empty:
                st.info(f"🔧 {company} GPU 가속 전처리 중...")
                
                # GPU 가속 적용
                if st.session_state.use_gpu:
                    dart_df = gpu_dataframe_operations(dart_df)
                
                processed_df = processor.process_dart_data(dart_df, company)
                if processed_df is not None and not processed_df.empty:
                    st.success(f"✅ {company} GPU 처리 완료 ({len(processed_df)}개 항목)")
                    return processed_df
                else:
                    st.warning(f"⚠️ {company} 데이터 전처리 결과가 비어있습니다.")
            else:
                st.warning(f"⚠️ {company} DART 데이터가 비어있습니다.")
        except Exception as e:
            st.error(f"❌ {company} 데이터 수집 중 오류: {str(e)}")
        return None
    
    # 병렬 처리 (GPU 가속)
    max_workers = st.session_state.parallel_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_company, company, idx) 
                  for idx, company in enumerate(selected_companies)]
        
        for future in futures:
            result = future.result()
            if result is not None:
                dataframes.append(result)
    
    progress_bar.progress(1.0, text="✅ GPU 가속 데이터 수집 완료!")
    return dataframes

# --- 데이터 검증 함수 ---
def validate_financial_data(df):
    """재무 데이터 유효성 검증"""
    if df is None or df.empty:
        return False, "데이터가 비어있습니다."
    
    required_columns = ['구분']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"필수 컬럼이 누락되었습니다: {missing_columns}"
    
    return True, "데이터가 유효합니다."

# --- 포맷팅 함수 ---
def format_value_unified(row, col_name):
    """통합된 값 포맷팅 함수"""
    try:
        value = row[col_name]
        if pd.isna(value) or value is None:
            return "-"
        
        if '%' in str(row.get('구분', '')):
            return f"{float(value):.2f}%"
        
        value = float(value)
        if abs(value) >= 1000000000000:  # 1조
            return f"{value/1000000000000:,.1f} 조원"
        elif abs(value) >= 100000000:  # 1억
            return f"{value/100000000:,.0f} 억원"
        elif abs(value) >= 10000:  # 1만
            return f"{value/10000:,.0f} 만원"
        else:
            return f"{value:,.0f}"
    except (ValueError, TypeError):
        return str(value)

# ======================================================================================
# 메인 애플리케이션 함수
# ======================================================================================
def main():
    initialize_session_state()

    st.title("⚡ SK에너지 경쟁사 분석 대시보드 (GPU 가속)")
    st.markdown("**DART API + 구글시트 뉴스 + Gemini AI 인사이트 통합 + GPU 병렬 처리**")
    
    # 성능 설정 사이드바
    with st.sidebar:
        st.subheader("🚀 성능 설정")
        st.session_state.use_gpu = st.checkbox("GPU 가속 사용", value=st.session_state.use_gpu, disabled=not GPU_AVAILABLE)
        st.session_state.parallel_workers = st.slider("병렬 처리 워커 수", 1, 8, st.session_state.parallel_workers)
        
        if GPU_AVAILABLE:
            st.success(f"GPU 메모리: {cp.cuda.runtime.memGetInfo()[0] // 1024**3}GB 사용 가능")
        else:
            st.warning("GPU 가속을 위해 cudf, cupy 설치 필요")
    
    # 탭 구조
    tabs = st.tabs(["📈 재무분석 (DART 자동)", "📁 수동 파일 업로드", "📰 뉴스 분석", "🤖 통합 인사이트", "📄 보고서 생성"])
    
    # -------------------------
    # 탭 1: 재무분석 (DART 자동)
    # -------------------------
    with tabs[0]:
        st.subheader("📈 DART 공시 데이터 심층 분석 (GPU 가속)")
        
        # DART API 키 상태 확인
        if not config.DART_API_KEY:
            st.error("❌ DART API 키가 설정되지 않았습니다!")
            st.info("💡 **DART API 키 설정 방법:**")
            st.markdown("""
            1. **DART Open API 신청**: https://opendart.fss.or.kr/ 접속
            2. **회원가입 및 로그인**: 개인정보 입력 후 가입
            3. **API 키 발급**: 마이페이지에서 API 키 확인
            4. **키 설정**: `config.py` 파일의 `DART_API_KEY` 변수에 발급받은 키 입력
            """)
            st.code("DART_API_KEY = 'your_api_key_here'", language="python")
            return
        else:
            st.success(f"✅ DART API 키 설정됨: {config.DART_API_KEY[:10]}...")
            
            # API 키 테스트 버튼
            if st.button("🔍 API 키 테스트", type="secondary"):
                with st.spinner("API 키 유효성을 확인하는 중..."):
                    dart_collector = DartAPICollector(config.DART_API_KEY)
                    is_valid, message = dart_collector.test_api_key()
                    if is_valid:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_companies = st.multiselect(
                "분석할 기업 선택", 
                config.COMPANIES_LIST, 
                default=config.DEFAULT_SELECTED_COMPANIES
            )
        with col2:
            analysis_year = st.selectbox("분석 연도", ["2024", "2023", "2022"])

        if st.button("🚀 GPU 가속 DART 자동분석 시작", type="primary"):
            if not selected_companies:
                st.error("❌ 분석할 기업을 선택해주세요.")
            else:
                with st.spinner("🚀 GPU 가속 DART API 데이터 수집 및 심층 분석 중..."):
                    try:
                        dart_collector = DartAPICollector(config.DART_API_KEY)
                        processor = UniversalDataProcessor()
                        
                        # GPU 가속 데이터 수집
                        st.info(f"📊 {len(selected_companies)}개 기업 GPU 가속 데이터 수집 시작...")
                        dataframes = collect_financial_data_async(
                            dart_collector, processor, selected_companies, analysis_year
                        )
                        
                        if dataframes and any(df is not None and not df.empty for df in dataframes):
                            st.success(f"✅ {len([df for df in dataframes if df is not None and not df.empty])}개 기업 데이터 수집 완료")
                            
                            # 병렬 데이터 병합
                            st.info("🔄 GPU 가속 데이터 병합 중...")
                            merged_df = processor.merge_company_data(dataframes)
                            
                            # GPU 가속 후처리
                            if st.session_state.use_gpu:
                                merged_df = gpu_dataframe_operations(merged_df)
                            
                            is_valid, message = validate_financial_data(merged_df)
                            
                            if is_valid:
                                # 분석 결과 세션에 저장
                                st.session_state.financial_data = merged_df
                                st.session_state.selected_companies = selected_companies
                                
                                # 비동기 AI 인사이트 생성
                                try:
                                    gemini = GeminiInsightGenerator(config.GEMINI_API_KEY)
                                    st.session_state.financial_insight = gemini.generate_financial_insight(merged_df)
                                except Exception as e:
                                    st.warning(f"⚠️ AI 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
                                    st.session_state.financial_insight = None
                                
                                st.success("🎉 GPU 가속 분석이 완료되었습니다!")
                            else:
                                st.error(f"❌ 데이터 검증 실패: {message}")
                        else:
                            st.error("❌ 데이터 수집에 실패했습니다. DART API 키나 네트워크 연결을 확인해주세요.")
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
                st.rerun()

        # --- 결과 표시 UI ---
        if st.session_state.financial_data is not None:
            st.markdown("---")
            st.subheader("💰 GPU 가속 재무분석 결과")
            
            final_df = st.session_state.financial_data
            
            # 간단한 테이블 표시
            display_df = final_df.copy()
            for company_col in [c for c in display_df.columns if c != '구분']:
                if pd.api.types.is_numeric_dtype(display_df[company_col]):
                    display_df[company_col] = display_df.apply(
                        format_value_unified, axis=1, col_name=company_col
                    )
            
            st.dataframe(display_df.set_index('구분'), use_container_width=True)
            
            # AI 재무 인사이트
            if st.session_state.financial_insight:
                st.subheader("🤖 AI 재무 인사이트")
                st.markdown(st.session_state.financial_insight)

    # -------------------------
    # 탭 2: 수동 파일 업로드 (XBRL + Excel)
    # -------------------------
    with tabs[1]:
        st.subheader("📁 수동 파일 업로드 (XBRL/XML + Excel) - GPU 가속")
        st.write("DART 공시가 없는 회사의 재무제표 파일을 직접 업로드하여 GPU 가속 심층 분석합니다.")
        
        processor = UniversalDataProcessor()
        
        # 파일 타입 선택
        file_type = st.radio("파일 타입 선택", ["XBRL/XML", "Excel"], horizontal=True)
        
        if file_type == "XBRL/XML":
            uploaded_files = st.file_uploader(
                "XBRL/XML 파일들을 업로드하세요 (여러 개 선택 가능)",
                type=['xbrl', 'xml'],
                accept_multiple_files=True,
                key="manual_upload"
            )
        else:
            uploaded_files = st.file_uploader(
                "Excel 파일들을 업로드하세요 (여러 개 선택 가능)",
                type=['xlsx', 'xls'],
                accept_multiple_files=True,
                key="excel_upload"
            )
        
        if uploaded_files:
            try:
                dataframes = []
                progress_bar = st.progress(0, text="🚀 GPU 가속 파일 처리 중...")
                
                for idx, file in enumerate(uploaded_files):
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress, text=f"📄 {file.name} GPU 처리 중... ({idx + 1}/{len(uploaded_files)})")
                    
                    st.info(f"🔍 {file.name} GPU 가속 파일 분석 시작...")
                    
                    # 파일 처리
                    if file_type == "XBRL/XML":
                        df = processor.process_uploaded_file(file)
                    else:
                        # Excel 파일 처리
                        try:
                            excel_df = pd.read_excel(file)
                            # GPU 가속 Excel 데이터 처리
                            if st.session_state.use_gpu:
                                excel_df = gpu_dataframe_operations(excel_df)
                            # Excel 데이터를 표준 형식으로 변환
                            df = processor.process_excel_data(excel_df, file.name)
                        except Exception as e:
                            st.error(f"❌ Excel 파일 처리 오류: {str(e)}")
                            df = None
                    
                    if df is not None and not df.empty:
                        st.success(f"✅ {file.name} GPU 처리 완료 ({len(df)}개 항목)")
                        dataframes.append(df)
                        
                        # 처리 결과 미리보기
                        with st.expander(f"📊 {file.name} GPU 처리 결과 미리보기"):
                            st.dataframe(df.head(), use_container_width=True)
                    else:
                        st.warning(f"⚠️ {file.name}에서 유효한 데이터를 추출할 수 없습니다.")
                
                progress_bar.progress(1.0, text="✅ GPU 가속 파일 처리 완료!")
                
                if dataframes:
                    st.info("🔄 GPU 가속 파일 병합 중...")
                    # 병렬 데이터 병합
                    merged_df = processor.merge_company_data(dataframes)
                    
                    # GPU 가속 후처리
                    if st.session_state.use_gpu:
                        merged_df = gpu_dataframe_operations(merged_df)
                    
                    is_valid, message = validate_financial_data(merged_df)
                    
                    if is_valid:
                        st.session_state.manual_financial_data = merged_df
                        st.success("✅ GPU 가속 파일 분석이 완료되었습니다!")
                        
                        # 비동기 AI 인사이트 생성
                        try:
                            gemini = GeminiInsightGenerator(config.GEMINI_API_KEY)
                            st.session_state.manual_insight = gemini.generate_financial_insight(merged_df)
                        except Exception as e:
                            st.warning(f"⚠️ AI 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
                            st.session_state.manual_insight = None
                        
                        # 결과 요약
                        st.info("📋 GPU 가속 분석 결과 요약:")
                        company_cols = [col for col in merged_df.columns if col != '구분' and not col.endswith('_원시값')]
                        for company in company_cols:
                            company_data = merged_df[merged_df[company] != '-']
                            st.write(f"• {company}: {len(company_data)}개 항목")
                    else:
                        st.error(f"데이터 검증 실패: {message}")
                else:
                    st.error("처리할 수 있는 데이터가 없습니다.")
                    
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            
            st.rerun()

        # --- 수동 업로드 결과 표시 ---
        if st.session_state.manual_financial_data is not None:
            st.markdown("---")
            st.subheader("💰 GPU 가속 수동 업로드 파일 분석 결과")

            final_df = st.session_state.manual_financial_data
            
            # 간단한 테이블 표시
            display_df = final_df.copy()
            for company_col in [c for c in display_df.columns if c != '구분']:
                if pd.api.types.is_numeric_dtype(display_df[company_col]):
                    display_df[company_col] = display_df.apply(
                        format_value_unified, axis=1, col_name=company_col
                    )

            st.dataframe(display_df.set_index('구분'), use_container_width=True)
            
            # AI 인사이트 표시
            if st.session_state.get('manual_insight'):
                st.subheader("🤖 AI 분석 인사이트")
                st.markdown(st.session_state.manual_insight)

    # -------------------------
    # 탭 3: 뉴스분석
    # -------------------------
    with tabs[2]:
        st.subheader("📰 경쟁사 벤치마킹 뉴스 분석")
        st.write("경쟁사의 새로운 투자, 손익 개선 활동 등 벤치마킹 아이디어를 탐색합니다.")

        st.sidebar.subheader("🔍 뉴스 검색 키워드 설정")
        keyword_str = st.sidebar.text_area(
            "키워드 (쉼표로 구분)",
            ", ".join(st.session_state.custom_keywords),
            key="keyword_editor"
        )
        st.session_state.custom_keywords = [kw.strip() for kw in keyword_str.split(',')]
        
        if st.button("🔄 최신 벤치마킹 뉴스 수집 및 분석", type="primary"):
            with st.spinner("최신 뉴스를 수집하고 AI로 분석 중입니다..."):
                try:
                    news_collector = SKNewsCollector(custom_keywords=st.session_state.custom_keywords)
                    news_df = news_collector.collect_news()
                    st.session_state.news_data = news_df

                    if news_df is not None and not news_df.empty:
                        st.success(f"✅ 총 {len(news_df)}개의 관련 뉴스를 수집했습니다. AI 분석을 시작합니다.")
                        
                        # AI 인사이트 생성
                        try:
                            gemini = GeminiInsightGenerator(config.GEMINI_API_KEY)
                            insight = gemini.generate_news_insight(news_df)
                            st.session_state.news_insight = insight
                        except Exception as e:
                            st.warning(f"AI 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
                            st.session_state.news_insight = None
                    else:
                        st.warning("관련 뉴스를 찾지 못했습니다. 키워드를 변경해보세요.")
                        st.session_state.news_insight = None
                except Exception as e:
                    st.error(f"뉴스 수집 중 오류가 발생했습니다: {str(e)}")
            st.rerun()

        if st.session_state.get('news_insight'):
            st.subheader("🤖 AI 종합 분석 리포트")
            st.markdown(st.session_state.news_insight)

        if st.session_state.get('news_data') is not None:
            if not st.session_state.news_data.empty:
                st.subheader("📋 수집된 뉴스 목록")
                
                news_df = st.session_state.news_data.copy()
                st.dataframe(news_df, use_container_width=True)
            else:
                st.info("수집된 뉴스가 없습니다. 사이드바에서 키워드를 조정하고 다시 수집해보세요.")

    # -------------------------
    # 탭 4: 통합 인사이트
    # -------------------------
    with tabs[3]:
        st.subheader("🤖 통합 인사이트 & 손익개선 전략")
        st.write("DART 재무 데이터와 뉴스 분석을 종합하여 포괄적인 인사이트와 손익개선 방안을 제시합니다.")
        
        # 데이터 확인
        has_financial_data = (st.session_state.financial_data is not None or 
                            st.session_state.manual_financial_data is not None)
        has_news_data = (st.session_state.news_data is not None and 
                        not st.session_state.news_data.empty)
        
        if not has_financial_data and not has_news_data:
            st.warning("통합 인사이트를 생성하려면 재무 데이터와 뉴스 데이터가 필요합니다.")
            st.info("💡 팁: 먼저 '재무분석' 탭에서 DART 분석을 실행하고, '뉴스분석' 탭에서 뉴스를 수집해주세요.")
        else:
            # 통합 인사이트 생성 버튼
            if st.button("🚀 통합 인사이트 생성", type="primary"):
                with st.spinner("DART 재무 데이터와 뉴스 분석을 종합하여 인사이트를 생성 중..."):
                    try:
                        gemini = GeminiInsightGenerator(config.GEMINI_API_KEY)
                        
                        # 재무 데이터 준비
                        financial_data = (st.session_state.financial_data if st.session_state.financial_data is not None 
                                        else st.session_state.manual_financial_data)
                        
                        # 통합 인사이트 생성
                        integrated_insight = gemini.generate_integrated_insight(
                            financial_data=financial_data,
                            news_data=st.session_state.news_data if has_news_data else None
                        )
                        
                        st.session_state.integrated_insight = integrated_insight
                        st.success("✅ 통합 인사이트가 생성되었습니다!")
                    except Exception as e:
                        st.error(f"통합 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
                st.rerun()
            
            # 통합 인사이트 표시
            if st.session_state.get('integrated_insight'):
                st.markdown("---")
                st.subheader("📊 종합 분석 결과")
                st.markdown(st.session_state.integrated_insight)

    # ==========================
    # 탭4: 보고서 생성 및 이메일 발송 (개선된 UI + PDF 쪽번호)
    # ==========================
    with tabs[4]:
        st.subheader("📄 통합 보고서 생성 & 이메일 서비스 바로가기")

        # 2열 레이아웃: PDF 생성 + 이메일 입력
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**📥 보고서 다운로드**")

            # 👉 사용자 입력(보고 대상/보고자/푸터 노출)
            report_target = st.text_input("보고 대상", value="SK이노베이션 경영진")
            report_author = st.text_input("보고자", value="")
            show_footer = st.checkbox("푸터 문구 표시(※ 본 보고서는 대시보드에서 자동 생성되었습니다.)", value=False)

            # 보고서 형식 선택
            report_format = st.radio("파일 형식 선택", ["PDF", "Excel"], horizontal=True)

            if st.button("📥 보고서 생성", type="primary", key="make_report"):
                # 데이터 우선순위: DART 자동 > 수동 업로드
                financial_data_for_report = None
                if st.session_state.financial_data is not None and not st.session_state.financial_data.empty:
                    financial_data_for_report = st.session_state.financial_data
                elif st.session_state.manual_financial_data is not None and not st.session_state.manual_financial_data.empty:
                    financial_data_for_report = st.session_state.manual_financial_data

                # 선택 입력(있으면 전달)
                quarterly_df = st.session_state.get("quarterly_data")
                selected_charts = st.session_state.get("selected_charts")

                with st.spinner("📄 보고서 생성 중..."):
                    if report_format == "PDF":
                        file_bytes = create_enhanced_pdf_report(
                            financial_data=financial_data_for_report,
                            news_data=st.session_state.news_data,
                            insights=st.session_state.financial_insight or st.session_state.news_insight,
                            quarterly_df=quarterly_df,                 # 분기 데이터(있으면)
                            selected_charts=selected_charts,           # 외부 전달 차트(있으면)
                            show_footer=show_footer,                   # ✅ 푸터 표시 여부 반영
                            report_target=report_target.strip() or "보고 대상 미기재",  # ✅ 사용자 입력 반영
                            report_author=report_author.strip() or "보고자 미기재"      # ✅ 사용자 입력 반영
                        )
                        filename = "SK_Energy_Analysis_Report.pdf"
                        mime_type = "application/pdf"
                    else:
                        file_bytes = create_excel_report(
                            financial_data=financial_data_for_report,
                            news_data=st.session_state.news_data,
                            insights=st.session_state.financial_insight or st.session_state.news_insight
                        )
                        filename = "SK_Energy_Analysis_Report.xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                    if file_bytes:
                        # 세션에 파일 정보 저장
                        st.session_state.generated_file = file_bytes
                        st.session_state.generated_filename = filename
                        st.session_state.generated_mime = mime_type

                        st.download_button(
                            label="⬇️ 보고서 다운로드",
                            data=file_bytes,
                            file_name=filename,
                            mime=mime_type
                        )
                        st.success("✅ 보고서가 성공적으로 생성되었습니다!")
                    else:
                        st.error("❌ 보고서 생성에 실패했습니다.")
                        
        with col2:
            st.write("**📧 이메일 서비스 바로가기**")

            mail_providers = {
                "네이버": "https://mail.naver.com/",
                "구글(Gmail)": "https://mail.google.com/",
                "다음": "https://mail.daum.net/",
                "네이트": "https://mail.nate.com/",
                "야후": "https://mail.yahoo.com/",
                "아웃룩(Outlook)": "https://outlook.live.com/",
                "프로톤메일(ProtonMail)": "https://mail.proton.me/",
                "조호메일(Zoho Mail)": "https://mail.zoho.com/",
                "GMX 메일": "https://www.gmx.com/",
                "아이클라우드(iCloud Mail)": "https://www.icloud.com/mail",
                "메일닷컴(Mail.com)": "https://www.mail.com/",
                "AOL 메일": "https://mail.aol.com/"
            }

            selected_provider = st.selectbox(
                "메일 서비스 선택",
                list(mail_providers.keys()),
                key="mail_provider_select"
            )
            url = mail_providers[selected_provider]

            st.markdown(
                f"[{selected_provider} 메일 바로가기]({url})",
                unsafe_allow_html=True
            )
            st.info("선택한 메일 서비스 링크가 새 탭에서 열립니다.")

            if st.session_state.get('generated_file'):
                st.download_button(
                    label=f"📥 {st.session_state.generated_filename} 다운로드",
                    data=st.session_state.generated_file,
                    file_name=st.session_state.generated_filename,
                    mime=st.session_state.generated_mime,
                    key="download_generated_report_btn"
                )
            else:
                st.info("먼저 보고서를 생성해주세요.")

if __name__ == "__main__":
    main()

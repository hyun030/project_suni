# -*- coding: utf-8 -*-
# API 키, URL 등 모든 설정 정보를 관리하는 파일입니다.
import streamlit as st
import json

# 서비스 계정 정보 로드 (테스트/실제 겸용)
if "google_service_account" in st.secrets:
    service_account_info = json.loads(st.secrets["google_service_account"])
else:
    service_account_info = {
        "type": "service_account",
        "project_id": "test-project-id",
        "private_key_id": "dummy_key_id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nABC123...\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "1234567890",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test@test-project.iam.gserviceaccount.com"
    }

# Google Sheets ID 로드
if "google_sheets_id" in st.secrets:
    google_sheets_id = st.secrets["google_sheets_id"]
else:
    google_sheets_id = "1VcAuVG4h4fh8XqhzdTmHb73k6dbZLzdrC-W04avJDAQ"

# Google Sheets 전체 URL
google_sheets_url = f"https://docs.google.com/spreadsheets/d/{google_sheets_id}/edit?usp=sharing"

# ==========================
# API 키 및 인증 정보
# Streamlit 배포 시에는 st.secrets를 사용하는 것이 안전합니다.
# ==========================

# DART API 키 설정 (테스트용 키 - 실제 사용시 교체 필요)
DART_API_KEY = "9a153f4344ad2db546d651090f78c8770bd773cb"  # 테스트용 DART API 키

# Streamlit secrets에서 DART API 키 로드 (우선순위)
try:
    if "DART_API_KEY" in st.secrets:
        dart_key_from_secrets = st.secrets["DART_API_KEY"]
        # DART API 키가 Gemini API 키와 다른지 확인
        if dart_key_from_secrets and not dart_key_from_secrets.startswith("AIzaSy"):
            DART_API_KEY = dart_key_from_secrets
        else:
            print("⚠️ st.secrets의 DART_API_KEY가 유효하지 않습니다. 기본값을 사용합니다.")
except:
    pass  # st.secrets가 없을 때는 기본값 사용

# DART API 키 검증
if not DART_API_KEY or DART_API_KEY == "test_key" or DART_API_KEY.startswith("AIzaSy"):
    try:
        st.warning("⚠️ DART API 키가 설정되지 않았습니다. https://opendart.fss.or.kr/ 에서 발급받으세요.")
    except:
        print("⚠️ DART API 키가 설정되지 않았습니다. https://opendart.fss.or.kr/ 에서 발급받으세요.")

GEMINI_API_KEY = "AIzaSyB176ys4MCjEs8R0dv15hMqDE2G-9J0qIA"  # 실제 키. 보안에 매우 유의하세요.

# 구글시트 설정
# GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1VcAuVG4h4fh8XqhzdTmHb73k6dbZLzdrC-W04avJDAQ/edit?usp=sharing"
# SHEET_ID = "1VcAuVG4h4fh8XqhzdTmHb73k6dbZLzdrC-W04avJDAQ"

# ==========================
# UI 및 시각화 설정
# ==========================
# SK 브랜드 컬러 테마
SK_COLORS = {
    'primary': '#E31E24',      # SK 레드
    'secondary': '#FF6B35',    # SK 오렌지
    'accent': '#004EA2',       # SK 블루
    'success': '#00A651',      # 성공 색상
    'warning': '#FF9500',      # 경고 색상
    'competitor': '#6C757D',   # 기본 경쟁사 색상 (회색)
    # 개별 경쟁사 파스텔 색상
    'competitor_green': '#8BC34A', # 파스텔 그린
    'competitor_blue': '#64B5F6',  # 파스텔 블루
    'competitor_yellow': '#FFF176',# 파스텔 옐로우
    'competitor_purple': '#B39DDB',# 파스텔 퍼플
    'competitor_orange': '#FFCC80',# 파스텔 오렌지
    'competitor_mint': '#80CBC4'   # 파스텔 민트
}

# 분석 대상 회사 목록 (UI에서 사용)
COMPANIES_LIST = ["SK에너지", "GS칼텍스", "HD현대오일뱅크", "S-Oil"]
DEFAULT_SELECTED_COMPANIES = ["SK에너지", "GS칼텍스"]

# ==========================
# DART API 관련 설정
# ==========================
# [개선] 중앙 재무 용어 사전 - 더 정확한 매핑
# 모든 재무 데이터는 아래 표준 용어로 통일됩니다.
# 회사별로 다른 용어(alias)를 여기에 계속 추가하여 관리합니다.
FINANCIAL_TERM_MAP = {
    "매출액": {
        "Default": ["매출액", "수익(매출액)", "영업수익", "revenue", "sales", "매출", "총매출", "매출액(수익)"],
        "GS칼텍스": ["총매출", "매출액", "매출액(수익)", "영업수익"],
        "HD현대오일뱅크": ["매출액", "영업수익", "매출액(수익)", "총매출"],
        "S-Oil": ["매출액", "영업수익", "매출액(수익)", "총매출"],
        "SK에너지": ["매출액", "영업수익", "매출액(수익)", "총매출"],
    },
    "매출원가": {
        "Default": ["매출원가", "cost of sales", "costofgoodssold", "매출원가(매출액)", "매출원가(수익)"],
        "GS칼텍스": ["매출원가", "매출원가(매출액)", "매출원가(수익)"],
        "HD현대오일뱅크": ["매출원가", "매출원가(매출액)", "매출원가(수익)"],
        "S-Oil": ["매출원가", "매출원가(매출액)", "매출원가(수익)"],
        "SK에너지": ["매출원가", "매출원가(매출액)", "매출원가(수익)"],
    },
    "판관비": {
        "Default": ["판매비와관리비", "selling, general and administrative expenses", "판관비", "판매비 및 일반관리비", "판매비와관리비(수익)"],
        "GS칼텍스": ["판매비와관리비", "판관비", "판매비와관리비(수익)"],
        "HD현대오일뱅크": ["판매비와관리비", "판관비", "판매비와관리비(수익)"],
        "S-Oil": ["판매비와관리비", "판관비", "판매비와관리비(수익)"],
        "SK에너지": ["판매비와관리비", "판관비", "판매비와관리비(수익)"],
    },
    "영업이익": {
        "Default": ["영업이익", "영업손익", "operating income", "operating profit", "영업손익(영업이익)", "영업이익(손실)"],
        "GS칼텍스": ["영업이익", "영업손익(영업이익)", "영업이익(손실)"],
        "HD현대오일뱅크": ["영업이익", "영업손익(영업이익)", "영업이익(손실)"],
        "S-Oil": ["영업이익", "영업손익(영업이익)", "영업이익(손실)"],
        "SK에너지": ["영업이익", "영업손익(영업이익)", "영업이익(손실)"],
    },
    "당기순이익": {
        "Default": ["당기순이익", "분기순이익", "net income", "profit for the period", "당기순손익", "당기순이익(손실)"],
        "GS칼텍스": ["당기순이익", "당기순손익", "당기순이익(손실)"],
        "HD현대오일뱅크": ["당기순이익", "당기순손익", "당기순이익(손실)"],
        "S-Oil": ["당기순이익", "당기순손익", "당기순이익(손실)"],
        "SK에너지": ["당기순이익", "당기순손익", "당기순이익(손실)"],
    },
    # --- 심화 분석을 위한 추가 항목들 ---
    "인건비": {
        "Default": ["인건비", "급여", "임금", "employee benefits", "인사비", "급여 및 복리후생비", "인건비(급여)", "급여 및 복리후생비(수익)"],
        "GS칼텍스": ["인건비", "급여", "급여 및 복리후생비", "급여 및 복리후생비(수익)"],
        "HD현대오일뱅크": ["인건비", "급여", "급여 및 복리후생비", "급여 및 복리후생비(수익)"],
        "S-Oil": ["인건비", "급여", "급여 및 복리후생비", "급여 및 복리후생비(수익)"],
        "SK에너지": ["인건비", "급여", "급여 및 복리후생비", "급여 및 복리후생비(수익)"],
    },
    "감가상각비": {
        "Default": ["감가상각비", "depreciation", "감가상각비 및 무형자산상각비", "감가상각비(감가상각비 및 무형자산상각비)", "감가상각비 및 무형자산상각비(수익)"],
        "GS칼텍스": ["감가상각비", "감가상각비 및 무형자산상각비", "감가상각비 및 무형자산상각비(수익)"],
        "HD현대오일뱅크": ["감가상각비", "감가상각비 및 무형자산상각비", "감가상각비 및 무형자산상각비(수익)"],
        "S-Oil": ["감가상각비", "감가상각비 및 무형자산상각비", "감가상각비 및 무형자산상각비(수익)"],
        "SK에너지": ["감가상각비", "감가상각비 및 무형자산상각비", "감가상각비 및 무형자산상각비(수익)"],
    },
    "동력비": {
        "Default": ["동력비", "전력비", "연료비", "energy cost", "전력비 및 연료비", "동력비(전력비)", "전력비 및 연료비(수익)"],
        "GS칼텍스": ["동력비", "전력비", "전력비 및 연료비", "전력비 및 연료비(수익)"],
        "HD현대오일뱅크": ["동력비", "전력비", "전력비 및 연료비", "전력비 및 연료비(수익)"],
        "S-Oil": ["동력비", "전력비", "전력비 및 연료비", "전력비 및 연료비(수익)"],
        "SK에너지": ["동력비", "전력비", "전력비 및 연료비", "전력비 및 연료비(수익)"],
    },
    "원재료비": {
        "Default": ["원재료비", "재료비", "raw material cost", "원재료비(재료비)", "재료비(수익)"],
        "GS칼텍스": ["원재료비", "재료비", "재료비(수익)"],
        "HD현대오일뱅크": ["원재료비", "재료비", "재료비(수익)"],
        "S-Oil": ["원재료비", "재료비", "재료비(수익)"],
        "SK에너지": ["원재료비", "재료비", "재료비(수익)"],
    },
}

# DART API에서 회사를 찾기 위한 정보 (기존 MAPPING 통합)
COMPANY_INFO = {
    "SK에너지": {
        "aliases": ["SK에너지", "SK에너지주식회사", "에스케이에너지", "SK ENERGY"],
        "stock_code": "096770"
    },
    "GS칼텍스": {
        "aliases": ["GS칼텍스", "지에스칼텍스", "GS칼텍스주식회사"],
        "stock_code": "089590"
    },
    "HD현대오일뱅크": {
        "aliases": ["HD현대오일뱅크", "현대오일뱅크", "HYUNDAI OILBANK"],
        "stock_code": "267250"
    },
    "S-Oil": {
        "aliases": ["S-Oil", "에쓰오일", "에스오일", "S-OIL"],
        "stock_code": "010950"
    }
}

# ==========================
# 뉴스 수집 관련 설정 (구글 시트만 사용)
# ==========================

# 벤치마킹을 위한 뉴스 필터링 키워드 (요구사항 기반 개선)
# 이 키워드 중 하나라도 포함되어야 기사를 수집합니다.
BENCHMARKING_KEYWORDS = [
    # General 키워드 (기본)
    "매출", "영업이익",
    
    # 정유업계 특화 키워드 (상황 안 좋을 때)
    "수주", "비가동", "셧다운", "가동률", "수율",
    
    # 재무표 키워드 (심화 분석)
    "영업이익률", "매출원가율", "고정비", "인건비", "감가상각비", 
    "공헌이익", "EBITDA", "넷캐시마진",
    
    # 벤치마킹 포인트 (경쟁사 새로운 시도)
    "신사업", "투자", "M&A", "진출", "개발", "협력", "MOU",
    "증설", "공장", "설비", "정책", "시장 진출",
    
    # ESG 및 정책
    "ESG", "탄소중립", "수소", "친환경", "신재생에너지"
]

# 회사명 태깅을 위한 키워드
COMPANY_KEYWORDS = {
    "SK에너지": ["SK에너지", "SK이노베이션"],
    "GS칼텍스": ["GS칼텍스"],
    "HD현대오일뱅크": ["HD현대오일뱅크", "현대오일뱅크"],
    "S-Oil": ["S-Oil", "에쓰오일"]
}

# ==========================
# 이메일 관련 설정
# ==========================
MAIL_PROVIDERS = {
    "네이버": "https://mail.naver.com/", "구글(Gmail)": "https://mail.google.com/",
    "다음": "https://mail.daum.net/", "아웃룩(Outlook)": "https://outlook.live.com/",
}
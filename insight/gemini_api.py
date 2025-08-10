# -*- coding: utf-8 -*-
# Gemini AI와 연동하여 텍스트 인사이트를 생성하는 모든 코드를 관리합니다.

import streamlit as st
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiInsightGenerator:
    def __init__(self, api_key):
        self.model = None
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                st.error(f"Gemini 모델 초기화 실패: {e}")
        elif not GEMINI_AVAILABLE:
            st.warning("Gemini API를 사용할 수 없습니다. API 키를 확인해주세요.")

    def generate_financial_insight(self, financial_data):
        """재무데이터 → 경쟁사 분석 인사이트"""
        if not self.model:
            return "Gemini API를 사용할 수 없습니다. API 키를 확인해주세요."
        
        try:
            # 재무데이터를 텍스트로 변환
            data_str = financial_data.to_string() if hasattr(financial_data, 'to_string') else str(financial_data)
            
            prompt = f"""
다음은 SK에너지 중심의 재무데이터입니다:

{data_str}

당신은 SK에너지 내부 전략기획팀 소속으로,
경쟁사 대비 SK에너지의 **재무 및 사업 경쟁력**을 분석하여
향후 **사업 전략 수립 및 개선 방향**을 도출하는 것이 목표입니다.

다음 항목에 맞춰 **전략적 인사이트**를 도출해주세요:

## 1. 📊 SK에너지 현재 재무 상황 분석
- 최근 수익성 변화와 원인 진단
- 경쟁사(GS칼텍스 등) 대비 수익구조의 강점/약점
- 영업이익률, 순이익률, 원가율, 판관비율 등 주요 지표 비교

## 2. 🔍 경쟁사 대비 사업 경쟁력 분석
- 경쟁사 대비 원가 효율성, 수익성, 비용 구조 차이
- SK에너지가 개선하거나 강화할 수 있는 포인트 도출

## 3. 🧩 전략적 시사점 및 내부 개선 방향
- 단기적으로 재무 개선을 위해 우선 검토해야 할 영역
- 중장기적으로 경쟁력을 높이기 위한 조직 차원의 전략 제언

## 4. 📌 리스크 요인 및 감시 항목
- 현재 가장 큰 재무적/사업적 리스크
- 외부환경(유가, 정책 등)에 따른 리스크 민감도
- 내부적으로 반드시 모니터링해야 할 주요 지표

## 5. DART 공시 데이터를 바탕으로 SK에너지와 경쟁사 간 갭 분석 및 추세 기반 전략 도출
[입력 데이터]

대상: SK에너지, HD현대오일뱅크, GS칼텍스, S-Oil
기간: 최근 12개 분기
항목: 매출액, 매출원가, 영업이익, 고정비(인건비/감가상각비), 변동비(동력비/원재료비), 생산량

[분석 지표]

영업이익률, 매출원가율
고정비율(인건비율, 감가상각비율)
변동비율(동력비율, 원재료비율)
공헌이익률(매출-변동비)/매출
단위당 영업이익(원/배럴, 원/톤)

[갭 계산]

현재 갭: SK지표 - 경쟁사평균
추세: 최근 4분기 갭 변화율
위험도: 우위→축소(高), 열위→확대(中), 열위→축소(기회)

[결과 양식]
📊 갭 현황
🟢 우위지표: [지표] SK[값] vs 경쟁사[값] (갭+[차이], 추세[%])
🔴 열위지표: [지표] SK[값] vs 경쟁사[값] (갭-[차이], 추세[%])

⚠️ 위험신호

긴급: [우위→급속축소 지표]
취약: [열위→지속확대 지표]

📈 전략방안
[지표명]

상황: "SK에너지는 ○○하므로"
조치: "○○가 필요하다"
목표: [시점]까지 갭 [현재]→[목표]

🎯 우선순위
1순위: [긴급지표] - [목표] - [방안]
2순위: [기회지표] - [목표] - [방안]

## 6. 🚀 향후 6개월 내 실질적 액션 플랜 제안
- 실행 가능한 내부 조치 3~5가지 제안
- KPI 기준 재설정 또는 목표 재정의 필요 여부

분석은 전문 컨설턴트 수준으로 해주시되, 실무자가 바로 보고 실행방안을 만들 수 있을 정도로 구체적이고 현실적인 조언을 포함해주세요.
"""

            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"AI 인사이트 생성 중 오류가 발생했습니다: {e}"

    def generate_news_insight(self, news_df: pd.DataFrame):
        """여러 경쟁사 뉴스를 종합하여 하나의 통합된 벤치마킹 보고서를 생성합니다."""
        if self.model is None: return "Gemini API를 사용할 수 없습니다."
        if news_df is None or news_df.empty: return "분석할 뉴스 데이터가 없습니다."

        competitor_news = news_df[~news_df['회사'].astype(str).str.contains("SK", na=False)].head(10)
        
        if competitor_news.empty:
            return "분석할 경쟁사의 최신 뉴스가 없습니다."

        news_data_str = ""
        for i, row in enumerate(competitor_news.iterrows(), 1):
            title = row[1].get('제목', '제목 없음')
            company = row[1].get('회사', '회사 불명')
            news_data_str += f"기사 {i} ({company}): {title}\n"

        prompt = f"""
        당신은 SK에너지의 수석 전략 분석가입니다. 아래 제공된 경쟁사들의 최신 뉴스 목록을 종합적으로 분석하여,
        우리 회사가 즉시 검토해볼 만한 핵심적인 벤치마킹 아이디어를 도출하는 것이 당신의 임무입니다.
        개별 기사를 단순히 요약하지 말고, 여러 기사에서 나타나는 공통적인 트렌드나 중요한 단일 활동을 꿰뚫어 보세요.

        [분석 대상 최신 경쟁사 뉴스 목록]
        {news_data_str}

        [보고서 작성 지침]
        1.  **통합 분석 요약 (Executive Summary):**
            - 뉴스들을 관통하는 가장 중요한 시장 트렌드는 무엇입니까? (예: 친환경 투자 가속화, 원가 절감 노력 등)
            - 우리 회사가 가장 시급하게 주목해야 할 경쟁사의 활동은 무엇이며, 그 이유는 무엇입니까?

        2.  **핵심 벤치마킹 아이디어 TOP 2:**
            - 가장 가치가 높다고 판단되는 아이디어 두 가지를 선정하여 아래 형식으로 구체적으로 제시해주세요.

            ---
            ### 💡 아이디어 1: [벤치마킹 활동 요약, 예: 바이오 항공유 신사업 진출]
            - **관련 기사:** [분석 대상 뉴스 목록에서 가장 관련 있는 기사 번호를 1~2개 언급]
            - **벤치마킹 포인트:** [경쟁사가 무엇을, 왜, 어떻게 하고 있는지 구체적으로 서술]
            - **우리 회사 적용 방안:** [이 아이디어를 우리 회사 상황에 맞게 어떻게 적용할 수 있을지 구체적인 실행 계획 제안]
            - **예상 손익 영향:** [매출, 원가, 비용 관점에서 어떤 긍정적/부정적 영향이 있을지 예측. (예: 초기 투자비(고정비) 증가, 장기적 신규 매출 발생)]
            - **적용 가능성:** [높음/중간/낮음]

            ### 💡 아이디어 2: [두 번째 벤치마킹 활동 요약]
            - **관련 기사:**
            - **벤치마킹 포인트:**
            - **우리 회사 적용 방안:**
            - **예상 손익 영향:**
            - **적용 가능성:**
            ---

        3.  **기타 주목할 만한 활동:**
            - 위 TOP 2 외에 추가로 주목할 만한 경쟁사의 활동이 있다면 간략하게 언급해주세요.

        최종 결과물은 바로 경영진에게 보고할 수 있는 수준의 명확하고 논리적인 보고서 형식이어야 합니다.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"AI 뉴스 인사이트 생성 중 오류가 발생했습니다: {e}")
            return "AI 인사이트 생성에 실패했습니다."

    def generate_integrated_insight(self, financial_data=None, news_data=None):
        """재무 데이터와 뉴스 데이터를 종합하여 통합 인사이트를 생성합니다."""
        if not self.model:
            return "Gemini API를 사용할 수 없습니다. API 키를 확인해주세요."
        
        try:
            # 데이터 준비
            financial_str = ""
            if financial_data is not None:
                financial_str = financial_data.to_string() if hasattr(financial_data, 'to_string') else str(financial_data)
            
            news_str = ""
            if news_data is not None and not news_data.empty:
                news_str = news_data.head(10).to_string()  # 상위 10개 뉴스만 사용
            
            prompt = f"""
당신은 SK에너지의 전략기획팀 소속으로, 재무 데이터와 뉴스 분석을 종합하여 
**포괄적인 인사이트와 손익개선 전략**을 도출하는 것이 목표입니다.

## 📊 재무 데이터
{financial_str}

## 📰 뉴스 데이터 (최근 관련 뉴스)
{news_str}

위 데이터를 종합하여 다음 항목에 맞춰 **전략적 인사이트**를 도출해주세요:

## 🎯 종합 분석 결과

### 1. 📈 현재 상황 진단
- SK에너지의 전반적인 재무 상황과 시장 포지션
- 경쟁사 대비 주요 강점과 약점
- 최근 뉴스에서 파악되는 시장 동향과 SK에너지에 미치는 영향

### 2. 🔍 핵심 문제점 및 기회요인
- 재무 데이터에서 발견되는 주요 문제점
- 뉴스에서 파악되는 시장 기회와 위험요인
- 경쟁사의 새로운 시도나 전략 변화

### 3. 💡 손익개선 전략 제안

#### 단기 개선 방안 (3-6개월)
- 즉시 실행 가능한 비용 절감 방안
- 수익성 개선을 위한 단기 조치
- 뉴스에서 파악된 시장 기회 활용 방안

#### 중장기 전략 방향 (6개월-2년)
- 구조적 개선을 위한 중장기 전략
- 경쟁사 벤치마킹을 통한 사업 모델 개선
- 시장 변화에 대응한 사업 포트폴리오 조정

### 4. ⚠️ 리스크 관리 및 모니터링
- 주요 리스크 요인과 대응 방안
- 지속적으로 모니터링해야 할 지표
- 외부 환경 변화에 대한 대응 전략

### 5. 📋 실행 로드맵
- 우선순위별 실행 계획
- 각 전략의 예상 효과와 소요 기간
- 성과 측정을 위한 KPI 제안

## 💰 핵심 손익개선 포인트
- **매출 증대**: [구체적 방안]
- **원가 절감**: [구체적 방안]  
- **효율성 향상**: [구체적 방안]
- **리스크 최소화**: [구체적 방안]

분석 결과는 **실행 가능한 구체적 전략**으로 제시하고, 
재무 데이터와 뉴스 정보를 모두 활용하여 **종합적이고 실용적인 인사이트**를 제공해주세요.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"통합 인사이트 생성 중 오류가 발생했습니다: {e}"
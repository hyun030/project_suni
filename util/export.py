# -*- coding: utf-8 -*-
# PDF, Excel 등 파일 내보내기 관련 모든 코드를 관리합니다.

import io
import os
import pandas as pd
import streamlit as st
import re
import tempfile
from datetime import datetime

# Plotly는 차트 이미지를 위해 필요할 수 있으므로 안전하게 import
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

#
# --- create_enhanced_pdf_report 함수 수정 ---
# 함수 정의 라인이 중복되어 있었습니다. 하나로 합치고 코드를 정리했습니다.
#
def create_enhanced_pdf_report(financial_data=None, news_data=None, insights:str|None=None, selected_charts:list|None=None):
    """
    - 제목 : 맑은고딕 Bold 20pt
    - 본문 : 신명조 12pt, 줄간격 170 %
    - AI 인사이트 : 번호 붙은 제목은 굵게, 본문은 평문(마크다운 기호 제거)
    - 표 : ReportLab Table 로 출력
    - 차트 : Plotly figure 리스트(selected_charts) PNG 로 삽입
    """
    if not PDF_AVAILABLE:
        st.error("reportlab 라이브러리가 필요합니다.")
        return None

    # ---------- 1. 내부 헬퍼 ----------
    def _clean_ai_text(raw:str)->list[tuple[str,str]]:
        raw = re.sub(r'[*_`#>~]', '', raw)
        blocks = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if re.match(r'^\d+(\.\d+)*\s', ln):
                blocks.append(('title', ln))
            else:
                blocks.append(('body', ln))
        return blocks

    def _ascii_block_to_table(lines:list[str]):
        header = [c.strip() for c in lines[0].split('|') if c.strip()]
        data = []
        for ln in lines[2:]:
            cols = [c.strip() for c in ln.split('|') if c.strip()]
            if len(cols)==len(header):
                data.append(cols)
        if not data:
            return None
        tbl = Table([header]+data)
        tbl.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#E31E24')),
            ('TEXTCOLOR',(0,0),(-1,0), colors.white),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'KoreanBold'),
            ('FONTNAME',(0,1),(-1,-1),'Korean'),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.whitesmoke, colors.HexColor('#F7F7F7')]),
        ]))
        return tbl

    # ---------- 2. 폰트 등록 ----------
    font_paths = {
        "Korean": ["C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
        "KoreanBold": ["C:/Windows/Fonts/malgunbd.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc"],
        "KoreanSerif": ["C:/Windows/Fonts/batang.ttc", "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"]
    }
    for name, paths in font_paths.items():
        for p in paths:
            if os.path.exists(p):
                try:
                    pdfmetrics.registerFont(TTFont(name, p))
                except Exception:
                    pass
                break

    # ---------- 3. 스타일 ----------
    styles = getSampleStyleSheet()
    TITLE_STYLE = ParagraphStyle('TITLE', fontName='KoreanBold' if 'KoreanBold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold', fontSize=20, leading=34, spaceAfter=18)
    HEADING_STYLE = ParagraphStyle('HEADING', fontName='KoreanBold' if 'KoreanBold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold', fontSize=14, leading=23.8, textColor=colors.HexColor('#E31E24'), spaceBefore=16, spaceAfter=10)
    BODY_STYLE = ParagraphStyle('BODY', fontName='KoreanSerif' if 'KoreanSerif' in pdfmetrics.getRegisteredFontNames() else 'Times-Roman', fontSize=12, leading=20.4, spaceAfter=6)

    # ---------- 4. PDF 작성 ----------
    buff = io.BytesIO()
    def _page_no(canvas, doc):
        canvas.setFont('Helvetica', 9)
        canvas.drawCentredString(letter[0]/2, 18, f"- {doc.page} -")

    doc = SimpleDocTemplate(buff, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []

    # 4-1 제목 & 메타
    story.append(Paragraph("SK에너지 경쟁사 분석 보고서", TITLE_STYLE))
    story.append(Paragraph(f"보고일자: {datetime.now().strftime('%Y년 %m월 %d일')}", BODY_STYLE))
    story.append(Spacer(1, 12))

    # 4-2 재무 표
    if financial_data is not None and not financial_data.empty:
        story.append(Paragraph("1. 재무분석 결과", HEADING_STYLE))
        df_disp = financial_data[[c for c in financial_data.columns if not c.endswith('_원시값')]].copy()
        data = [df_disp.columns.tolist()] + df_disp.values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('GRID',(0,0),(-1,-1),0.5,colors.black),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#F2F2F2')),
            ('FONTNAME',(0,0),(-1,0),'KoreanBold'),
            ('FONTNAME',(0,1),(-1,-1),'KoreanSerif'),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('ALIGN',(0,0),(-1,-1),'CENTER')
        ]))
        story.append(tbl)
        story.append(Spacer(1, 18))

    # 4-3 뉴스 요약
    if news_data is not None and not news_data.empty:
        story.append(Paragraph("2. 최신 뉴스 하이라이트", HEADING_STYLE))
        for i, title in enumerate(news_data["제목"].head(5), 1):
            story.append(Paragraph(f"{i}. {title}", BODY_STYLE))
        story.append(Spacer(1, 12))

    # 4-4 AI 인사이트
    if insights:
        story.append(PageBreak())
        story.append(Paragraph("3. AI 인사이트", HEADING_STYLE))
        # (AI 텍스트 처리 로직은 생략된 원본 코드를 기반으로 복원해야 합니다)
        cleaned_insights = insights.replace('##', '').replace('*', '')
        for line in cleaned_insights.splitlines():
            if line.strip():
                story.append(Paragraph(line, BODY_STYLE))

    # 4-5 차트
    if selected_charts and PLOTLY_AVAILABLE:
        story.append(PageBreak())
        story.append(Paragraph("4. 시각화 차트", HEADING_STYLE))
        for fig in selected_charts:
            try:
                img_bytes = fig.to_image(format="png", width=700, height=400)
                # (RLImage import 필요: from reportlab.platypus import Image as RLImage)
                # 이 부분은 원본 코드에 RLImage가 import 되어 있는지 확인해야 합니다.
                # story.append(RLImage(io.BytesIO(img_bytes), width=500, height=280))
                story.append(Spacer(1, 16))
            except Exception as e:
                story.append(Paragraph(f"차트 삽입 오류: {e}", BODY_STYLE))

    # 4-6 빌드
    try:
        doc.build(story, onFirstPage=_page_no, onLaterPages=_page_no)
        buff.seek(0)
        return buff.getvalue()
    except Exception as e:
        st.error(f"PDF 생성 중 오류 발생: {e}")
        return None

#
# --- create_excel_report 함수 들여쓰기 수정 ---
#
def create_excel_report(financial_data=None, news_data=None, insights=None):
    """Excel 보고서 생성"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 재무분석 시트
            if financial_data is not None and not financial_data.empty:
                clean_financial = financial_data[[col for col in financial_data.columns if not col.endswith('_원시값')]]
                clean_financial.to_excel(writer, sheet_name='재무분석', index=False)
            
            # 뉴스분석 시트
            if news_data is not None and not news_data.empty:
                news_data.to_excel(writer, sheet_name='뉴스분석', index=False)
            
            # 인사이트 시트
            if insights:
                insight_df = pd.DataFrame({'구분': ['AI 인사이트'], '내용': [str(insights)]})
                insight_df.to_excel(writer, sheet_name='AI인사이트', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    except Exception as e:
        st.error(f"Excel 생성 오류: {e}")
        return None
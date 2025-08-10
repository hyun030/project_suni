# -*- coding: utf-8 -*-
# 이메일 관련 유틸리티 함수들을 관리합니다.

import streamlit as st
import config

def render_email_links():
    st.write("**📧 이메일 서비스 바로가기**")
    providers = config.MAIL_PROVIDERS
    selected = st.selectbox("메일 서비스 선택", list(providers.keys()))
    url = providers[selected]
    st.markdown(f'<a href="{url}" target="_blank">{selected} 메일 바로가기</a>', unsafe_allow_html=True)
    st.info("선택한 메일 서비스 링크가 새 탭에서 열립니다.")
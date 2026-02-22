from __future__ import annotations
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MMAD Inspector Dashboard", layout="wide")
st.title("MMAD Inspector Dashboard (Step 1 / MVP)")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) 이미지 업로드 → 검사")
    up = st.file_uploader("검사할 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if up is not None:
        st.image(Image.open(up), caption="Input", use_container_width=True)
        if st.button("검사 실행"):
            files = {"file": (up.name, up.getvalue())}
            res = requests.post(f"{API_URL}/inspect", files=files, timeout=300)
            st.session_state["last_result"] = res.json()

with right:
    st.subheader("2) 최신 리포트")
    limit = st.slider("조회 개수", 5, 50, 10)
    lst = requests.get(f"{API_URL}/reports", params={"limit": limit}, timeout=60).json()
    for item in lst.get("items", []):
        with st.expander(f"#{item['id']} | {item['decision']} | {item['timestamp']}"):
            st.write(item)

st.divider()
st.subheader("3) 마지막 검사 결과")
last = st.session_state.get("last_result")
if last:
    report = last["report"]
    st.json(report)
    arts = report.get("artifacts", {})
    overlay = arts.get("overlay_path")
    heatmap = arts.get("heatmap_path")

    c1, c2 = st.columns(2)
    if overlay and Path(overlay).exists():
        with c1:
            st.image(Image.open(overlay), caption="Overlay", use_container_width=True)
    if heatmap and Path(heatmap).exists():
        with c2:
            st.image(Image.open(heatmap), caption="Heatmap", use_container_width=True)

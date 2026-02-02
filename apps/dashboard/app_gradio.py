from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import requests
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def run_inspection(image):
    """이미지 검사 실행"""
    if image is None:
        return None, None, None, "이미지를 업로드해주세요."

    # PIL Image를 바이트로 변환
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("image.png", buf.getvalue(), "image/png")}
    try:
        res = requests.post(f"{API_URL}/inspect", files=files, timeout=300)
        result = res.json()
        report = result.get("report", {})

        # artifacts에서 이미지 경로 추출
        arts = report.get("artifacts", {})
        overlay_path = arts.get("overlay_path")
        heatmap_path = arts.get("heatmap_path")

        overlay_img = None
        heatmap_img = None

        if overlay_path and Path(overlay_path).exists():
            overlay_img = Image.open(overlay_path)
        if heatmap_path and Path(heatmap_path).exists():
            heatmap_img = Image.open(heatmap_path)

        return overlay_img, heatmap_img, json.dumps(report, indent=2, ensure_ascii=False), "검사 완료"
    except Exception as e:
        return None, None, None, f"오류 발생: {str(e)}"


def fetch_reports(limit):
    """최신 리포트 조회"""
    try:
        res = requests.get(f"{API_URL}/reports", params={"limit": int(limit)}, timeout=60)
        data = res.json()
        items = data.get("items", [])

        if not items:
            return "리포트가 없습니다."

        # 리포트 목록을 마크다운 형식으로 변환
        md_lines = []
        for item in items:
            md_lines.append(f"### #{item.get('id', 'N/A')} | {item.get('decision', 'N/A')} | {item.get('timestamp', 'N/A')}")
            md_lines.append("```json")
            md_lines.append(json.dumps(item, indent=2, ensure_ascii=False))
            md_lines.append("```")
            md_lines.append("---")

        return "\n".join(md_lines)
    except Exception as e:
        return f"오류 발생: {str(e)}"


# Gradio UI 구성
with gr.Blocks(title="MMAD Inspector Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MMAD Inspector Dashboard (Step 1 / MVP)")

    with gr.Row():
        # 좌측: 이미지 업로드 → 검사
        with gr.Column(scale=1):
            gr.Markdown("## 1) 이미지 업로드 → 검사")
            input_image = gr.Image(label="검사할 이미지를 업로드하세요", type="pil")
            inspect_btn = gr.Button("검사 실행", variant="primary")
            status_text = gr.Textbox(label="상태", interactive=False)

        # 우측: 최신 리포트
        with gr.Column(scale=1):
            gr.Markdown("## 2) 최신 리포트")
            limit_slider = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="조회 개수")
            fetch_btn = gr.Button("리포트 조회")
            reports_display = gr.Markdown(label="리포트 목록")

    gr.Markdown("---")
    gr.Markdown("## 3) 마지막 검사 결과")

    with gr.Row():
        overlay_output = gr.Image(label="Overlay", type="pil")
        heatmap_output = gr.Image(label="Heatmap", type="pil")

    result_json = gr.Code(label="검사 결과 JSON", language="json")

    # 이벤트 연결
    inspect_btn.click(
        fn=run_inspection,
        inputs=[input_image],
        outputs=[overlay_output, heatmap_output, result_json, status_text]
    )

    fetch_btn.click(
        fn=fetch_reports,
        inputs=[limit_slider],
        outputs=[reports_display]
    )

    # 페이지 로드 시 리포트 조회
    demo.load(
        fn=fetch_reports,
        inputs=[limit_slider],
        outputs=[reports_display]
    )


if __name__ == "__main__":
    demo.launch(server_port=7860)

"""
app.py — Anaglyph 3D Studio
============================
HuggingFace Spaces · Streamlit · Production Build

Upload any image or video → instant vivid red-cyan 3D anaglyph
Powered by Depth Anything V2 (NeurIPS 2024)

FILES NEEDED IN SAME FOLDER:
  app.py          ← this file
  engine.py       ← processing engine
  requirements.txt
"""

import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import time

from engine import (
    DEVICE,
    load_depth_model,
    convert_image,
    VideoConverter,
    bgr_to_rgb,
    encode_png,
    compute_auto_params,
    render_glasses_preview,
)

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Anaglyph 3D Studio",
    layout="wide",
    page_icon="🥽",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"]   { background: #07070d; }
[data-testid="stHeader"]             { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }
[data-testid="stSidebar"]            { background: #0f0f1a; }

.stProgress > div > div > div {
    background: linear-gradient(90deg, #e63946, #06b6d4) !important;
    border-radius: 4px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #e63946, #06b6d4) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: .65rem 1.5rem !important;
    width: 100%;
}
.stButton > button:hover { opacity: .88 !important; }
.stDownloadButton > button {
    background: rgba(6,182,212,0.12) !important;
    color: #06b6d4 !important;
    border: 1px solid rgba(6,182,212,0.4) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    width: 100%;
}
.stDownloadButton > button:hover { background: rgba(6,182,212,0.22) !important; }
h1, h2, h3 { color: #eeeef8 !important; }
p, li       { color: #9898b8 !important; }
[data-testid="stFileUploader"] {
    background: #16162a;
    border: 2px dashed rgba(6,182,212,0.35);
    border-radius: 16px; padding: 1rem;
}
div[data-testid="metric-container"] {
    background: #16162a; border: 1px solid #2a2a4a;
    border-radius: 10px; padding: .8rem 1rem;
}
.stTabs [data-baseweb="tab-list"] { background: #0f0f1a; border-radius: 10px; gap: 4px; }
.stTabs [data-baseweb="tab"]      { color: #9898b8 !important; border-radius: 8px !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"]    { background: rgba(6,182,212,0.15) !important; color: #eeeef8 !important; }
.stAlert { border-radius: 10px !important; }
.param-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .6rem 0; }
.param-badge {
    background: #16162a; border: 1px solid #2a2a4a;
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; color: #9898b8;
}
.param-badge b { color: #06b6d4; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL  (cached — downloads once, lives in memory)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    return load_depth_model()

# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='
    background: linear-gradient(90deg,#e63946,#ff8c94 38%,#06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.5px; margin-bottom: 0;
'>🥽 Anaglyph 3D Studio</h1>
<p style='color:#9898b8; margin-top:.35rem; font-size:15px'>
    Upload any image or video → vivid cinema-grade red-cyan 3D anaglyph<br>
    <small style='color:#555'>
        Depth Anything V2 (NeurIPS 2024) · Auto-optimised · No settings needed
    </small>
</p>
""", unsafe_allow_html=True)

st.markdown("---")

with st.spinner("⏳ Loading Depth Anything V2 AI model..."):
    processor, model = get_model()

engine_name = "✅ Depth Anything V2 (AI)" if model else "⚠️ Classical CV (fallback)"
device_name = {"cuda": "🟢 GPU", "mps": "🟡 Apple MPS", "cpu": "🔵 CPU"}.get(
    str(DEVICE.type), str(DEVICE)
)

col_l, col_r = st.columns([3, 1])
with col_l:
    st.caption(f"Engine: **{engine_name}** · Device: `{device_name}` · Parameters auto-computed per image")
with col_r:
    st.markdown(
        "<div style='text-align:right'>"
        "<span style='background:#16162a;border:1px solid #2a2a4a;"
        "border-radius:20px;padding:5px 14px;font-size:11px;color:#06b6d4'>"
        "👓 Use red-cyan glasses</span></div>",
        unsafe_allow_html=True
    )

st.markdown("")

# ══════════════════════════════════════════════════════════════════
#  HELPER — params badge strip
# ══════════════════════════════════════════════════════════════════
def show_params(params: dict, elapsed: float):
    total = params.get("total_parallax", params["base_shift"] * 2)
    badges = [
        ("Total parallax", f"{total}px"),
        ("Per eye",        f"{params['base_shift']}px"),
        ("Depth strength", f"{params['depth_strength']:.2f}"),
        ("Resolution",     f"{params['width']}×{params['height']}"),
        ("Time",           f"{elapsed:.2f}s"),
    ]
    html = '<div class="param-strip">'
    for label, val in badges:
        html += f'<span class="param-badge">{label}: <b>{val}</b></span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════
tab_img, tab_vid, tab_sim = st.tabs(["🖼  Image", "🎬  Video", "🥽  Glasses Simulator"])


# ══════════════════════════════════════════════════════════════════
#  IMAGE TAB
# ══════════════════════════════════════════════════════════════════
with tab_img:
    st.markdown("### Upload Image")
    st.caption("JPG · PNG · WebP · BMP · Any resolution — processed at full quality")

    uploaded = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_upload",
    )

    if uploaded:
        raw     = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the image. Please try another file.")
        else:
            h, w = img_bgr.shape[:2]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Resolution",  f"{w}×{h}")
            m2.metric("File",        uploaded.name[:22])
            m3.metric("Method",      "Vivid Full-Color")
            m4.metric("Device",      DEVICE.type.upper())
            st.markdown("")

            with st.spinner("🔬 Running AI depth estimation and generating 3D..."):
                t0 = time.time()
                anaglyph, depth_vis, params = convert_image(img_bgr, processor, model)
                elapsed = time.time() - t0

            st.success(f"✅ 3D anaglyph generated in **{elapsed:.2f}s**")
            show_params(params, elapsed)

            # ── 3-column result ───────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original**")
                st.image(bgr_to_rgb(img_bgr), use_container_width=True)
            with c2:
                st.markdown("**🥽 3D Anaglyph**")
                st.image(bgr_to_rgb(anaglyph), use_container_width=True)
            with c3:
                st.markdown("**AI Depth Map**")
                st.image(bgr_to_rgb(depth_vis), use_container_width=True)

            st.markdown("")

            # ── Downloads ─────────────────────────────────────────
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇ Download 3D Anaglyph (PNG — lossless)",
                    encode_png(anaglyph),
                    file_name="anaglyph_3d.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "⬇ Download Depth Map (PNG)",
                    encode_png(depth_vis),
                    file_name="depth_map.png",
                    mime="image/png",
                    use_container_width=True,
                )

            st.info(
                "👓 **How to view:** Wear red-cyan 3D glasses. "
                "**Red lens over LEFT eye** · Cyan lens over RIGHT eye."
            )

            # ── Glasses preview ───────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🥽 Glasses On — Your 3D View")
            gp = render_glasses_preview(anaglyph, canvas_w=900)
            st.image(
                cv2.cvtColor(gp, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Put on your red-cyan glasses — this is exactly what you will see"
            )


# ══════════════════════════════════════════════════════════════════
#  VIDEO TAB
# ══════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown("### Upload Video")
    st.caption("MP4 · AVI · MOV · MKV · WebM — original resolution and FPS preserved")

    vid_file = st.file_uploader(
        "Drop your video here or click to browse",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="vid_upload",
    )

    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        input_path = tfile.name
        tfile.close()

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Cannot open video. Please try another file.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps      = cap.get(cv2.CAP_PROP_FPS) or 24.0
            src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / max(src_fps, 1)

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Resolution", f"{src_w}×{src_h}")
            v2.metric("Frame Rate", f"{src_fps:.0f} fps")
            v3.metric("Duration",   f"{duration_sec:.1f}s")
            v4.metric("Frames",     f"{total_frames:,}")
            st.markdown("")

            if st.button("🎬 Convert to 3D Anaglyph", use_container_width=True):
                fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
                out       = cv2.VideoWriter(output_path, fourcc, src_fps, (src_w, src_h))
                converter = VideoConverter(processor, model)

                progress   = st.progress(0.0)
                status     = st.empty()
                preview_ph = st.empty()
                start      = time.time()
                frame_idx  = 0
                last_ana   = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    anaglyph_f, depth_f = converter.process_frame(frame)
                    out.write(anaglyph_f)
                    last_ana   = anaglyph_f
                    frame_idx += 1

                    if frame_idx % 5 == 0:
                        pct       = frame_idx / max(total_frames, 1)
                        elapsed_v = time.time() - start
                        fps_proc  = frame_idx / max(elapsed_v, 0.1)
                        remaining = (total_frames - frame_idx) / max(fps_proc, 0.1)
                        progress.progress(min(pct, 1.0))
                        status.markdown(
                            f"Frame **{frame_idx:,}** / {total_frames:,} · "
                            f"Speed: **{fps_proc:.1f} fps** · "
                            f"Remaining: **{remaining:.0f}s**"
                        )
                    if frame_idx % 20 == 0 and last_ana is not None:
                        preview_ph.image(
                            bgr_to_rgb(last_ana),
                            caption="Live preview",
                            use_container_width=True,
                        )

                cap.release()
                out.release()
                total_time = time.time() - start

                progress.progress(1.0)
                status.markdown(
                    f"✅ Done! **{frame_idx:,}** frames in **{total_time:.1f}s** "
                    f"({frame_idx / max(total_time, 0.1):.1f} fps avg)"
                )
                preview_ph.empty()

                if converter.params:
                    show_params(converter.params, total_time)

                # Last frame comparison
                if last_ana is not None:
                    st.markdown("#### Last Frame Preview")
                    lc1, lc2 = st.columns(2)
                    cap_prev = cv2.VideoCapture(input_path)
                    cap_prev.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                    ret_p, prev_orig = cap_prev.read()
                    cap_prev.release()
                    with lc1:
                        st.markdown("**Original**")
                        if ret_p:
                            st.image(bgr_to_rgb(prev_orig), use_container_width=True)
                    with lc2:
                        st.markdown("**🥽 3D Anaglyph**")
                        st.image(bgr_to_rgb(last_ana), use_container_width=True)

                # Download
                st.markdown("")
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        "⬇ Download 3D Anaglyph Video (MP4 · original FPS & resolution)",
                        video_bytes,
                        file_name="anaglyph_3d.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

                st.info(
                    "👓 **How to view:** Wear red-cyan 3D glasses. "
                    "**Red lens over LEFT eye** · Cyan lens over RIGHT eye."
                )

                for p in (input_path, output_path):
                    try: os.unlink(p)
                    except Exception: pass


# ══════════════════════════════════════════════════════════════════
#  GLASSES SIMULATOR TAB
# ══════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("### 🥽 See Your 3D Result Through the Glasses")
    st.caption("Upload the anaglyph you generated — see exactly what it looks like with glasses on.")

    sim_upload = st.file_uploader(
        "Drop your anaglyph image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="sim_upload",
    )

    if sim_upload:
        raw_sim     = np.frombuffer(sim_upload.read(), np.uint8)
        anaglyph_in = cv2.imdecode(raw_sim, cv2.IMREAD_COLOR)

        if anaglyph_in is None:
            st.error("Could not read image.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Without glasses**")
                st.image(bgr_to_rgb(anaglyph_in), use_container_width=True)
            with col_b:
                st.markdown("**🥽 Glasses on — your 3D view**")
                with st.spinner("Applying lens simulation..."):
                    glasses_view = render_glasses_preview(anaglyph_in, canvas_w=900)
                st.image(
                    cv2.cvtColor(glasses_view, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            st.download_button(
                "⬇ Download Glasses View",
                encode_png(glasses_view),
                file_name="glasses_view.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.markdown(
            "<div style='background:#16162a;border:2px dashed #2a2a4a;"
            "border-radius:16px;padding:4rem 2rem;text-align:center'>"
            "<p style='font-size:52px;margin-bottom:.8rem'>🥽</p>"
            "<p style='color:#9898b8;font-size:16px;margin-bottom:.4rem'>"
            "Generate an anaglyph in the Image tab first, then upload it here</p>"
            "<p style='color:#555;font-size:13px'>"
            "See exactly what your 3D image looks like through red-cyan glasses</p>"
            "</div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#3a3a5a;font-size:12px;padding:.5rem 0'>
    Anaglyph 3D Studio &nbsp;·&nbsp;
    Depth Anything V2 (NeurIPS 2024) &nbsp;·&nbsp;
    Vivid Full-Color Anaglyph &nbsp;·&nbsp;
    PyTorch · OpenCV · Streamlit
</div>
""", unsafe_allow_html=True)

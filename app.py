"""
app.py — Anaglyph 3D AI Studio
================================
Production-ready Streamlit UI.
Processing logic lives entirely in engine.py — this file is UI only.

INSTALL:
  pip install streamlit opencv-python-headless numpy torch transformers pillow

RUN:
  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

# ── Import our separated engine ───────────────────────────────────
from engine import (
    DEVICE,
    load_depth_model,
    convert_image,
    VideoConverter,
    bgr_to_rgb,
    encode_png,
    compute_auto_params,
    render_glasses_preview,
    add_glasses_frame_overlay,
)

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Anaglyph 3D Studio",
    layout="wide",
    page_icon="🥽",
    initial_sidebar_state="collapsed",    # No sidebar needed — no controls
)

# ══════════════════════════════════════════════════════════════════
#  CSS — Cinema dark theme, production quality
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Core layout ── */
[data-testid="stAppViewContainer"]  { background: #07070d; }
[data-testid="stHeader"]            { background: transparent; }
[data-testid="stSidebar"]           { background: #0f0f1a; }
[data-testid="stMainBlockContainer"]{ padding-top: 1.5rem; }

/* ── Progress bar: red-cyan gradient ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #e63946, #06b6d4) !important;
    border-radius: 4px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e63946, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: .65rem 1.5rem !important;
    width: 100%;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── Download buttons ── */
.stDownloadButton > button {
    background: rgba(6,182,212,0.12) !important;
    color: #06b6d4 !important;
    border: 1px solid rgba(6,182,212,0.4) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: rgba(6,182,212,0.22) !important;
}

/* ── Typography ── */
h1, h2, h3 { color: #eeeef8 !important; }
p, li       { color: #9898b8 !important; }
.stMarkdown p { color: #9898b8 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #16162a;
    border: 2px dashed rgba(6,182,212,0.35);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color .2s;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #16162a;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: .8rem 1rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 10px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #9898b8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(6,182,212,0.15) !important;
    color: #eeeef8 !important;
}

/* ── Info / Success boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Image comparison card ── */
.img-card {
    background: #16162a;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: .75rem;
    text-align: center;
}

/* ── Params badge strip ── */
.param-strip {
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
    margin: .6rem 0;
}
.param-badge {
    background: #16162a;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #9898b8;
}
.param-badge b { color: #06b6d4; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LOAD AI MODEL  (cached — runs once per session)
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
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0;
'>🥽 Anaglyph 3D Studio</h1>
<p style='color:#9898b8;margin-top:.35rem;font-size:15px'>
    Upload any image or video → instant cinema-grade red-cyan 3D anaglyph<br>
    <small style='color:#555'>
        Depth Anything V2 (NeurIPS 2024) · Auto-optimised · No settings needed
    </small>
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Device + model status row ─────────────────────────────────────
col_l, col_r = st.columns([3, 1])
with col_l:
    with st.spinner("Loading Depth Anything V2..."):
        processor, model = get_model()

    engine_name = "Depth Anything V2 (AI)" if model else "Classical CV (fallback)"
    device_name = {"cuda": "🟢 GPU (CUDA)", "mps": "🟡 Apple MPS", "cpu": "🔵 CPU"}.get(
        str(DEVICE.type), str(DEVICE)
    )
    st.caption(
        f"Engine: **{engine_name}** · Device: `{device_name}` · "
        f"All parameters auto-computed per image"
    )

with col_r:
    st.markdown(
        "<div style='text-align:right;padding-top:.3rem'>"
        "<span style='background:#16162a;border:1px solid #2a2a4a;"
        "border-radius:20px;padding:5px 14px;font-size:11px;color:#06b6d4'>"
        "👓 Use red-cyan glasses to view</span></div>",
        unsafe_allow_html=True
    )

st.markdown("")


# ══════════════════════════════════════════════════════════════════
#  HELPER: render auto-params as a readable badge strip
# ══════════════════════════════════════════════════════════════════
def show_params(params: dict, elapsed: float):
    total = params.get("total_parallax", params["base_shift"] * 2)
    badges = [
        ("Total parallax", f"{total}px"),
        ("Per eye",        f"{params['base_shift']}px"),
        ("Depth str",      f"{params['depth_strength']:.2f}"),
        ("Res",            f"{params['width']}×{params['height']}"),
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
    st.caption("JPG · PNG · WebP · BMP · Any resolution — processed at full HD quality")

    uploaded = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_upload",
    )

    if uploaded:
        # ── Decode at FULL quality — no compression introduced ────
        raw     = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the image. Please try another file.")
        else:
            with st.spinner("Running AI depth estimation and generating 3D..."):
                t0 = time.time()
                anaglyph, depth_vis, params = convert_image(img_bgr, processor, model)
                elapsed = time.time() - t0

            st.success(f"✅ 3D anaglyph generated in **{elapsed:.2f}s**")
            show_params(params, elapsed)

            # ── Side-by-side comparison ───────────────────────────
            st.markdown("#### Result")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    "<div class='img-card'><p style='color:#9898b8;font-size:12px;"
                    "margin-bottom:.4rem'>ORIGINAL</p></div>",
                    unsafe_allow_html=True
                )
                st.image(bgr_to_rgb(img_bgr), use_container_width=True)

            with c2:
                st.markdown(
                    "<div class='img-card'><p style='color:#06b6d4;font-size:12px;"
                    "margin-bottom:.4rem'>🥽 3D ANAGLYPH</p></div>",
                    unsafe_allow_html=True
                )
                st.image(bgr_to_rgb(anaglyph), use_container_width=True)

            with c3:
                st.markdown(
                    "<div class='img-card'><p style='color:#f59e0b;font-size:12px;"
                    "margin-bottom:.4rem'>AI DEPTH MAP</p></div>",
                    unsafe_allow_html=True
                )
                st.image(bgr_to_rgb(depth_vis), use_container_width=True)

            st.markdown("")

            # ── Download — PNG, lossless ──────────────────────────
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇ Download 3D Anaglyph (PNG — lossless)",
                    encode_png(anaglyph),           # lossless PNG
                    file_name="anaglyph_3d.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "⬇ Download Depth Map (PNG)",
                    encode_png(depth_vis),
                    file_name="depth_map.png",
                    mime="image/png",
                    use_container_width=True,
                )

            st.info(
                "👓 **How to view:** Wear red-cyan 3D glasses. "
                "Red lens over **LEFT** eye · Cyan lens over **RIGHT** eye."
            )

            # ── GLASSES PREVIEW — show the 3D result as seen through glasses ──
            st.markdown("---")
            st.markdown("#### 🥽 Glasses On — This is your 3D view")
            _gp = render_glasses_preview(anaglyph, canvas_w=900)
            st.image(
                cv2.cvtColor(_gp, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Put on your red-cyan glasses and look at the screen — this is exactly what you will see"
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
        # ── Write to temp file (OpenCV needs a path) ──────────────
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        input_path = tfile.name
        tfile.close()

        output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Cannot open video. Please try another file.")
        else:
            # ── Read video metadata ───────────────────────────────
            total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps       = cap.get(cv2.CAP_PROP_FPS) or 24.0
            src_w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec  = total_frames / max(src_fps, 1)

            # ── Video info display ────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Resolution",  f"{src_w}×{src_h}")
            m2.metric("Frame Rate",  f"{src_fps:.0f} fps")
            m3.metric("Duration",    f"{duration_sec:.1f}s")
            m4.metric("Frames",      f"{total_frames:,}")

            st.markdown("")

            # ── Convert button ────────────────────────────────────
            if st.button("🎬 Convert to 3D Anaglyph", use_container_width=True):

                # Setup VideoWriter — SAME resolution as source
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out    = cv2.VideoWriter(
                    output_path, fourcc, src_fps, (src_w, src_h)
                )

                # Progress UI
                progress    = st.progress(0.0)
                status      = st.empty()
                metrics_row = st.empty()

                # Initialize stateful converter
                converter = VideoConverter(processor, model)
                start     = time.time()
                frame_idx = 0
                last_ana  = None
                last_dep  = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame (temporal smoothing inside converter)
                    anaglyph_f, depth_f = converter.process_frame(frame)
                    out.write(anaglyph_f)

                    last_ana = anaglyph_f
                    last_dep = depth_f
                    frame_idx += 1

                    # Update UI every 5 frames
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

                cap.release()
                out.release()

                total_time = time.time() - start
                progress.progress(1.0)
                status.markdown(
                    f"✅ Done! **{frame_idx:,}** frames in **{total_time:.1f}s** "
                    f"({frame_idx/max(total_time,0.1):.1f} fps avg)"
                )
                metrics_row.empty()

                # ── Show params used ──────────────────────────────
                if converter.params:
                    show_params(converter.params, total_time)

                # ── Last frame preview ────────────────────────────
                if last_ana is not None:
                    st.markdown("#### Last Frame Preview")
                    prev_c1, prev_c2 = st.columns(2)
                    with prev_c1:
                        st.markdown("<p style='color:#9898b8;font-size:12px'>Original</p>",
                                    unsafe_allow_html=True)
                        # Re-read last original frame for comparison
                        cap_prev = cv2.VideoCapture(input_path)
                        cap_prev.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                        ret_p, prev_orig = cap_prev.read()
                        cap_prev.release()
                        if ret_p:
                            st.image(bgr_to_rgb(prev_orig), use_container_width=True)
                    with prev_c2:
                        st.markdown("<p style='color:#06b6d4;font-size:12px'>🥽 3D Anaglyph</p>",
                                    unsafe_allow_html=True)
                        st.image(bgr_to_rgb(last_ana), use_container_width=True)

                # ── Download button ───────────────────────────────
                st.markdown("")
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()

                    st.download_button(
                        "⬇ Download 3D Anaglyph Video (MP4)",
                        video_bytes,
                        file_name="anaglyph_3d.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

                st.info(
                    "👓 **How to view:** Wear red-cyan 3D glasses. "
                    "Red lens over **LEFT** eye · Cyan lens over **RIGHT** eye."
                )

                # ── Cleanup temp files ────────────────────────────
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════
#  GLASSES SIMULATOR TAB
# ══════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("### 🥽 See Your 3D Result Through the Glasses")
    st.caption("Upload the anaglyph you generated — see exactly what it looks like when you put on your glasses.")

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
            # Side by side: anaglyph vs glasses view
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    "<p style='color:#9898b8;font-size:13px;font-weight:600;"
                    "margin-bottom:.4rem'>Without glasses</p>",
                    unsafe_allow_html=True)
                st.image(bgr_to_rgb(anaglyph_in), use_container_width=True)

            with col_b:
                st.markdown(
                    "<p style='color:#06b6d4;font-size:13px;font-weight:600;"
                    "margin-bottom:.4rem'>Glasses on — this is your 3D view</p>",
                    unsafe_allow_html=True)
                glasses_view = render_glasses_preview(anaglyph_in, canvas_w=800)
                st.image(
                    cv2.cvtColor(glasses_view, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            # Full width immersive view
            st.markdown("#### Full Screen — Glasses On")
            glasses_full = render_glasses_preview(anaglyph_in, canvas_w=1200)
            st.image(
                cv2.cvtColor(glasses_full, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Put on your red-cyan glasses and look at the 3D image displayed through the lenses above"
            )

            st.download_button(
                "⬇ Download Glasses View",
                encode_png(cv2.cvtColor(glasses_full, cv2.COLOR_RGB2BGR)
                           if glasses_full.shape[2] == 3
                           else glasses_full),
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
            "Drop your anaglyph here to see it through the glasses</p>"
            "<p style='color:#555;font-size:13px'>"
            "Generate one in the Image tab first, then come back here</p>"
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
    PyTorch · OpenCV · Streamlit &nbsp;·&nbsp;
    No training needed · Auto-optimised
</div>
""", unsafe_allow_html=True)

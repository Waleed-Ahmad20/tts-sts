import os
import io
import time
import json
import shutil
import threading
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# Project paths
ROOT = Path(__file__).parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
MODEL_CACHE = ROOT / "model_cache"
BUILD_PS1 = ROOT / "build.ps1"

PDF_NAME = "book.pdf"
VOICE_NAME = "voice_sample.wav"

st.set_page_config(page_title="PDF → Audiobook (XTTS)", layout="wide")

# Hint caches for model/data so Streamlit Cloud can persist between runs
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / ".cache").resolve()))
os.environ.setdefault("HF_HOME", str((ROOT / ".cache").resolve()))
os.environ.setdefault("TTS_HOME", str(MODEL_CACHE.resolve()))

# --- Helpers ---

def ensure_dirs():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_CACHE.mkdir(exist_ok=True)


def save_uploaded(file, target_path: Path):
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file, f)


class StreamlitLogHandler(logging.Handler):
    """A logging handler that streams logs into a Streamlit code box."""
    def __init__(self, placeholder: st.delta_generator.DeltaGenerator):
        super().__init__()
        self.placeholder = placeholder
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.lines.append(msg)
            # Keep last N lines for performance
            self.lines = self.lines[-500:]
            self.placeholder.code("\n".join(self.lines), language="text")
        except Exception:
            pass


def run_pipeline_with_logs(resume: bool, keep_chunks: bool, pdf_path: Path | None = None, voice_path: Path | None = None, log_placeholder: Optional[st.delta_generator.DeltaGenerator] = None):
    """Run the pipeline and live-stream logs to Streamlit."""
    # Import lazily after environment variables are set
    from app import main as run_pipeline
    from app import Config as PipelineConfig
    logger = logging.getLogger()
    # Temporarily attach our handler
    handler = None
    if log_placeholder is not None:
        handler = StreamlitLogHandler(log_placeholder)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    prev_level = logger.level
    try:
        logger.setLevel(logging.INFO)
        # Override config paths if custom inputs provided
        if pdf_path:
            PipelineConfig.INPUT_PDF = pdf_path
        if voice_path:
            PipelineConfig.VOICE_WAV = voice_path
        # Run
        run_pipeline(resume=resume, keep_chunks=keep_chunks)
    finally:
        logger.setLevel(prev_level)
        if handler is not None:
            logger.removeHandler(handler)


# --- UI ---

st.title("Text-to-Speech Audiobook Generator (XTTS)")
st.caption("Upload a PDF and a short voice sample, then run the pipeline directly. Optimized for Streamlit Community Cloud (no Docker required).")

ensure_dirs()

# Left column: inputs
left, right = st.columns([1, 1])

with left:
    st.subheader("Inputs")
    uploaded_pdf = st.file_uploader("PDF", type=["pdf"], key="pdf")
    uploaded_voice = st.file_uploader("Voice sample (WAV)", type=["wav"], key="voice")

    col1, col2 = st.columns(2)
    with col1:
        keep_chunks = st.checkbox("Keep chunks after run", value=False)
    with col2:
        no_resume = st.checkbox("Do not resume", value=False, help="Regenerate all chunks")

    if st.button("Save uploads to input/ and prepare", type="primary"):
        # Save as required filenames if provided
        if uploaded_pdf is not None:
            save_uploaded(uploaded_pdf, INPUT_DIR / PDF_NAME)
            st.success(f"Saved PDF → {INPUT_DIR/ PDF_NAME}")
        if uploaded_voice is not None:
            save_uploaded(uploaded_voice, INPUT_DIR / VOICE_NAME)
            st.success(f"Saved Voice → {INPUT_DIR/ VOICE_NAME}")

        # Sanity check presence
        pdf_exists = (INPUT_DIR / PDF_NAME).exists()
        voice_exists = (INPUT_DIR / VOICE_NAME).exists()
        if not pdf_exists:
            st.warning("No PDF found. Either upload or ensure input/book.pdf exists.")
        if not voice_exists:
            st.warning("No voice sample found. Either upload or ensure input/voice_sample.wav exists.")

with right:
    st.subheader("Run pipeline")
    st.write("This runs the Python pipeline directly. On first run, the XTTS model will download (~hundreds of MB).")

    # Advanced options
    with st.expander("Advanced options"):
        model_id = st.text_input("Model id", value="tts_models/multilingual/multi-dataset/xtts_v2")
        language = st.text_input("Language code", value="en")
        require_voice = st.checkbox("Require voice sample (voice cloning)", value=True, help="Disable to use a built-in multi-speaker model (no cloning)")
        speaker = st.text_input("Speaker (for multi-speaker models)", value="")
        limit_pages = st.number_input("Optional: Limit PDF pages (0 = all)", min_value=0, max_value=2000, value=0, step=1, help="Use to test small runs on Streamlit Cloud")
        limit_chunks = st.number_input("Optional: Limit text chunks (0 = all)", min_value=0, max_value=5000, value=0, step=1)

    # Note: app.py doesn't support page limiting; we implement a simple pre-trim here by copying a subset PDF if requested
    def maybe_create_limited_pdf(src: Path, dst: Path, num_pages: int) -> Path:
        if num_pages <= 0:
            return src
        try:
            from pypdf import PdfReader, PdfWriter
            r = PdfReader(str(src))
            w = PdfWriter()
            for i in range(min(num_pages, len(r.pages))):
                w.add_page(r.pages[i])
            with open(dst, "wb") as f:
                w.write(f)
            return dst
        except Exception:
            return src

    if st.button("Run pipeline", type="secondary"):
        st.session_state["run_started_at"] = time.time()

        log_box = st.empty()
        output_placeholder = st.empty()

        # Prepare potential limited PDF
        pdf_to_use = INPUT_DIR / PDF_NAME
        if limit_pages > 0 and pdf_to_use.exists():
            limited_pdf = INPUT_DIR / f"limited_{limit_pages}_pages.pdf"
            pdf_to_use = maybe_create_limited_pdf(INPUT_DIR / PDF_NAME, limited_pdf, limit_pages)

        with st.status("Running pipeline... This can take several minutes on CPU.", state="running"):
            try:
                # Apply UI overrides to config before running
                from app import Config as C
                C.MODEL_ID = model_id.strip() or C.MODEL_ID
                C.LANGUAGE = language.strip() or C.LANGUAGE
                C.REQUIRE_VOICE = bool(require_voice)
                C.SPEAKER = speaker.strip() or None
                C.MAX_CHUNKS_TO_SYNTH = int(limit_chunks) if limit_chunks else 0

                run_pipeline_with_logs(
                    resume=not no_resume,
                    keep_chunks=keep_chunks,
                    pdf_path=pdf_to_use if pdf_to_use.exists() else None,
                    voice_path=(INPUT_DIR / VOICE_NAME if (INPUT_DIR / VOICE_NAME).exists() else None),
                    log_placeholder=log_box,
                )
                st.success("Pipeline completed")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

        # Final refresh of outputs
        wavs = sorted((OUTPUT_DIR / "chunks").glob("*.wav")) if (OUTPUT_DIR / "chunks").exists() else []
        txts = sorted((OUTPUT_DIR / "chunks").glob("*.txt")) if (OUTPUT_DIR / "chunks").exists() else []
        final_wav = OUTPUT_DIR / "audiobook.wav"
        final_mp3 = OUTPUT_DIR / "audiobook.mp3"
        with output_placeholder.container():
            st.write(f"Chunks: {len(txts)} text, {len(wavs)} audio")
            if final_wav.exists():
                st.audio(str(final_wav))
                st.success("Final WAV ready")
            if final_mp3.exists():
                st.audio(str(final_mp3))
                st.success("Final MP3 ready")

st.divider()

# Quick view of current inputs and outputs
with st.expander("Current files"):
    st.write("input/")
    for p in sorted(INPUT_DIR.glob("*")):
        st.write("-", p.name, p.stat().st_size, "bytes")

    st.write("output/")
    if OUTPUT_DIR.exists():
        for p in sorted(OUTPUT_DIR.glob("**/*")):
            if p.is_file():
                st.write("-", p.relative_to(OUTPUT_DIR))
    else:
        st.write("(none)")

# TTS-STS Streamlit App

A Streamlit app that converts a PDF into an audiobook using Coqui TTS (XTTS voice cloning or other supported models).

## Deploy to Streamlit Community Cloud

- App file: `streamlit_app.py`
- Python: 3.10
- Dependencies: `requirements.txt`
- System packages: `packages.txt` (tesseract, poppler-utils, ffmpeg)

## Usage

1. Upload a PDF and an optional voice sample WAV (10–60 seconds works well).
2. Choose model options in Advanced options:
   - Default: `tts_models/multilingual/multi-dataset/xtts_v2` (voice cloning; requires voice sample)
   - You can switch to a built-in multi-speaker model and disable voice requirement.
3. Click Run pipeline. First run downloads the model and can take several minutes.

Results will appear under `output/` (WAV and MP3). Intermediate chunks are in `output/chunks/` and can be kept via the checkbox.

## Notes

- On CPU-only environments (Streamlit Cloud), synthesis is slow. Use the chunk/page limits to test.
- OCR fallback requires `tesseract-ocr` and `poppler-utils` (provided via `packages.txt`).
- Large models require significant disk; Streamlit’s cache may evict between deployments.

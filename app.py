import os
import re
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# PDF / OCR
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Audio
from pydub import AudioSegment

# Coqui TTS
import torch
from TTS.api import TTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration
class Config:
    INPUT_PDF = Path("input/book.pdf")
    VOICE_WAV = Path("input/voice_sample.wav")
    OUT_DIR = Path("output")
    CHUNKS_DIR = OUT_DIR / "chunks"
    TEXT_PATH = OUT_DIR / "extracted_text.txt"
    FINAL_WAV = OUT_DIR / "audiobook.wav"
    FINAL_MP3 = OUT_DIR / "audiobook.mp3"
    
    # Chunking parameters - Optimized for XTTS
    MAX_CHARS_PER_CHUNK = 250   # Slightly increased for better context
    MIN_CHUNK_CHARS = 50        # Skip very short chunks
    
    # XTTS model
    MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
    # If using non-cloning multi-speaker models (e.g. vctk/vits), set SPEAKER
    SPEAKER: Optional[str] = None
    # Language hint (used by XTTS)
    LANGUAGE: str = "en"
    # Require a voice sample (only for cloning models)
    REQUIRE_VOICE: bool = True
    # Limit the number of chunks to synthesize (useful on Streamlit Cloud). 0 = no limit
    MAX_CHUNKS_TO_SYNTH: int = 0
    
    # Audio settings
    MP3_BITRATE = "192k"
    PAUSE_BETWEEN_CHUNKS = 0.5  # seconds

def validate_inputs() -> None:
    """Validate required input files exist and are valid."""
    if not Config.INPUT_PDF.exists():
        raise FileNotFoundError(f"PDF file not found: {Config.INPUT_PDF}")
    
    if Config.REQUIRE_VOICE:
        if not Config.VOICE_WAV.exists():
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_WAV}")
    
    # Validate voice sample
    if Config.REQUIRE_VOICE:
        try:
            voice_audio = AudioSegment.from_file(Config.VOICE_WAV)
            duration = len(voice_audio) / 1000  # Convert to seconds
            if duration < 6:
                logging.warning(f"Voice sample is {duration:.1f}s. Recommend 10-60s for better quality.")
            elif duration > 120:
                logging.warning(f"Voice sample is {duration:.1f}s. Very long samples may not improve quality.")
        except Exception as e:
            raise ValueError(f"Invalid voice sample file: {e}")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF with fallback to OCR if needed."""
    logging.info("Extracting text from PDF...")
    
    # First try direct text extraction
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            # Clean up common PDF extraction artifacts
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
            text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', text)  # Fix hyphenated words
            text_parts.append(text)
            
        combined_text = "\n\n".join(text_parts).strip()
        
        # Check if extraction was successful (heuristic: should have reasonable word count)
        word_count = len(combined_text.split())
        if word_count > 100:  # Reasonable threshold
            logging.info(f"Successfully extracted {len(combined_text)} characters, {word_count} words")
            return combined_text
        
    except Exception as e:
        logging.warning(f"Direct text extraction failed: {e}")
    
    # Fallback to OCR
    logging.info("Using OCR fallback...")
    try:
        pages = convert_from_path(str(pdf_path), dpi=200)  # Reduced from 300 for speed
        ocr_texts = []
        
        for i, image in enumerate(pages, 1):
            text = pytesseract.image_to_string(image, lang="eng")
            # Clean OCR artifacts
            text = re.sub(r'\n\s*\n', '\n\n', text)
            ocr_texts.append(text)
            logging.info(f"OCR page {i}/{len(pages)}: {len(text)} chars")
        
        combined_text = "\n\n".join(ocr_texts).strip()
        logging.info(f"OCR complete: {len(combined_text)} characters")
        return combined_text
        
    except Exception as e:
        raise RuntimeError(f"Both text extraction and OCR failed: {e}")

def smart_chunk_text(text: str, max_chars: int = Config.MAX_CHARS_PER_CHUNK) -> List[str]:
    """Intelligently chunk text preserving sentence and paragraph boundaries."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph fits in current chunk, add it
        if len(current_chunk) + len(paragraph) + 2 <= max_chars:  # +2 for \n\n
            current_chunk = (current_chunk + "\n\n" + paragraph).strip()
        else:
            # Save current chunk if it exists
            if current_chunk and len(current_chunk) >= Config.MIN_CHUNK_CHARS:
                chunks.append(current_chunk)
            
            # Handle oversized paragraphs
            if len(paragraph) > max_chars:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 <= max_chars:
                        temp_chunk = (temp_chunk + " " + sentence).strip()
                    else:
                        if temp_chunk and len(temp_chunk) >= Config.MIN_CHUNK_CHARS:
                            chunks.append(temp_chunk)
                        
                        # Handle oversized sentences (rare but possible)
                        if len(sentence) > max_chars:
                            # Hard split at word boundaries
                            words = sentence.split()
                            word_chunk = ""
                            for word in words:
                                if len(word_chunk) + len(word) + 1 <= max_chars:
                                    word_chunk = (word_chunk + " " + word).strip()
                                else:
                                    if word_chunk:
                                        chunks.append(word_chunk)
                                    word_chunk = word
                            if word_chunk:
                                temp_chunk = word_chunk
                            else:
                                temp_chunk = ""
                        else:
                            temp_chunk = sentence
                
                if temp_chunk and len(temp_chunk) >= Config.MIN_CHUNK_CHARS:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = paragraph
    
    # Don't forget the last chunk
    if current_chunk and len(current_chunk) >= Config.MIN_CHUNK_CHARS:
        chunks.append(current_chunk)
    
    # Filter out very short chunks
    filtered_chunks = [c for c in chunks if len(c) >= Config.MIN_CHUNK_CHARS]
    
    logging.info(f"Created {len(filtered_chunks)} chunks from {len(chunks)} raw chunks")
    return filtered_chunks

def setup_directories():
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    # Create chunks directory
    os.makedirs(Config.CHUNKS_DIR, exist_ok=True)
    
    for filename in os.listdir(Config.OUT_DIR):
        file_path = os.path.join(Config.OUT_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Could not delete {file_path}. Reason: {e}")
    
    # Recreate chunks directory after cleanup
    os.makedirs(Config.CHUNKS_DIR, exist_ok=True)

def synthesize_audio_chunks(chunks: List[str], voice_wav_path: Optional[Path], resume: bool = True) -> None:
    """Synthesize audio chunks using XTTS with enhanced error handling."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading XTTS model on {device}...")
    
    # Initialize TTS model with enhanced error handling
    try:
        # Force CPU mode for compatibility
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        tts = TTS(Config.MODEL_ID, gpu=False, progress_bar=True)
        logging.info(f"TTS model loaded successfully")
        
        # Test model initialization
        if hasattr(tts.synthesizer, 'tts_model'):
            logging.info("TTS model ready for synthesis")
        else:
            logging.warning("TTS model may not be properly initialized")
            
    except Exception as e:
        logging.error(f"Failed to initialize TTS model: {e}")
        logging.info("Trying alternative initialization...")
        try:
            # Alternative initialization
            import warnings
            warnings.filterwarnings("ignore")
            tts = TTS(model_name=Config.MODEL_ID, progress_bar=False, gpu=False)
        except Exception as e2:
            logging.error(f"Alternative initialization also failed: {e2}")
            raise RuntimeError(f"Could not initialize TTS model. Original error: {e}")
    
    # Process chunks with enhanced error handling
    total = len(chunks)
    if Config.MAX_CHUNKS_TO_SYNTH and Config.MAX_CHUNKS_TO_SYNTH > 0:
        total = min(total, Config.MAX_CHUNKS_TO_SYNTH)
        logging.info(f"Limiting synthesis to first {total} chunks")

    for i, chunk_text in enumerate(tqdm(chunks[:total], desc="Synthesizing chunks")):
        chunk_wav_path = Config.CHUNKS_DIR / f"chunk_{i:03d}.wav"
        
        # Skip if file exists and resume is enabled
        if resume and chunk_wav_path.exists():
            logging.debug(f"Skipping existing chunk {i}")
            continue
        
        try:
            # Save chunk text for debugging
            (Config.CHUNKS_DIR / f"chunk_{i:03d}.txt").write_text(chunk_text, encoding="utf-8")
            
            # Clean up text for TTS
            clean_text = re.sub(r'[^\w\s.,!?;:-]', '', chunk_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 10:  # Skip very short chunks
                logging.warning(f"Skipping very short chunk {i}: '{clean_text[:50]}...'")
                # Create short silence
                silence = AudioSegment.silent(duration=500)  # 0.5 second
                silence.export(chunk_wav_path, format="wav")
                continue
            
            logging.debug(f"Synthesizing chunk {i}: '{clean_text[:50]}...'")
            
            # Use the TTS API with proper error handling
            try:
                # Determine synthesis params based on model type
                kwargs = {"text": clean_text}
                # XTTS voice cloning
                if "xtts" in Config.MODEL_ID and Config.REQUIRE_VOICE and voice_wav_path and voice_wav_path.exists():
                    kwargs.update({
                        "speaker_wav": str(voice_wav_path),
                        "language": Config.LANGUAGE or "en",
                    })
                else:
                    # Multi-speaker models expect a speaker id
                    if getattr(tts, "speakers", None) and (Config.SPEAKER or "") in getattr(tts, "speakers", []):
                        kwargs.update({"speaker": Config.SPEAKER})
                    elif getattr(tts, "speakers", None):
                        # pick first available as fallback
                        kwargs.update({"speaker": tts.speakers[0]})
                    # language may be required by some multilingual models
                    if getattr(tts, "languages", None) and Config.LANGUAGE in tts.languages:
                        kwargs.update({"language": Config.LANGUAGE})

                wav = tts.tts(**kwargs)
                
                # Save using TTS's built-in method
                if hasattr(tts, 'save_wav'):
                    tts.save_wav(wav, str(chunk_wav_path))
                elif hasattr(tts.synthesizer, 'save_wav'):
                    tts.synthesizer.save_wav(wav, str(chunk_wav_path))
                else:
                    # Fallback: save as numpy array
                    import soundfile as sf
                    sf.write(str(chunk_wav_path), wav, 22050)
                    
            except Exception as synth_error:
                logging.error(f"Synthesis failed for chunk {i}: {synth_error}")
                # Create silence as fallback
                silence = AudioSegment.silent(duration=2000)  # 2 seconds
                silence.export(chunk_wav_path, format="wav")
                
        except Exception as e:
            logging.error(f"Failed to process chunk {i}: {e}")
            # Create a short silence as fallback
            try:
                silence = AudioSegment.silent(duration=1000)  # 1 second
                silence.export(chunk_wav_path, format="wav")
            except Exception as silence_error:
                logging.error(f"Could not even create silence for chunk {i}: {silence_error}")
                continue

def combine_audio_chunks() -> None:
    """Combine all audio chunks into final audiobook files."""
    logging.info("Combining audio chunks...")
    
    chunk_files = sorted(Config.CHUNKS_DIR.glob("chunk_*.wav"))
    if not chunk_files:
        raise RuntimeError("No audio chunks found to combine")
    
    # Combine chunks with small pauses
    combined_audio = AudioSegment.empty()
    pause = AudioSegment.silent(duration=int(Config.PAUSE_BETWEEN_CHUNKS * 1000))
    
    successful_chunks = 0
    for chunk_file in tqdm(chunk_files, desc="Combining chunks"):
        try:
            chunk_audio = AudioSegment.from_wav(chunk_file)
            if len(chunk_audio) > 0:  # Only add non-empty chunks
                combined_audio += chunk_audio + pause
                successful_chunks += 1
        except Exception as e:
            logging.error(f"Failed to load chunk {chunk_file}: {e}")
            # Add silence instead
            combined_audio += AudioSegment.silent(duration=1000) + pause
            continue
    
    logging.info(f"Successfully combined {successful_chunks}/{len(chunk_files)} chunks")
    
    # Remove final pause
    if len(combined_audio) > len(pause):
        combined_audio = combined_audio[:-len(pause)]
    
    if len(combined_audio) == 0:
        raise RuntimeError("No audio content was generated")
    
    # Export both WAV and MP3
    logging.info(f"Exporting WAV: {Config.FINAL_WAV}")
    combined_audio.export(Config.FINAL_WAV, format="wav")
    
    logging.info(f"Exporting MP3: {Config.FINAL_MP3}")
    combined_audio.export(Config.FINAL_MP3, format="mp3", bitrate=Config.MP3_BITRATE)
    
    duration_minutes = len(combined_audio) / (1000 * 60)
    logging.info(f"Audiobook complete: {duration_minutes:.1f} minutes")

def cleanup_chunks(keep_chunks: bool = False) -> None:
    """Clean up intermediate files."""
    if not keep_chunks and Config.CHUNKS_DIR.exists():
        logging.info("Cleaning up chunk files...")
        shutil.rmtree(Config.CHUNKS_DIR)

def main(
    resume: bool = True,
    keep_chunks: bool = False,
    model_id: Optional[str] = None,
    speaker: Optional[str] = None,
    require_voice: Optional[bool] = None,
    language: Optional[str] = None,
    limit_chunks: Optional[int] = None,
) -> None:
    """Main pipeline function."""
    try:
        logging.info("Starting XTTS audiobook generation pipeline...")

        # Print system info
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")

        # Apply overrides from arguments
        if model_id:
            Config.MODEL_ID = model_id
        if speaker is not None:
            Config.SPEAKER = speaker
        if require_voice is not None:
            Config.REQUIRE_VOICE = require_voice
        if language:
            Config.LANGUAGE = language
        if limit_chunks is not None:
            Config.MAX_CHUNKS_TO_SYNTH = max(0, int(limit_chunks))

        # Validate inputs
        validate_inputs()

        # Setup directories
        setup_directories()

        # Extract text
        text = extract_text_from_pdf(Config.INPUT_PDF)
        if len(text) < 100:
            raise ValueError("Extracted text too short - check PDF file")

        # Save extracted text
        Config.TEXT_PATH.write_text(text, encoding="utf-8")

        # Create chunks
        chunks = smart_chunk_text(text)
        logging.info(f"Created {len(chunks)} text chunks")

        if len(chunks) == 0:
            raise ValueError("No valid text chunks created")

        # Show sample chunks for debugging
        logging.info("Sample chunks:")
        for i in range(min(3, len(chunks))):
            logging.info(f"  Chunk {i}: '{chunks[i][:100]}...'")

        # Synthesize audio
        synthesize_audio_chunks(chunks, Config.VOICE_WAV if Config.REQUIRE_VOICE else None, resume=resume)

        # Combine final audio
        combine_audio_chunks()

        # Cleanup
        cleanup_chunks(keep_chunks)

        logging.info("✅ Audiobook generation completed successfully!")
        logging.info(f"Output files: {Config.FINAL_WAV}, {Config.FINAL_MP3}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDF to audiobook using XTTS voice cloning")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing chunks")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep intermediate chunk files")
    parser.add_argument("--pdf", type=str, help="Path to PDF file")
    parser.add_argument("--voice", type=str, help="Path to voice sample file")
    parser.add_argument("--model-id", type=str, help="Override TTS model id (e.g. tts_models/en/vctk/vits)")
    parser.add_argument("--speaker", type=str, help="Speaker id for multi-speaker models (e.g. p225)")
    parser.add_argument("--no-voice", action="store_true", help="Do not require or use a voice sample (for non-cloning models)")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g. en, es)")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Limit number of chunks to synthesize (0 = all)")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.pdf:
        Config.INPUT_PDF = Path(args.pdf)
    if args.voice:
        Config.VOICE_WAV = Path(args.voice)
    
    if args.model_id:
        Config.MODEL_ID = args.model_id
    if args.speaker:
        Config.SPEAKER = args.speaker
    if args.no_voice:
        Config.REQUIRE_VOICE = False
    if args.language:
        Config.LANGUAGE = args.language
    if args.limit_chunks is not None:
        Config.MAX_CHUNKS_TO_SYNTH = max(0, int(args.limit_chunks))

    main(
        resume=not args.no_resume,
        keep_chunks=args.keep_chunks,
        model_id=Config.MODEL_ID,
        speaker=Config.SPEAKER,
        require_voice=Config.REQUIRE_VOICE,
        language=Config.LANGUAGE,
        limit_chunks=Config.MAX_CHUNKS_TO_SYNTH,
    )
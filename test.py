"""
Audio -> VAD -> STT -> LLM -> TTS Pipeline (Local CPU-Compatible)

This script implements a complete conversation processing pipeline:
1. Audio Input: Load audio from file or URL
2. VAD (Voice Activity Detection): Detect speech segments
3. STT (Speech-to-Text): Transcribe using Whisper
4. LLM: Process transcribed text with language model
5. TTS (Text-to-Speech): Convert LLM response to audio

Simplified for local development without GPU, CUDA, or MongoDB dependencies.
"""

import os
import io
import json
import wave
import shutil
import numpy as np
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import logging

from dotenv import load_dotenv
import librosa
import soundfile as sf
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI

# DeepFilterNet for complex ML-based noise detection
try:
    import torch
    from df.enhance import enhance, init_df
    DEEPFILTERNET_AVAILABLE = True
except ImportError:
    DEEPFILTERNET_AVAILABLE = False
    logger = logging.getLogger(__name__)  # Get logger early for warning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for FFmpeg availability (required by Whisper)
if shutil.which("ffmpeg") is None:
    logger.error("="*60)
    logger.error("FFmpeg NOT FOUND - Required for Whisper STT")
    logger.error("="*60)
    logger.error("Please install FFmpeg and add it to your system PATH:")
    logger.error("1. Download from: https://github.com/GyanD/codexffmpeg/releases")
    logger.error("2. Extract to C:\\ffmpeg")
    logger.error("3. Add C:\\ffmpeg\\bin to system PATH")
    logger.error("4. Restart terminal and verify with: ffmpeg -version")
    logger.error("="*60)
    raise RuntimeError("FFmpeg is required but not found in PATH")

# ==================== CONFIGURATION ====================

# Whisper configuration for CPU
WHISPER_MODEL_NAME = "base.en"  # Smaller model for CPU (base vs medium)
WHISPER_MODEL_DIR = "models"
DEVICE = "cpu"  # Force CPU usage

# Ensure model directory exists
os.makedirs(WHISPER_MODEL_NAME, exist_ok=True)

# Initialize Whisper model (loads once, reused for all calls)
logger.info(f"Loading Whisper model ({WHISPER_MODEL_NAME}) on {DEVICE}...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE, download_root=WHISPER_MODEL_DIR)
logger.info("Whisper model loaded successfully")

# Initialize LLM (Google Gemini)
GOOGLE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")  # Flash is faster on CPU

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_tokens=512,
    max_retries=3,
    timeout=30,
)
logger.info(f"LLM initialized: {GEMINI_MODEL}")

# ==================== STEP 1: AUDIO LOADING ====================

def load_audio_file(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy array with sample rate.
    
    Args:
        audio_path (str): Path to audio file (.wav, .mp3, .m4a, etc.)
    
    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
    """
    try:
        logger.info(f"Loading audio file: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # librosa handles multiple formats automatically
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")
        return audio_data, sample_rate
    
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        raise


def load_audio_from_url(url: str) -> Tuple[np.ndarray, int]:
    """
    Download and load audio from URL.
    
    Args:
        url (str): URL to audio file
    
    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
    """
    try:
        logger.info(f"Downloading audio from: {url}")
        import requests
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Load audio from bytes
        audio_bytes = io.BytesIO(response.content)
        audio_data, sample_rate = librosa.load(audio_bytes, sr=None)
        
        logger.info(f"Audio downloaded and loaded: {len(audio_data)} samples at {sample_rate} Hz")
        return audio_data, sample_rate
    
    except Exception as e:
        logger.error(f"Error downloading audio from URL: {e}")
        raise


# ==================== STEP 2: VAD (Voice Activity Detection) ====================

def simple_vad(audio_data: np.ndarray, sample_rate: int, threshold_db: float = -30) -> List[Tuple[float, float]]:
    """
    Simple Voice Activity Detection using energy-based approach.
    Returns list of (start_time, end_time) tuples for speech segments.
    
    Args:
        audio_data (np.ndarray): Audio samples
        sample_rate (int): Sample rate
        threshold_db (float): Energy threshold in dB
    
    Returns:
        List[Tuple[float, float]]: List of (start_time, end_time) for speech segments
    """
    try:
        logger.info("Running Voice Activity Detection (VAD)...")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data)
        
        # Calculate frame-by-frame energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        energy = np.array([
            np.sum(audio_data[i:i+frame_length]**2)
            for i in range(0, len(audio_data), hop_length)
        ])
        
        # Convert to dB
        energy_db = 10 * np.log10(energy + 1e-10)
        
        # Threshold
        speech_frames = energy_db > threshold_db
        
        # Find speech segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = start_frame * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                segments.append((start_time, end_time))
                in_speech = False
        
        # Handle case where audio ends while in speech
        if in_speech:
            start_time = start_frame * hop_length / sample_rate
            end_time = len(audio_data) / sample_rate
            segments.append((start_time, end_time))
        
        logger.info(f"VAD detected {len(segments)} speech segment(s)")
        for i, (start, end) in enumerate(segments):
            logger.info(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s")
        
        return segments
    
    except Exception as e:
        logger.error(f"Error in VAD: {e}")
        raise

# ==================== STEP 2.5: Noise  Detection ====================
def simple_noise_detector(audio_data: np.ndarray, sample_rate: int, speech_segments: List[Tuple[float, float]]) -> List[Dict]:
    """
    Analyzes non-speech segments from VAD to classify them as 'Noise' or 'Silence'.
    This uses two key features common in noise analysis: RMS Energy and Zero-Crossing Rate (ZCR).

    Args:
        audio_data (np.ndarray): Full audio samples (from librosa.load).
        sample_rate (int): Sample rate of the audio.
        speech_segments (List[Tuple]): List of (start_time, end_time) for *speech* segments (from VAD).

    Returns:
        List[Dict]: List of identified *non-speech* segments with a noise classification and metrics.
    """
    logger.info("Running Simple AI Noise Detection on non-speech segments...")
    
    # --- Configuration (Adjust these thresholds based on your call center data!) ---
    # Noise usually has high, steady ZCR and medium energy. Silence has low ZCR and low energy.
    ZCR_THRESHOLD = 0.05   
    RMS_ENERGY_THRESHOLD = 0.005 # Threshold for distinguishing silence from ambient noise (in normalized [-1, 1] audio)
    
    # --- Identify Non-Speech Segments ---
    full_duration = len(audio_data) / sample_rate
    non_speech_segments = []
    last_end = 0.0
    
    # 1. Calculate non-speech intervals (the gaps between speech)
    for start, end in speech_segments:
        # Check if gap between end of last speech and start of current speech is long enough (> 100ms)
        if start > last_end + 0.1:
            non_speech_segments.append((last_end, start))
        last_end = end
    
    # 2. Check for non-speech at the end of the file
    if full_duration > last_end + 0.1:
        non_speech_segments.append((last_end, full_duration))

    noise_analysis_results = []
    
    # --- Analyze Each Non-Speech Segment ---
    for start_time, end_time in non_speech_segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]
        
        # Skip if segment is too short for meaningful analysis
        if len(segment_audio) < sample_rate * 0.1: continue

        # Calculate features using librosa
        # RMS Energy: Average loudness/power of the segment
        rms_energy = librosa.feature.rms(y=segment_audio).mean()
        # ZCR: Rate of sign changes (indicates frequency content - higher for high-frequency noise like hiss, lower for hum)
        zcr = librosa.feature.zero_crossing_rate(y=segment_audio, frame_length=512, hop_length=256).mean()
        
        # --- Classification Logic ---
        classification = "Silence (Background)"
        
        if rms_energy > RMS_ENERGY_THRESHOLD:
            if zcr > ZCR_THRESHOLD * 2: # High Energy and High ZCR (e.g., rustling, static, sharp non-speech noise)
                 classification = "Noise (High Frequency/Energy)"
            elif zcr > ZCR_THRESHOLD: # Medium Energy and Medium ZCR (e.g., constant hum, low background chatter)
                classification = "Background Noise (Ambient Hum/Chatter)"
            else: # High Energy and Low ZCR (e.g., music, low-frequency constant noise)
                classification = "Noise (Low Frequency/Steady)"

        noise_analysis_results.append({
            "start_time": f"{start_time:.2f}s",
            "end_time": f"{end_time:.2f}s",
            "duration": f"{end_time - start_time:.2f}s",
            "classification": classification,
            "metrics": {"rms_energy": f"{rms_energy:.4f}", "zero_crossing_rate": f"{zcr:.4f}"}
        })
        
    logger.info(f"Noise Detector analyzed {len(non_speech_segments)} non-speech segments.")
    return noise_analysis_results

# ==================== STEP 2.5b: Complex ML Noise Detection (DeepFilterNet) ====================

# Global DeepFilterNet model (loaded once for efficiency)
DF_MODEL = None
DF_STATE = None
DF_SR = None

def load_deepfilternet_model():
    """
    Load DeepFilterNet model once globally for efficient reuse.
    """
    global DF_MODEL, DF_STATE, DF_SR
    
    if not DEEPFILTERNET_AVAILABLE:
        logger.warning("DeepFilterNet not available. Install with: pip install deepfilternet torch torchaudio")
        return False
    
    if DF_MODEL is None:
        try:
            logger.info("Loading DeepFilterNet model for complex noise detection...")
            DF_MODEL, DF_STATE, _ = init_df()  # Third return is model name, not sample rate
            DF_SR = DF_STATE.sr()  # Get sample rate from state object (48000Hz)
            logger.info(f"DeepFilterNet3 model loaded successfully (expects {DF_SR}Hz audio)")
            return True
        except Exception as e:
            logger.error(f"Failed to load DeepFilterNet model: {e}")
            return False
    return True


def complex_noise_detector(audio_data: np.ndarray, sample_rate: int, 
                           speech_segments: List[Tuple[float, float]],
                           use_deepfilternet: bool = True,
                           save_enhanced_audio: bool = True) -> Dict:
    """
    Complex ML-based noise detection using DeepFilterNet.
    Analyzes non-speech segments and measures noise reduction ratio.
    Optionally saves the full enhanced (noise-reduced) audio for comparison.
    
    Args:
        audio_data (np.ndarray): Full audio samples.
        sample_rate (int): Sample rate of the audio.
        speech_segments (List[Tuple]): List of (start_time, end_time) for speech segments from VAD.
        use_deepfilternet (bool): Whether to use DeepFilterNet for enhancement.
        save_enhanced_audio (bool): Whether to save the full enhanced audio file.
    
    Returns:
        Dict: Complex noise detection results including enhanced audio metrics.
    """
    logger.info("Running Complex ML Noise Detection (DeepFilterNet)...")
    
    # --- Identify Non-Speech Segments (same logic as simple detector) ---
    full_duration = len(audio_data) / sample_rate
    non_speech_segments = []
    last_end = 0.0
    
    for start, end in speech_segments:
        if start > last_end + 0.1:
            non_speech_segments.append((last_end, start))
        last_end = end
    
    if full_duration > last_end + 0.1:
        non_speech_segments.append((last_end, full_duration))
    
    # --- Check if DeepFilterNet is available and loaded ---
    df_available = False
    if use_deepfilternet and DEEPFILTERNET_AVAILABLE:
        df_available = load_deepfilternet_model()
    
    complex_results = []
    total_flagged = 0
    total_analyzed = 0
    enhanced_audio_path = None
    
    # Resample audio for DeepFilterNet if needed (requires 48kHz)
    if df_available and DF_SR and sample_rate != DF_SR:
        logger.info(f"Resampling audio from {sample_rate}Hz to {DF_SR}Hz for DeepFilterNet...")
        audio_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=DF_SR)
        resample_ratio = DF_SR / sample_rate
    else:
        audio_resampled = audio_data
        resample_ratio = 1.0
    
    # --- FULL AUDIO ENHANCEMENT: Process entire audio and save for comparison ---
    if df_available and save_enhanced_audio and DF_MODEL is not None:
        try:
            logger.info("Enhancing FULL audio with DeepFilterNet for comparison...")
            full_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0)
            full_enhanced = enhance(DF_MODEL, DF_STATE, full_tensor)
            
            if isinstance(full_enhanced, torch.Tensor):
                full_enhanced_np = full_enhanced.squeeze().numpy()
            else:
                full_enhanced_np = np.array(full_enhanced).squeeze()
            
            # Save enhanced audio at 48kHz (DeepFilterNet output rate)
            enhanced_audio_path = "enhanced_audio_deepfilternet.wav"
            sf.write(enhanced_audio_path, full_enhanced_np, DF_SR)
            logger.info(f"Enhanced audio saved to: {enhanced_audio_path} ({DF_SR}Hz)")
            
            # Also save a version resampled back to original rate for easier comparison
            enhanced_original_sr = librosa.resample(full_enhanced_np, orig_sr=DF_SR, target_sr=sample_rate)
            enhanced_audio_path_original_sr = "enhanced_audio_original_sr.wav"
            sf.write(enhanced_audio_path_original_sr, enhanced_original_sr, sample_rate)
            logger.info(f"Enhanced audio (original sample rate) saved to: {enhanced_audio_path_original_sr} ({sample_rate}Hz)")
            
        except Exception as e:
            logger.error(f"Failed to enhance full audio: {e}")
            enhanced_audio_path = None
    
    # --- Analyze Each Non-Speech Segment with ML Model ---
    for start_time, end_time in non_speech_segments:
        # Calculate sample indices (use resampled audio if DeepFilterNet)
        if df_available:
            start_sample = int(start_time * DF_SR)
            end_sample = int(end_time * DF_SR)
            segment_audio = audio_resampled[start_sample:end_sample]
        else:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
        
        # Skip very short segments
        min_samples = int(0.1 * (DF_SR if df_available else sample_rate))
        if len(segment_audio) < min_samples:
            continue
        
        total_analyzed += 1
        
        # Calculate original RMS energy
        original_rms = np.sqrt(np.mean(segment_audio**2))
        
        if df_available and DF_MODEL is not None:
            try:
                # Run DeepFilterNet enhancement
                # DeepFilterNet expects audio as torch tensor
                segment_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0)
                enhanced_audio = enhance(DF_MODEL, DF_STATE, segment_tensor)
                
                # Convert back to numpy
                if isinstance(enhanced_audio, torch.Tensor):
                    enhanced_np = enhanced_audio.squeeze().numpy()
                else:
                    enhanced_np = np.array(enhanced_audio).squeeze()
                
                # Calculate enhanced RMS
                enhanced_rms = np.sqrt(np.mean(enhanced_np**2))
                
                # Noise reduction ratio: how much energy was removed
                if original_rms > 1e-10:
                    noise_reduction_ratio = 1.0 - (enhanced_rms / original_rms)
                else:
                    noise_reduction_ratio = 0.0
                
                detection_method = "DeepFilterNet (DNN)"
                
            except Exception as e:
                logger.warning(f"DeepFilterNet processing failed for segment {start_time:.2f}s-{end_time:.2f}s: {e}")
                # Fallback to simple metrics
                zcr = librosa.feature.zero_crossing_rate(y=segment_audio).mean()
                noise_reduction_ratio = min(zcr * 2, 0.5)  # Estimate based on ZCR
                detection_method = "Fallback (ZCR-based)"
        else:
            # Fallback: Use spectral features for estimation
            zcr = librosa.feature.zero_crossing_rate(y=segment_audio).mean()
            spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sample_rate).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sample_rate).mean()
            
            # Estimate noise based on spectral features
            # High spectral centroid + high ZCR = likely noise
            normalized_centroid = spectral_centroid / (sample_rate / 2)
            noise_estimate = (zcr * 0.5 + normalized_centroid * 0.5)
            noise_reduction_ratio = min(noise_estimate, 1.0)
            detection_method = "Spectral Analysis (No DeepFilterNet)"
        
        # Classification based on noise reduction ratio
        ML_NOISE_THRESHOLD_HIGH = 0.25  # >25% noise removed = High noise
        ML_NOISE_THRESHOLD_MED = 0.15   # 15-25% = Medium noise
        ML_NOISE_THRESHOLD_LOW = 0.05   # 5-15% = Low noise
        
        if noise_reduction_ratio > ML_NOISE_THRESHOLD_HIGH:
            classification = "High Noise (Significant non-speech content)"
            is_noisy = True
        elif noise_reduction_ratio > ML_NOISE_THRESHOLD_MED:
            classification = "Medium Noise (Moderate background)"
            is_noisy = True
        elif noise_reduction_ratio > ML_NOISE_THRESHOLD_LOW:
            classification = "Low Noise (Minor background)"
            is_noisy = False
        else:
            classification = "Clean Silence"
            is_noisy = False
        
        if is_noisy:
            total_flagged += 1
        
        complex_results.append({
            "start_time": f"{start_time:.2f}s",
            "end_time": f"{end_time:.2f}s",
            "duration": f"{end_time - start_time:.2f}s",
            "detection_method": detection_method,
            "original_rms": f"{original_rms:.6f}",
            "noise_reduction_ratio": f"{noise_reduction_ratio:.2%}",
            "is_flagged_as_noise": is_noisy,
            "classification": classification
        })
    
    logger.info(f"Complex Noise Detector analyzed {total_analyzed} non-speech segments.")
    logger.info(f"Complex Detection: {total_flagged} noisy segments flagged (ML-based).")
    
    return {
        "status": "completed",
        "model_used": "DeepFilterNet" if df_available else "Spectral Analysis (Fallback)",
        "total_segments_analyzed": total_analyzed,
        "noisy_segments_flagged": total_flagged,
        "enhanced_audio_path": enhanced_audio_path,
        "enhanced_audio_original_sr_path": "enhanced_audio_original_sr.wav" if enhanced_audio_path else None,
        "analysis_results": complex_results
    }

# ==================== STEP 3: STT (Speech-to-Text) ====================

def transcribe_audio(audio_path: str) -> Dict[str, any]:
    """
    Transcribe audio file using Whisper (local CPU-compatible).
    
    Args:
        audio_path (str): Path to audio file
    
    Returns:
        Dict: Transcription result with text and word-level timestamps
    """
    try:
        logger.info("Starting Speech-to-Text transcription...")
        
        result = whisper_model.transcribe(audio_path, fp16=False, language="en")
        
        logger.info(f"Transcription complete: {len(result['text'])} characters")
        logger.info(f"Full transcript: {result['text'][:200]}...")
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "en")
        }
    
    except Exception as e:
        logger.error(f"Error in STT: {e}")
        raise


def transcribe_speech_segments(audio_data: np.ndarray, sample_rate: int, 
                               segments: List[Tuple[float, float]]) -> List[Dict]:
    """
    Transcribe individual speech segments detected by VAD.
    
    Args:
        audio_data (np.ndarray): Full audio data
        sample_rate (int): Sample rate
        segments (List[Tuple]): List of (start_time, end_time) tuples
    
    Returns:
        List[Dict]: Transcriptions for each segment
    """
    try:
        logger.info("Transcribing individual segments...")
        
        segment_results = []
        
        for i, (start_time, end_time) in enumerate(segments):
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Save temporary segment file
            temp_path = f"temp_segment_{i}.wav"
            sf.write(temp_path, segment_audio, sample_rate)
            
            # Transcribe segment
            result = whisper_model.transcribe(temp_path, fp16=False, language="en")
            
            # Extract confidence safely
            segments_data = result.get("segments", [])
            confidence = 0
            if segments_data and len(segments_data) > 0:
                confidence = segments_data[0].get("confidence", 0)
            
            segment_results.append({
                "segment_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "text": result["text"],
                "confidence": confidence
            })
            
            # Cleanup
            os.remove(temp_path)
            
            logger.info(f"  Segment {i+1}: {result['text'][:100]}...")
        
        return segment_results
    
    except Exception as e:
        logger.error(f"Error transcribing segments: {e}")
        raise


# ==================== STEP 4: LLM (Language Model Processing) ====================

def process_with_llm(transcript: str, system_prompt: Optional[str] = None) -> str:
    """
    Process transcribed text through Language Model for analysis/response.
    
    Args:
        transcript (str): Transcribed text to process
        system_prompt (str, optional): Custom system prompt. If None, uses default.
    
    Returns:
        str: LLM response
    """
    try:
        logger.info("Processing transcript with LLM...")
        
        if not system_prompt:
            system_prompt = """You are a helpful assistant analyzing customer service calls. 
Provide a concise summary (max 3 sentences) of the call transcript with key points and sentiment."""
        
        messages = [
            ("system", system_prompt),
            ("human", f"Analyze this transcript:\n\n{transcript}")
        ]
        
        response = llm.invoke(messages)
        llm_response = response.content
        
        logger.info(f"LLM response: {llm_response[:200]}...")
        
        return llm_response
    
    except Exception as e:
        logger.error(f"Error processing with LLM: {e}")
        raise


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response if present.
    
    Args:
        response (str): LLM response text
    
    Returns:
        Dict or None: Extracted JSON if found
    """
    try:
        import re
        
        # Try to find JSON in response
        json_pattern = r'\{[\s\S]*\}|\[[\s\S]*\]'
        match = re.search(json_pattern, response)
        
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        return None
    
    except Exception as e:
        logger.warning(f"Could not extract JSON from response: {e}")
        return None


# ==================== STEP 5: TTS (Text-to-Speech) ====================

def text_to_speech(text: str, output_path: str = "response_audio.wav") -> str:
    """
    Convert text to speech using gTTS (Google Text-to-Speech).
    Falls back to simple beep if gTTS unavailable.
    
    Args:
        text (str): Text to convert to speech
        output_path (str): Output audio file path
    
    Returns:
        str: Path to generated audio file
    """
    try:
        logger.info("Converting text to speech...")
        
        try:
            from gtts import gTTS
            
            # Create gTTS object and save
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            
            logger.info(f"TTS audio saved to: {output_path}")
            return output_path
        
        except ImportError:
            logger.warning("gTTS not installed. Install with: pip install gtts")
            logger.info("Generating placeholder audio...")
            
            # Create a simple placeholder WAV file
            sample_rate = 22050
            duration = 2
            frequency = 440
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            sf.write(output_path, audio, sample_rate)
            logger.info(f"Placeholder audio created at: {output_path}")
            
            return output_path
    
    except Exception as e:
        logger.error(f"Error in TTS: {e}")
        raise


# ==================== MAIN PIPELINE ====================

def process_audio_pipeline(audio_input: str, use_vad: bool = True, 
                          custom_llm_prompt: Optional[str] = None,
                          skip_llm_processing: bool = False) -> Dict:
    """
    Complete Audio -> VAD -> STT -> LLM -> TTS pipeline.
    
    Args:
        audio_input (str): Path to audio file or URL
        use_vad (bool): Whether to use VAD for segment detection
        custom_llm_prompt (str, optional): Custom system prompt for LLM
        skip_llm_processing (bool): Whether to skip the LLM and TTS steps.
    
    Returns:
        Dict: Complete pipeline result with all stages
    """
    
    result = {
        "status": "processing",
        "timestamp": datetime.now().isoformat(),
        "audio_input": audio_input,
        "steps": {}
    }
    
    try:
        # STEP 1: Load Audio
        logger.info("\n" + "="*50)
        logger.info("STEP 1: LOADING AUDIO")
        logger.info("="*50)
        
        if audio_input.startswith("http"):
            audio_data, sample_rate = load_audio_from_url(audio_input)
        else:
            audio_data, sample_rate = load_audio_file(audio_input)
        
        result["steps"]["audio_loading"] = {
            "status": "completed",
            "sample_rate": sample_rate,
            "duration_seconds": len(audio_data) / sample_rate
        }
        
        # STEP 2: VAD
        logger.info("\n" + "="*50)
        logger.info("STEP 2: VOICE ACTIVITY DETECTION (VAD)")
        logger.info("="*50)
        
        if use_vad:
            vad_segments = simple_vad(audio_data, sample_rate)
            result["steps"]["vad"] = {
                "status": "completed",
                "segments_detected": len(vad_segments),
                "segments": vad_segments
            }
        else:
            vad_segments = [(0, len(audio_data) / sample_rate)]
            result["steps"]["vad"] = {
                "status": "skipped",
                "message": "VAD disabled, processing entire audio"
            }
        # step 2.5: Noise Detection on non-speech segments
        logger.info("\n" + "="*50)
        logger.info("STEP 2.5: AI NOISE DETECTION")
        logger.info("="*50)

        # The noise detection only runs if VAD was successful
        if use_vad and "vad" in result["steps"] and result["steps"]["vad"]["status"] == "completed":
            try:
                vad_segments = result["steps"]["vad"]["segments"]
                
                # Run the new noise detection function on the full audio data
                noise_segments_analysis = simple_noise_detector(audio_data, sample_rate, vad_segments)
                
                result["steps"]["noise_detection"] = {
                    "status": "completed",
                    "total_noise_segments_found": len([d for d in noise_segments_analysis if 'Noise' in d['classification']]),
                    "analysis_results": noise_segments_analysis
                }
                logger.info(f"Noise Detection: {result['steps']['noise_detection']['total_noise_segments_found']} noisy segments flagged.")
                
            except Exception as e:
                logger.error(f"Noise Detection error: {e}")
                result["steps"]["noise_detection"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            result["steps"]["noise_detection"] = {
                "status": "skipped",
                "message": "VAD was not used or failed, skipping noise analysis on non-speech segments."
            }
        
        # STEP 2.5b: COMPLEX ML NOISE DETECTION (DeepFilterNet)
        logger.info("\n" + "="*50)
        logger.info("STEP 2.5b: COMPLEX ML NOISE DETECTION (DeepFilterNet)")
        logger.info("="*50)
        
        if use_vad and "vad" in result["steps"] and result["steps"]["vad"]["status"] == "completed":
            try:
                vad_segments = result["steps"]["vad"]["segments"]
                
                # Run complex ML-based noise detection
                complex_noise_result = complex_noise_detector(
                    audio_data, sample_rate, vad_segments, 
                    use_deepfilternet=DEEPFILTERNET_AVAILABLE
                )
                
                result["steps"]["noise_detection_complex"] = complex_noise_result
                
            except Exception as e:
                logger.error(f"Complex Noise Detection error: {e}")
                result["steps"]["noise_detection_complex"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            result["steps"]["noise_detection_complex"] = {
                "status": "skipped",
                "message": "VAD was not used or failed, skipping complex noise analysis."
            }

        # STEP 3: STT (Speech-to-Text)
        logger.info("\n" + "="*50)
        logger.info("STEP 3: SPEECH-TO-TEXT (STT)")
        logger.info("="*50)
        
        # Resample to 16kHz if needed (Whisper's preferred rate)
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Save temp audio file for Whisper
        temp_audio_path = "temp_input_audio.wav"
        sf.write(temp_audio_path, audio_data, sample_rate)
        
        stt_result = transcribe_audio(temp_audio_path)
        
        # Segment-level transcription is disabled by default (very slow for many segments)
        # Enable only if you need per-segment transcripts and have few segments
        segment_transcriptions = []
        # Uncomment below to enable segment-level transcription:
        # if use_vad and len(vad_segments) > 1 and len(vad_segments) < 20:
        #     logger.info(f"Transcribing {len(vad_segments)} segments individually...")
        #     segment_transcriptions = transcribe_speech_segments(audio_data, sample_rate, vad_segments)
        
        full_transcript = stt_result["text"]
        
        result["steps"]["stt"] = {
            "status": "completed",
            "full_transcript": full_transcript,
            "segment_transcriptions": segment_transcriptions,
            "language": stt_result.get("language", "en")
        }
        
        # Cleanup temp file
        os.remove(temp_audio_path)
        
        # STEP 4: LLM Processing
        logger.info("\n" + "="*50)
        logger.info("STEP 4: LANGUAGE MODEL (LLM) PROCESSING")
        logger.info("="*50)
        
        if not skip_llm_processing:
            llm_response = process_with_llm(full_transcript, custom_llm_prompt)
            
            # Try to extract JSON if present
            json_response = extract_json_from_response(llm_response)
            
            result["steps"]["llm"] = {
                "status": "completed",
                "response": llm_response,
                "json_extracted": json_response is not None,
                "json_response": json_response
            }
        else:
            result["steps"]["llm"] = {
                "status": "skipped",
                "message": "LLM processing skipped by request."
            }

        
        # STEP 5: TTS (Text-to-Speech)
        logger.info("\n" + "="*50)
        logger.info("STEP 5: TEXT-TO-SPEECH (TTS)")
        logger.info("="*50)
        
        if not skip_llm_processing:
            # Generate TTS for LLM response
            tts_output_path = text_to_speech(llm_response[:500])  # Limit text for TTS
            
            result["steps"]["tts"] = {
                "status": "completed",
                "audio_output_path": tts_output_path,
                "text_used": llm_response[:500]
            }
        else:
            result["steps"]["tts"] = {
                "status": "skipped",
                "message": "TTS skipped as LLM processing was skipped."
            }

        
        result["status"] = "completed"
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        return result
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        result["status"] = "error"
        result["error"] = str(e)
        return result


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    
    # Example 1: Process local audio file
    LOCAL_AUDIO_PATH = "test_aud.mp3"  # Replace with actual path
    
    if os.path.exists(LOCAL_AUDIO_PATH):
        logger.info(f"Processing local audio: {LOCAL_AUDIO_PATH}")
        
        pipeline_result = process_audio_pipeline(
            audio_input=LOCAL_AUDIO_PATH,
            use_vad=True,
            custom_llm_prompt="Provide a brief summary of this conversation."
        )
        
        # Save results to JSON
        with open("pipeline_result.json", "w") as f:
            json.dump(pipeline_result, f, indent=2)
        
        print("\n" + "="*50)
        print("PIPELINE RESULT")
        print("="*50)
        print(json.dumps(pipeline_result, indent=2))
    
    else:
        logger.warning(f"Sample audio file not found: {LOCAL_AUDIO_PATH}")
        logger.info("To test the pipeline, provide a valid audio file path.")
        logger.info("\nExample usage:")
        print("""
        result = process_audio_pipeline(
              audio_input="https://callhippo-media.s3.amazonaws.com/callrecordings_hippa/6c3fede2-26b8-42b2-acf9-0355d15e80bf.mp3",
              use_vad=True
        )

        """)

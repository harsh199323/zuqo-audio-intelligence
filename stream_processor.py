"""
Real-Time Streaming Audio Analysis Pipeline

This module implements the STREAMING pipeline for real-time conversation intelligence:
- Receives audio chunks via WebSocket or direct input
- Streams audio to Google Cloud Speech-to-Text for low-latency transcription
- Performs incremental LLM analysis on each finalized transcript segment
- Publishes real-time metrics, alerts, and state updates

Use Cases:
- Live agent coaching during calls
- Real-time sentiment monitoring
- Script adherence tracking
- Immediate escalation alerts

Comparison with Batch Pipeline (test.py):
- Streaming: Fast, incremental, lower accuracy, real-time alerts
- Batch: Slow, complete, higher accuracy, post-call analysis
"""

import os
import sys
import json
import asyncio
import logging
import queue
import threading
from datetime import datetime
from typing import Optional, Callable, AsyncGenerator, Dict, Any

import numpy as np
from dotenv import load_dotenv

# Import shared utilities
from utils import (
    StreamingAnalysisState,
    MetricsPublisher,
    get_streaming_llm,
    process_streaming_llm_chunk,
    extract_json_from_text,
    check_script_adherence,
    format_duration,
    reset_streaming_state,
    get_streaming_stats,
    local_sentiment_analysis,
    STREAMING_SAMPLE_RATE,
    STREAMING_STT_LANGUAGE,
    STREAMING_CHUNK_DURATION_MS,
    LLM_CALLS_PER_MINUTE,
    LLM_BATCH_TURNS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ==================== GOOGLE CLOUD STT CONFIGURATION ====================

# Check for Google Cloud Speech-to-Text availability
GOOGLE_CLOUD_STT_AVAILABLE = False
try:
    from google.cloud import speech_v1 as speech
    from google.cloud.speech_v1 import types
    GOOGLE_CLOUD_STT_AVAILABLE = True
    logger.info("Google Cloud Speech-to-Text SDK available")
except ImportError:
    logger.warning("Google Cloud Speech-to-Text not installed. "
                  "Install with: pip install google-cloud-speech")

# Alternative: Check for Whisper streaming (lower latency than file-based)
WHISPER_STREAMING_AVAILABLE = False
try:
    import importlib.util
    spec = importlib.util.find_spec("whisper")
    WHISPER_STREAMING_AVAILABLE = spec is not None
    if WHISPER_STREAMING_AVAILABLE:
        logger.info("Whisper STT available (will be loaded on demand)")
except Exception:
    WHISPER_STREAMING_AVAILABLE = False

# WebSocket support
WEBSOCKETS_AVAILABLE = False
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
    logger.info("WebSocket support available")
except ImportError:
    logger.warning("WebSocket support not available. Install with: pip install websockets")


# ==================== STREAMING STT PROVIDERS ====================

class StreamingSTTProvider:
    """Base class for streaming Speech-to-Text providers."""
    
    def __init__(self, sample_rate: int = STREAMING_SAMPLE_RATE, language: str = STREAMING_STT_LANGUAGE):
        self.sample_rate = sample_rate
        self.language = language
        self.is_active = False
    
    async def start(self):
        """Start the STT stream."""
        raise NotImplementedError
    
    async def send_audio(self, audio_chunk: bytes):
        """Send an audio chunk to the STT service."""
        raise NotImplementedError
    
    async def get_results(self) -> AsyncGenerator[Dict, None]:
        """Yield transcription results as they become available."""
        raise NotImplementedError
    
    async def stop(self):
        """Stop the STT stream."""
        raise NotImplementedError


class GoogleCloudSTT(StreamingSTTProvider):
    """
    Google Cloud Speech-to-Text streaming provider.
    
    Provides low-latency, real-time transcription with interim results.
    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable.
    """
    
    def __init__(self, sample_rate: int = STREAMING_SAMPLE_RATE, language: str = STREAMING_STT_LANGUAGE):
        super().__init__(sample_rate, language)
        
        if not GOOGLE_CLOUD_STT_AVAILABLE:
            raise ImportError("google-cloud-speech not installed")
        
        self.client = None
        self.streaming_config = None
        self.audio_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
    
    async def start(self):
        """Initialize the Google Cloud STT client and streaming config."""
        self.client = speech.SpeechClient()
        
        # Configure recognition
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            enable_automatic_punctuation=True,
            model="phone_call",  # Optimized for phone calls
            use_enhanced=True,   # Use enhanced model if available
        )
        
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,  # Get partial results for real-time display
            single_utterance=False  # Continue listening for multiple utterances
        )
        
        self.is_active = True
        logger.info("Google Cloud STT streaming started")
    
    async def send_audio(self, audio_chunk: bytes):
        """Queue audio chunk for streaming."""
        if self.is_active:
            await self.audio_queue.put(audio_chunk)
    
    def _generate_requests(self):
        """Generator for streaming requests."""
        # First request with config
        yield speech.StreamingRecognizeRequest(streaming_config=self.streaming_config)
        
        # Subsequent requests with audio
        while self.is_active:
            try:
                audio = self.audio_queue.get_nowait()
                yield speech.StreamingRecognizeRequest(audio_content=audio)
            except:
                break
    
    async def get_results(self) -> AsyncGenerator[Dict, None]:
        """Stream transcription results."""
        if not self.client or not self.is_active:
            return
        
        try:
            # Run in executor since the client is synchronous
            loop = asyncio.get_event_loop()
            
            def process_stream():
                requests = self._generate_requests()
                responses = self.client.streaming_recognize(requests)
                
                for response in responses:
                    for result in response.results:
                        yield {
                            "text": result.alternatives[0].transcript,
                            "is_final": result.is_final,
                            "confidence": result.alternatives[0].confidence if result.is_final else 0,
                            "timestamp": datetime.now().isoformat()
                        }
            
            for result in await loop.run_in_executor(None, lambda: list(process_stream())):
                yield result
                
        except Exception as e:
            logger.error(f"Error in Google Cloud STT: {e}")
    
    async def stop(self):
        """Stop the streaming session."""
        self.is_active = False
        logger.info("Google Cloud STT streaming stopped")


class LocalWhisperSTT(StreamingSTTProvider):
    """
    Local Whisper-based STT with pseudo-streaming.
    
    Since Whisper doesn't natively support streaming, this implementation
    buffers audio and transcribes in chunks (higher latency than cloud).
    
    Use for development/testing when Google Cloud isn't available.
    """
    
    def __init__(self, sample_rate: int = STREAMING_SAMPLE_RATE, 
                 language: str = "en",
                 model_name: str = "tiny",
                 chunk_duration: float = 1.5):  # Transcribe every N seconds
        super().__init__(sample_rate, language[:2])  # Whisper uses 2-letter codes
        
        if not WHISPER_STREAMING_AVAILABLE:
            raise ImportError("whisper not installed")
        
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.model = None
        self.audio_buffer = []
        self.buffer_samples = 0
        self.results_queue = queue.Queue()
    
    async def start(self):
        """Load Whisper model."""
        import whisper
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        self.audio_buffer = []
        self.buffer_samples = 0
        self.is_active = True
        logger.info("Local Whisper STT ready (pseudo-streaming mode)")
    
    async def send_audio(self, audio_chunk: bytes):
        """Buffer audio for batch transcription."""
        if not self.is_active:
            return
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.append(audio_array)
        self.buffer_samples += len(audio_array)
        
        # Check if we have enough audio to transcribe
        samples_needed = int(self.chunk_duration * self.sample_rate)
        
        if self.buffer_samples >= samples_needed:
            # Combine buffer and transcribe
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            self.buffer_samples = 0
            
            # Transcribe in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._transcribe, full_audio)
            
            if result:
                self.results_queue.put(result)
    
    def _transcribe(self, audio: np.ndarray) -> Optional[Dict]:
        """Transcribe audio chunk with Whisper."""
        try:
            # Whisper expects 16kHz audio
            if self.sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=16000)
            
            result = self.model.transcribe(audio, fp16=False, language=self.language)
            
            if result["text"].strip():
                return {
                    "text": result["text"].strip(),
                    "is_final": True,  # Whisper always returns final results
                    "confidence": 1.0,  # Whisper doesn't provide confidence
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
        
        return None
    
    async def get_results(self) -> AsyncGenerator[Dict, None]:
        """Yield transcription results."""
        while self.is_active or not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                yield result
            except queue.Empty:
                await asyncio.sleep(0.1)
    
    async def stop(self):
        """Stop and flush remaining audio."""
        self.is_active = False
        
        # Transcribe any remaining buffered audio
        if self.audio_buffer:
            full_audio = np.concatenate(self.audio_buffer)
            result = self._transcribe(full_audio)
            if result:
                self.results_queue.put(result)
        
        logger.info("Local Whisper STT stopped")


# ==================== STREAMING PROCESSOR ====================

class StreamingAudioProcessor:
    """
    Main streaming audio processor for real-time conversation intelligence.
    
    Orchestrates:
    1. Audio input (WebSocket, file simulation, microphone)
    2. Streaming STT (Google Cloud or Local Whisper)
    3. Incremental LLM analysis
    4. Real-time metrics publishing
    """
    
    def __init__(
        self,
        stt_provider: Optional[StreamingSTTProvider] = None,
        use_google_cloud_stt: bool = True,
        enable_llm_analysis: bool = True,
        publish_metrics: bool = True
    ):
        # Initialize STT provider
        if stt_provider:
            self.stt = stt_provider
        else:
            # Only use Google STT if SDK installed AND credentials are present
            google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            has_google_creds = bool(google_creds_path and os.path.exists(google_creds_path))

            if use_google_cloud_stt and GOOGLE_CLOUD_STT_AVAILABLE and has_google_creds:
                self.stt = GoogleCloudSTT()
            elif WHISPER_STREAMING_AVAILABLE:
                self.stt = LocalWhisperSTT()
            else:
                raise RuntimeError("No STT provider available. Install google-cloud-speech or whisper.")
        
        # Initialize LLM for analysis
        self.enable_llm = enable_llm_analysis
        self.llm = get_streaming_llm() if enable_llm_analysis else None
        
        # Initialize state and metrics
        self.state = StreamingAnalysisState()
        self.publisher = MetricsPublisher(enable_console=publish_metrics)
        
        # Callbacks
        self.on_transcript: Optional[Callable[[str, bool], None]] = None
        self.on_analysis: Optional[Callable[[Dict], None]] = None
        self.on_alert: Optional[Callable[[Dict], None]] = None
        
        # Control
        self.is_running = False
    
    async def start(self):
        """Start the streaming processor."""
        # Reset streaming state (rate limits, pending chunks) for new session
        reset_streaming_state()
        
        # Increase default thread pool size to handle Whisper + LLM concurrently
        try:
            loop = asyncio.get_event_loop()
            from concurrent.futures import ThreadPoolExecutor
            current = getattr(loop, '_default_executor', None)
            if current is None or getattr(current, '_max_workers', 0) < 12:
                loop.set_default_executor(ThreadPoolExecutor(max_workers=12))
        except Exception as e:
            logger.warning(f"Could not adjust default executor: {e}")

        # Attempt to start chosen STT; if Google fails (e.g., missing ADC), fall back to Whisper
        try:
            await self.stt.start()
        except Exception as e:
            logger.error(f"Failed to start STT provider: {e}")
            if WHISPER_STREAMING_AVAILABLE and not isinstance(self.stt, LocalWhisperSTT):
                logger.info("Falling back to Local Whisper STT (tiny model for speed)")
                self.stt = LocalWhisperSTT(model_name="tiny", chunk_duration=1.5)
                await self.stt.start()
            else:
                raise
        self.state.reset()
        self.is_running = True
        
        logger.info("Streaming audio processor started")
        logger.info(f"LLM Config: {LLM_CALLS_PER_MINUTE} calls/min, batch every {LLM_BATCH_TURNS} turns")
    
    async def stop(self):
        """Stop the streaming processor."""
        self.is_running = False
        await self.stt.stop()
        
        # Get final stats
        stats = get_streaming_stats()
        
        logger.info(f"Streaming session ended. Duration: {format_duration((datetime.now() - self.state.start_time).total_seconds())}")
        logger.info(f"Total words: {self.state.word_count}, Turns: {self.state.turn_count}")
        logger.info(f"LLM Stats: {stats['llm_calls_last_minute']} calls in last min, quota_exhausted={stats['llm_quota_exhausted']}")
    
    async def process_audio_chunk(self, audio_chunk: bytes):
        """
        Process a single audio chunk through the pipeline.
        
        Args:
            audio_chunk: Raw audio bytes (16-bit PCM)
        """
        if not self.is_running:
            return
        
        # Send to STT
        await self.stt.send_audio(audio_chunk)
    
    async def run_analysis_loop(self):
        """
        Main loop that processes STT results and runs LLM analysis.
        
        Run this as a separate task alongside audio input.
        """
        async for stt_result in self.stt.get_results():
            text = stt_result.get("text", "")
            is_final = stt_result.get("is_final", False)
            
            # Callback for transcript updates
            if self.on_transcript:
                self.on_transcript(text, is_final)
            
            # Only analyze finalized text
            if not is_final:
                continue
            
            # Update state with new text
            self.state.add_transcript_chunk(text)
            self.state.turn_count += 1
            
            logger.info(f"[TRANSCRIPT] {text}")
            
            # Run LLM analysis if enabled
            analysis_result = {}
            if self.enable_llm and self.llm:
                # Offload potentially blocking LLM work to a thread pool with higher concurrency
                loop = asyncio.get_event_loop()
                executor = getattr(loop, '_default_executor', None)
                if executor is None or executor._max_workers < 10:
                    # Create a custom executor with more workers to prevent starvation
                    from concurrent.futures import ThreadPoolExecutor
                    executor = ThreadPoolExecutor(max_workers=20)
                    loop.set_default_executor(executor)
                
                try:
                    analysis_result = await loop.run_in_executor(
                        None,
                        lambda: process_streaming_llm_chunk(text, self.state, self.llm)
                    )
                except Exception as e:
                    logger.error(f"LLM analysis error: {e}")
                    analysis_result = {}
                
                # Callback for analysis results
                if self.on_analysis:
                    self.on_analysis(analysis_result)
            
            # Check for and handle alerts
            if self.state.active_alerts:
                for alert in self.state.active_alerts:
                    if self.on_alert:
                        self.on_alert(alert)
                    logger.warning(f"[ALERT] {alert.get('message', 'Unknown alert')}")
                self.state.clear_alerts()
            
            # Publish metrics
            self.publisher.publish(self.state, analysis_result)
    
    async def process_audio_file(self, file_path: str, simulate_realtime: bool = True):
        """
        Process an audio file as if it were streaming.
        
        Useful for testing the streaming pipeline with recorded audio.
        
        Args:
            file_path: Path to audio file
            simulate_realtime: If True, add delays to simulate real-time streaming
        """
        import librosa
        import soundfile as sf
        
        logger.info(f"Processing audio file in streaming mode: {file_path}")
        
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=STREAMING_SAMPLE_RATE)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Calculate chunk size
        chunk_samples = int(STREAMING_CHUNK_DURATION_MS * STREAMING_SAMPLE_RATE / 1000)
        
        # Start processor
        await self.start()
        
        # Start analysis loop in background
        analysis_task = asyncio.create_task(self.run_analysis_loop())
        
        # Stream chunks
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples]
            await self.process_audio_chunk(chunk.tobytes())
            
            if simulate_realtime:
                await asyncio.sleep(STREAMING_CHUNK_DURATION_MS / 1000)
        
        # Wait for analysis to complete
        await asyncio.sleep(2)  # Give time for final processing
        
        # Stop
        await self.stop()
        analysis_task.cancel()
        
        return self.state


# ==================== WEBSOCKET SERVER ====================

async def websocket_audio_handler(websocket, path=None):
    """
    WebSocket handler for receiving audio streams.
    
    Protocol:
    - Client sends binary audio chunks (16-bit PCM, 16kHz)
    - Server sends JSON analysis results
    """
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    
    # Prefer Google STT only if SDK is installed AND credentials are configured
    use_google = False
    try:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        use_google = bool(GOOGLE_CLOUD_STT_AVAILABLE and cred_path and os.path.exists(cred_path))
    except Exception:
        use_google = False

    processor = StreamingAudioProcessor(
        use_google_cloud_stt=use_google,
        enable_llm_analysis=True
    )
    
    # Set up callbacks to send results back to client
    async def send_transcript(text: str, is_final: bool):
        try:
            await websocket.send(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": is_final
            }))
        except:
            pass
    
    async def send_analysis(result: Dict):
        try:
            await websocket.send(json.dumps({
                "type": "analysis",
                "data": result
            }))
        except:
            pass
    
    async def send_alert(alert: Dict):
        try:
            await websocket.send(json.dumps({
                "type": "alert",
                "data": alert
            }))
        except:
            pass
    
    processor.on_transcript = lambda t, f: asyncio.create_task(send_transcript(t, f))
    processor.on_analysis = lambda r: asyncio.create_task(send_analysis(r))
    processor.on_alert = lambda a: asyncio.create_task(send_alert(a))
    
    try:
        await processor.start()
        
        # Start analysis loop
        analysis_task = asyncio.create_task(processor.run_analysis_loop())
        
        # Receive and process audio chunks
        async for message in websocket:
            if isinstance(message, bytes):
                await processor.process_audio_chunk(message)
            elif isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "stop":
                    break
        
        await processor.stop()
        analysis_task.cancel()
        
        # Send final state
        await websocket.send(json.dumps({
            "type": "session_complete",
            "final_state": processor.state.to_dict()
        }))
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")


async def start_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """Start WebSocket server for audio streaming."""
    if not WEBSOCKETS_AVAILABLE:
        raise ImportError("websockets not installed. Run: pip install websockets")
    
    import websockets
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    
    # Increase ping interval/timeout to avoid premature keepalive closures during heavy processing
    # These are aggressive timeouts for Whisper transcription (can take 30+ seconds for 3 seconds of audio)
    async with websockets.serve(
        websocket_audio_handler,
        host,
        port,
        ping_interval=120,
        ping_timeout=300,  # 5 minutes for slow Whisper transcriptions
        max_size=16 * 1024 * 1024  # allow larger frames if needed
    ):
        await asyncio.Future()  # Run forever


# ==================== MAIN / CLI ====================

async def demo_file_streaming(file_path: str):
    """Demo streaming analysis on a file."""
    processor = StreamingAudioProcessor(
        use_google_cloud_stt=False,  # Use local Whisper for demo
        enable_llm_analysis=True
    )
    
    final_state = await processor.process_audio_file(file_path, simulate_realtime=False)
    
    print("\n" + "="*60)
    print("STREAMING SESSION COMPLETE")
    print("="*60)
    print(f"Duration: {format_duration((datetime.now() - final_state.start_time).total_seconds())}")
    print(f"Total Words: {final_state.word_count}")
    print(f"Total Turns: {final_state.turn_count}")
    print(f"Final Sentiment: {final_state.current_sentiment} ({final_state.sentiment_score:.2f})")
    print(f"Entities Found: {json.dumps(final_state.entities, indent=2)}")
    print(f"Alerts Triggered: {len(final_state.alert_history)}")
    
    # Save session data
    session_file = f"streaming_session_{final_state.conversation_id}.json"
    with open(session_file, "w") as f:
        json.dump({
            "state": final_state.to_dict(),
            "full_transcript": final_state.full_transcript,
            "sentiment_history": final_state.sentiment_history,
            "alert_history": final_state.alert_history
        }, f, indent=2)
    print(f"\nSession saved to: {session_file}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-Time Streaming Audio Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start WebSocket server for live audio streaming
  python stream_processor.py --server
  
  # Demo streaming analysis on an audio file
  python stream_processor.py --file test_aud.mp3
  
  # Start server on custom port
  python stream_processor.py --server --port 9000
"""
    )
    
    parser.add_argument("--server", action="store_true",
                       help="Start WebSocket server for audio streaming")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                       help="Server port (default: 8765)")
    parser.add_argument("--file", type=str,
                       help="Process audio file in streaming mode (for testing)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM analysis (STT only)")
    
    args = parser.parse_args()
    
    if args.server:
        # Start WebSocket server
        if not WEBSOCKETS_AVAILABLE:
            print("ERROR: websockets not installed. Run: pip install websockets")
            sys.exit(1)
        
        print(f"Starting streaming server on ws://{args.host}:{args.port}")
        asyncio.run(start_websocket_server(args.host, args.port))
    
    elif args.file:
        # Demo file streaming
        if not os.path.exists(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        
        asyncio.run(demo_file_streaming(args.file))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

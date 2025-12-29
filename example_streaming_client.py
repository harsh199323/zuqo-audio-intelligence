"""
WebSocket Client Example for Streaming Audio Analysis

This script demonstrates how to connect to the streaming pipeline server
and send audio for real-time analysis.

Usage:
    1. Start the server: python stream_processor.py --server
    2. Run this client: python example_streaming_client.py --file test_aud.mp3

Modes:
    --file <path>     Stream audio from a file
    --microphone      Stream from system microphone (requires pyaudio)
"""

import asyncio
import json
import sys
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for websockets
try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# Audio processing
try:
    import numpy as np
    import librosa
except ImportError:
    print("ERROR: librosa/numpy not installed. Run: pip install librosa numpy")
    sys.exit(1)


class StreamingClient:
    """WebSocket client for streaming audio to the analysis server."""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        
        # Stats
        self.chunks_sent = 0
        self.transcripts_received = 0
        self.alerts_received = 0
    
    async def connect(self):
        """Connect to the streaming server."""
        logger.info(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        self.is_connected = True
        logger.info("Connected to streaming server")
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            # Send stop signal
            await self.websocket.send(json.dumps({"type": "stop"}))
            
            # Wait for final state
            try:
                final_message = await asyncio.wait_for(self.websocket.recv(), timeout=5)
                data = json.loads(final_message)
                if data.get("type") == "session_complete":
                    logger.info("\n" + "="*50)
                    logger.info("SESSION COMPLETE")
                    logger.info("="*50)
                    final_state = data.get("final_state", {})
                    logger.info(f"Duration: {final_state.get('duration_seconds', 0):.1f}s")
                    logger.info(f"Words: {final_state.get('word_count', 0)}")
                    logger.info(f"Sentiment: {final_state.get('current_sentiment', 'unknown')}")
            except asyncio.TimeoutError:
                pass
            
            await self.websocket.close()
        
        self.is_connected = False
        logger.info(f"\nStats: {self.chunks_sent} chunks sent, "
                   f"{self.transcripts_received} transcripts, "
                   f"{self.alerts_received} alerts")
    
    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send an audio chunk to the server."""
        if self.websocket and self.is_connected:
            await self.websocket.send(audio_bytes)
            self.chunks_sent += 1
    
    async def receive_responses(self):
        """Receive and process server responses."""
        while self.is_connected:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                data = json.loads(message)
                
                msg_type = data.get("type")
                
                if msg_type == "transcript":
                    text = data.get("text", "")
                    is_final = data.get("is_final", False)
                    prefix = "✓" if is_final else "..."
                    print(f"[{prefix}] {text}")
                    if is_final:
                        self.transcripts_received += 1
                
                elif msg_type == "analysis":
                    analysis = data.get("data", {})
                    sentiment = analysis.get("sentiment", "")
                    score = analysis.get("sentiment_score", 0)
                    if sentiment:
                        print(f"    [SENTIMENT] {sentiment} ({score:+.2f})")
                
                elif msg_type == "alert":
                    alert = data.get("data", {})
                    self.alerts_received += 1
                    print(f"\n⚠️  ALERT: {alert.get('message', 'Unknown')}")
                    print(f"    Severity: {alert.get('severity', 'unknown')}")
                    print(f"    Suggestion: {alert.get('suggestion', 'N/A')}\n")
                
                elif msg_type == "session_complete":
                    break
                    
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break
    
    async def stream_file(self, file_path: str, chunk_duration_ms: int = 1000):
        """
        Stream an audio file to the server.
        
        Args:
            file_path: Path to audio file
            chunk_duration_ms: Chunk duration in milliseconds
        """
        logger.info(f"Loading audio file: {file_path}")
        
        # Load and resample to 16kHz (streaming STT requirement)
        audio, sr = librosa.load(file_path, sr=16000)
        logger.info(f"Audio loaded: {len(audio)/sr:.1f}s at {sr}Hz")
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Calculate chunk size
        chunk_samples = int(chunk_duration_ms * sr / 1000)
        
        # Connect
        await self.connect()
        
        # Start receiver task
        receiver_task = asyncio.create_task(self.receive_responses())
        
        logger.info(f"Streaming audio in {chunk_duration_ms}ms chunks...")
        print("\n" + "="*50)
        print("REAL-TIME TRANSCRIPTION")
        print("="*50)
        
        # Stream chunks
        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i:i + chunk_samples]
            await self.send_audio_chunk(chunk.tobytes())
            
            # Simulate real-time by waiting
            await asyncio.sleep(chunk_duration_ms / 1000)
            
            # Progress indicator
            progress = min(100, int((i / len(audio_int16)) * 100))
            if progress % 10 == 0 and i > 0:
                logger.debug(f"Progress: {progress}%")
        
        # Small delay for final processing
        await asyncio.sleep(2)
        
        # Disconnect
        await self.disconnect()
        receiver_task.cancel()
    
    async def stream_microphone(self):
        """Stream audio from the system microphone."""
        try:
            import pyaudio
        except ImportError:
            print("ERROR: pyaudio not installed. Run: pip install pyaudio")
            return
        
        # Audio settings
        RATE = 16000
        CHUNK = 1600  # 100ms at 16kHz
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        logger.info("Microphone streaming started. Press Ctrl+C to stop.")
        
        await self.connect()
        receiver_task = asyncio.create_task(self.receive_responses())
        
        print("\n" + "="*50)
        print("REAL-TIME TRANSCRIPTION (MICROPHONE)")
        print("="*50)
        print("Speak into your microphone...\n")
        
        try:
            while self.is_connected:
                # Read audio from microphone
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Send to server
                await self.send_audio_chunk(audio_data)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.05)
        
        except KeyboardInterrupt:
            logger.info("\nStopping microphone stream...")
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            await self.disconnect()
            receiver_task.cancel()


async def main():
    parser = argparse.ArgumentParser(
        description="WebSocket client for streaming audio analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream an audio file
  python example_streaming_client.py --file test_aud.mp3
  
  # Stream from microphone
  python example_streaming_client.py --microphone
  
  # Connect to remote server
  python example_streaming_client.py --file test.mp3 --server ws://192.168.1.100:8765
"""
    )
    
    parser.add_argument("--file", type=str, help="Audio file to stream")
    parser.add_argument("--microphone", action="store_true", help="Stream from microphone")
    parser.add_argument("--server", default="ws://localhost:8765", 
                       help="WebSocket server URL (default: ws://localhost:8765)")
    parser.add_argument("--chunk-ms", type=int, default=1000,
                       help="Chunk duration in milliseconds (default: 1000)")
    
    args = parser.parse_args()
    
    client = StreamingClient(args.server)
    
    if args.file:
        import os
        if not os.path.exists(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        await client.stream_file(args.file, args.chunk_ms)
    
    elif args.microphone:
        await client.stream_microphone()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

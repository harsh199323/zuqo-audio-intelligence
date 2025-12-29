# Hybrid Conversation Intelligence Architecture

## Overview

This project implements a **hybrid batch + streaming** architecture for comprehensive conversation intelligence in a call center environment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE OVERVIEW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LIVE CALL                          POST-CALL                               │
│   ─────────                          ─────────                               │
│   ┌──────────────┐                   ┌──────────────┐                       │
│   │  Audio       │                   │  Audio       │                       │
│   │  Stream      │                   │  File        │                       │
│   └──────┬───────┘                   └──────┬───────┘                       │
│          │                                  │                                │
│          ▼                                  ▼                                │
│   ┌──────────────┐                   ┌──────────────┐                       │
│   │  STREAMING   │                   │    BATCH     │                       │
│   │  PIPELINE    │                   │   PIPELINE   │                       │
│   │              │                   │              │                       │
│   │  • Fast STT  │                   │  • Whisper   │                       │
│   │  • Gemini    │                   │  • Gemini    │                       │
│   │    Flash     │                   │    Pro       │                       │
│   │  • Real-time │                   │  • Deep      │                       │
│   │    alerts    │                   │    analysis  │                       │
│   └──────┬───────┘                   └──────┬───────┘                       │
│          │                                  │                                │
│          ▼                                  ▼                                │
│   ┌──────────────┐                   ┌──────────────┐                       │
│   │  Live Agent  │                   │  QA Report   │                       │
│   │  Dashboard   │                   │  & Training  │                       │
│   └──────────────┘                   └──────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Comparison

| Feature | Streaming Pipeline | Batch Pipeline |
|---------|-------------------|----------------|
| **File** | `stream_processor.py` | `test.py` + `run_advanced_analysis.py` |
| **Latency** | ~1-3 seconds | Minutes (full file) |
| **STT Provider** | Google Cloud Speech-to-Text | OpenAI Whisper |
| **LLM Model** | `gemini-2.5-flash` (fast) | `gemini-1.5-pro` (accurate) |
| **Analysis Depth** | Incremental, shallow | Complete, deep |
| **Use Case** | Live coaching | Post-call QA |
| **Output** | Real-time metrics, alerts | Full JSON report |

## File Structure

```
zuqo/
├── utils.py                    # Shared utilities (LLM, JSON, state management)
├── stream_processor.py         # STREAMING: Real-time analysis pipeline
├── test.py                     # BATCH: Full-file processing pipeline
├── run_advanced_analysis.py    # BATCH: Deep LLM analysis runner
├── helper_test.py              # Prompt templates for batch analysis
├── example_streaming_client.py # Example WebSocket client
├── requirements.txt            # Dependencies
├── .env.example                # Environment configuration template
└── models/                     # Downloaded model files
```

## Streaming Pipeline (`stream_processor.py`)

### Purpose
Real-time conversation intelligence for **live agent coaching** during calls.

### Features
- **Low-latency STT**: Google Cloud Speech-to-Text streaming (interim + final results)
- **Incremental LLM**: Analyzes each sentence as it's transcribed
- **Live Metrics**: Sentiment, entities, script adherence updated in real-time
- **Instant Alerts**: Triggers when sentiment drops or script deviation detected
- **WebSocket Server**: Accepts audio streams from any client

### Usage

```bash
# Start the WebSocket server
python stream_processor.py --server --port 8765

# In another terminal, stream audio from a file
python example_streaming_client.py --file test_aud.mp3

# Or stream from microphone (requires pyaudio)
python example_streaming_client.py --microphone
```

### WebSocket Protocol

**Client → Server:**
- Binary: Raw audio bytes (16-bit PCM, 16kHz)
- JSON: `{"type": "stop"}` to end session

**Server → Client:**
```json
{"type": "transcript", "text": "Hello how are you", "is_final": true}
{"type": "analysis", "data": {"sentiment": "neutral", "sentiment_score": 0.1}}
{"type": "alert", "data": {"message": "Customer frustrated", "severity": "high"}}
{"type": "session_complete", "final_state": {...}}
```

### Real-Time Metrics

The streaming pipeline tracks:
- **Sentiment**: Current emotional state and score (-1 to +1)
- **Entities**: Names, locations, products, dates, amounts
- **Script Adherence**: Which steps completed, any deviations
- **Alerts**: Automatic warnings for agents

## Batch Pipeline (`test.py` + `run_advanced_analysis.py`)

### Purpose
Comprehensive **post-call quality assurance** and analysis.

### Features
- **High-Accuracy STT**: Whisper with full audio context
- **Noise Detection**: DeepFilterNet ML-based noise reduction
- **Deep Analysis**: Issue tree, call rating, detailed summaries
- **Complete Report**: JSON with all metrics and insights

### Usage

```bash
# Run full pipeline with audio file
python test.py

# Run advanced LLM analysis only (uses transcript from test.py)
python run_advanced_analysis.py
```

### Output Files

- `pipeline_result.json`: Full pipeline results
- `advanced_result.json`: Deep LLM analysis
- `enhanced_audio_deepfilternet.wav`: Noise-reduced audio

## Shared Utilities (`utils.py`)

Common functions used by both pipelines:

```python
from utils import (
    # LLM
    initialize_llm_model,
    get_streaming_llm,      # Fast model for streaming
    get_batch_llm,          # Accurate model for batch
    
    # JSON
    extract_json_from_text,
    
    # Streaming State
    StreamingAnalysisState,
    process_streaming_llm_chunk,
    
    # Batch Analysis
    call_llm_with_retry,
    
    # Metrics
    MetricsPublisher,
)
```

## Configuration

### Environment Variables (`.env`)

```bash
# API Keys
GOOGLE_API_KEY=your-key-here

# Models
GEMINI_MODEL_FAST=gemini-2.5-flash    # Streaming
GEMINI_MODEL_DEEP=gemini-1.5-pro      # Batch

# Streaming
STREAMING_SAMPLE_RATE=16000
STREAMING_CHUNK_DURATION_MS=1000
STREAMING_PORT=8765

# Alerts
SENTIMENT_ALERT_THRESHOLD=-0.5
```

### Google Cloud Setup (for Streaming STT)

1. Create a Google Cloud project
2. Enable Speech-to-Text API
3. Create a service account with `Speech-to-Text` role
4. Download JSON key file
5. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# For streaming (recommended)
pip install google-cloud-speech websockets

# For microphone input (optional)
pip install pyaudio
```

## Architecture Benefits

### 1. Cost Optimization
- **Streaming**: Uses cheap, fast `gemini-2.5-flash` for real-time
- **Batch**: Uses powerful `gemini-1.5-pro` only for deep analysis

### 2. Latency vs Accuracy Trade-off
- **Streaming**: Low latency (<3s) for live coaching
- **Batch**: High accuracy for compliance and training

### 3. Scalability
- **Streaming**: WebSocket server handles concurrent connections
- **Batch**: File-based processing, easy to parallelize

### 4. Resilience
- If streaming fails, batch analysis still provides complete report
- If LLM quota exceeded, STT and noise detection continue working

## Example Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. CALL STARTS                                                         │
│      │                                                                   │
│      ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────┐          │
│   │  STREAMING PIPELINE (Real-Time)                          │          │
│   │  ─────────────────────────────                           │          │
│   │  • Agent sees live transcript                            │          │
│   │  • Sentiment score updates every sentence                │          │
│   │  • Alert pops up: "Customer frustrated - offer discount" │          │
│   │  • Agent adjusts approach in real-time                   │          │
│   └──────────────────────────────────────────────────────────┘          │
│      │                                                                   │
│      ▼                                                                   │
│   2. CALL ENDS                                                           │
│      │                                                                   │
│      ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────┐          │
│   │  BATCH PIPELINE (Post-Call)                              │          │
│   │  ─────────────────────────────                           │          │
│   │  • High-accuracy Whisper transcription                   │          │
│   │  • DeepFilterNet noise reduction                         │          │
│   │  • Full issue tree analysis                              │          │
│   │  • Call quality rating                                   │          │
│   │  • Detailed compliance report                            │          │
│   └──────────────────────────────────────────────────────────┘          │
│      │                                                                   │
│      ▼                                                                   │
│   3. REPORTS & TRAINING                                                  │
│      • QA manager reviews call quality                                   │
│      • Training team uses examples                                       │
│      • Analytics dashboard updated                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Deploy Streaming Server**: Run `stream_processor.py --server` on a cloud VM
2. **Integrate with Phone System**: Connect audio streams via WebSocket
3. **Build Agent Dashboard**: Display real-time metrics and alerts
4. **Set Up Batch Jobs**: Schedule `run_advanced_analysis.py` for post-call processing
5. **Add Database Storage**: Store analysis results in MongoDB or PostgreSQL

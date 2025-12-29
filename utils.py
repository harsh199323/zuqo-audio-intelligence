"""
Shared Utilities for Conversation Intelligence Pipeline

This module provides shared utilities used by both:
- Batch Pipeline (test.py, run_advanced_analysis.py) - Full-file post-call analysis
- Streaming Pipeline (stream_processor.py) - Real-time incremental analysis

Functions:
- LLM initialization and configuration
- JSON extraction from LLM responses
- Incremental state processing for streaming
- Shared constants and configurations
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================

# API Keys
GOOGLE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)

# Model Configuration
GEMINI_MODEL_FAST = os.environ.get("GEMINI_MODEL_FAST", "gemini-2.5-flash")  # For streaming (fast, cheap)
GEMINI_MODEL_DEEP = os.environ.get("GEMINI_MODEL_DEEP", "gemini-1.5-pro")    # For batch (accurate, thorough)

# Streaming Configuration
STREAMING_CHUNK_DURATION_MS = int(os.environ.get("STREAMING_CHUNK_DURATION_MS", "1000"))  # 1 second chunks
STREAMING_STT_LANGUAGE = os.environ.get("STREAMING_STT_LANGUAGE", "en-US")
STREAMING_SAMPLE_RATE = int(os.environ.get("STREAMING_SAMPLE_RATE", "16000"))  # 16kHz for STT

# Real-time Analysis Thresholds
SENTIMENT_ALERT_THRESHOLD = float(os.environ.get("SENTIMENT_ALERT_THRESHOLD", "-0.5"))  # Trigger alert below this
SILENCE_ALERT_SECONDS = float(os.environ.get("SILENCE_ALERT_SECONDS", "10.0"))  # Alert after N seconds silence

# LLM Rate Limiting (to avoid quota exhaustion)
LLM_CALLS_PER_MINUTE = int(os.environ.get("LLM_CALLS_PER_MINUTE", "5"))  # Max LLM calls per minute
LLM_BATCH_TURNS = int(os.environ.get("LLM_BATCH_TURNS", "5"))  # Batch N turns before LLM call
USE_LOCAL_SENTIMENT_FALLBACK = os.environ.get("USE_LOCAL_SENTIMENT_FALLBACK", "true").lower() == "true"


# ==================== LLM INITIALIZATION ====================

def initialize_llm_model(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_retries: int = 3,
    timeout: int = 30
) -> ChatGoogleGenerativeAI:
    """
    Initialize a Google Gemini LLM instance.
    
    Args:
        model_name: Model to use (defaults to GEMINI_MODEL_FAST)
        temperature: Creativity parameter (0-1)
        max_tokens: Maximum response tokens
        max_retries: Number of retries on failure
        timeout: Request timeout in seconds
    
    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance
    """
    if not GOOGLE_API_KEY:
        raise ValueError("API key not found! Set GOOGLE_API_KEY or GEMINI_API_KEY in .env file")
    
    model = model_name or GEMINI_MODEL_FAST
    
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
    )
    
    logger.info(f"LLM initialized: {model} (temp={temperature})")
    return llm


def get_streaming_llm() -> ChatGoogleGenerativeAI:
    """
    Get an LLM optimized for streaming (fast model, low latency).
    """
    return initialize_llm_model(
        model_name=GEMINI_MODEL_FAST,
        temperature=0.3,  # Lower temperature for consistency
        max_tokens=256,   # Smaller responses for speed
        timeout=15        # Shorter timeout for real-time
    )


def get_batch_llm() -> ChatGoogleGenerativeAI:
    """
    Get an LLM optimized for batch processing (accurate model, thorough analysis).
    """
    return initialize_llm_model(
        model_name=GEMINI_MODEL_DEEP,
        temperature=0.7,
        max_tokens=2048,  # Larger responses for detailed analysis
        timeout=120       # Longer timeout for complex prompts
    )


# ==================== JSON EXTRACTION ====================

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON object or array from text (handles markdown code blocks).
    
    Args:
        text: Raw text potentially containing JSON
    
    Returns:
        Parsed JSON as dict/list, or None if not found
    """
    if not text:
        return None
    
    try:
        # First, try to parse the entire text as JSON
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_matches = re.findall(code_block_pattern, text)
    
    for match in code_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON object or array
    json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
    match = re.search(json_pattern, text)
    
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    logger.warning("Could not extract JSON from text")
    return None


# ==================== STREAMING STATE MANAGEMENT ====================

class StreamingAnalysisState:
    """
    Manages the evolving state of a real-time conversation analysis.
    
    This state object is updated incrementally as new transcript chunks arrive,
    maintaining context across the entire conversation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset state for a new conversation."""
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        
        # Accumulated transcript
        self.full_transcript = ""
        self.transcript_chunks = []
        
        # Real-time metrics
        self.current_sentiment = "neutral"
        self.sentiment_score = 0.0
        self.sentiment_history = []
        
        # Entities detected
        self.entities = {
            "people": [],
            "locations": [],
            "products": [],
            "dates": [],
            "amounts": []
        }
        
        # Script adherence tracking
        self.script_steps_completed = []
        self.script_adherence_issues = []
        
        # Alerts
        self.active_alerts = []
        self.alert_history = []
        
        # Call metrics
        self.word_count = 0
        self.turn_count = 0
        self.silence_duration = 0.0
        
        logger.info(f"New streaming session started: {self.conversation_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for LLM context and serialization."""
        return {
            "conversation_id": self.conversation_id,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "current_sentiment": self.current_sentiment,
            "sentiment_score": self.sentiment_score,
            "entities": self.entities,
            "script_steps_completed": self.script_steps_completed,
            "script_adherence_issues": self.script_adherence_issues,
            "active_alerts": self.active_alerts,
            "word_count": self.word_count,
            "turn_count": self.turn_count,
            "transcript_length": len(self.full_transcript)
        }
    
    def add_transcript_chunk(self, chunk: str):
        """Add a new transcript chunk."""
        self.transcript_chunks.append({
            "text": chunk,
            "timestamp": datetime.now().isoformat(),
            "word_count": len(chunk.split())
        })
        self.full_transcript += " " + chunk
        self.word_count += len(chunk.split())
        self.last_update = datetime.now()
    
    def update_from_llm_response(self, llm_response: Dict[str, Any]):
        """Update state from LLM analysis response."""
        if not llm_response:
            return
        
        # Update sentiment
        if "sentiment" in llm_response:
            self.current_sentiment = llm_response["sentiment"]
        if "sentiment_score" in llm_response:
            self.sentiment_score = float(llm_response["sentiment_score"])
            self.sentiment_history.append({
                "score": self.sentiment_score,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update entities
        if "entities" in llm_response:
            for entity_type, entities in llm_response["entities"].items():
                if entity_type in self.entities:
                    for entity in entities:
                        if entity not in self.entities[entity_type]:
                            self.entities[entity_type].append(entity)
        
        # Update script adherence
        if "script_step_completed" in llm_response:
            step = llm_response["script_step_completed"]
            if step and step not in self.script_steps_completed:
                self.script_steps_completed.append(step)
        
        if "script_issue" in llm_response and llm_response["script_issue"]:
            self.script_adherence_issues.append({
                "issue": llm_response["script_issue"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Handle alerts
        if "alert" in llm_response and llm_response["alert"]:
            alert = {
                "type": llm_response.get("alert_type", "general"),
                "message": llm_response["alert"],
                "severity": llm_response.get("alert_severity", "medium"),
                "timestamp": datetime.now().isoformat()
            }
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
    
    def check_sentiment_alert(self) -> Optional[Dict]:
        """Check if sentiment has dropped below threshold."""
        if self.sentiment_score < SENTIMENT_ALERT_THRESHOLD:
            return {
                "type": "sentiment_drop",
                "message": f"Customer sentiment is negative ({self.sentiment_score:.2f})",
                "severity": "high",
                "suggestion": "Consider offering assistance or escalating"
            }
        return None
    
    def clear_alerts(self):
        """Clear active alerts (after they've been acknowledged)."""
        self.active_alerts = []


# ==================== LOCAL SENTIMENT FALLBACK ====================

# Simple keyword-based sentiment analysis (no API calls)
POSITIVE_KEYWORDS = {
    "thank", "thanks", "great", "good", "excellent", "perfect", "wonderful",
    "appreciate", "helpful", "happy", "pleased", "awesome", "love", "best",
    "okay", "sure", "yes", "fine", "right", "correct"
}

NEGATIVE_KEYWORDS = {
    "angry", "frustrated", "annoyed", "terrible", "awful", "bad", "worst",
    "hate", "disappointed", "upset", "wrong", "problem", "issue", "complaint",
    "ridiculous", "unacceptable", "stupid", "useless", "waste", "never"
}

UNSATISFIED_PHRASES = [
    "not working", "doesn't work", "can't believe", "this is ridiculous",
    "speak to manager", "speak to supervisor", "escalate", "cancel",
    "refund", "never again", "worst experience"
]


def local_sentiment_analysis(text: str) -> Dict[str, Any]:
    """
    Fast, local sentiment analysis without API calls.
    Used as fallback when LLM quota is exhausted.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dict with sentiment, score, and detected indicators
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    
    # Count keyword matches
    positive_count = len(words & POSITIVE_KEYWORDS)
    negative_count = len(words & NEGATIVE_KEYWORDS)
    
    # Check for unsatisfied phrases
    phrase_matches = [p for p in UNSATISFIED_PHRASES if p in text_lower]
    if phrase_matches:
        negative_count += len(phrase_matches) * 2  # Weight phrases higher
    
    # Calculate score
    total = positive_count + negative_count
    if total == 0:
        score = 0.0
        sentiment = "neutral"
    else:
        score = (positive_count - negative_count) / max(total, 1)
        score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
        
        if score > 0.3:
            sentiment = "positive"
        elif score < -0.3:
            sentiment = "negative" if score > -0.6 else "frustrated"
        else:
            sentiment = "neutral"
    
    # Simple entity extraction
    entities = extract_simple_entities(text)
    
    return {
        "sentiment": sentiment,
        "sentiment_score": round(score, 2),
        "entities": entities,
        "analysis_method": "local_fallback",
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "phrase_matches": phrase_matches
    }


def extract_simple_entities(text: str) -> Dict[str, List[str]]:
    """
    Simple regex-based entity extraction (no LLM required).
    """
    entities = {
        "people": [],
        "locations": [],
        "products": [],
        "dates": [],
        "amounts": []
    }
    
    # Date/time patterns
    date_patterns = [
        r'\b(today|tomorrow|yesterday|morning|afternoon|evening|night)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b',
        r'\b(\d{1,2}(?:st|nd|rd|th)?)\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if match and match not in entities["dates"]:
                entities["dates"].append(match)
    
    # Amount patterns (currency)
    amount_patterns = [
        r'\$[\d,]+(?:\.\d{2})?',
        r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|rupees?|rs\.?|inr|usd)\b',
        r'\b(?:rs\.?|inr|\$)\s*\d+(?:,\d{3})*(?:\.\d{2})?\b'
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match and match not in entities["amounts"]:
                entities["amounts"].append(match)
    
    return entities


# ==================== STREAMING LLM PROCESSING ====================

# Default prompt for streaming analysis
STREAMING_ANALYSIS_PROMPT = """You are a real-time conversation analyst for a call center.
Analyze the new transcript chunk and update the conversation state.

Current conversation state:
{current_state}

New transcript chunk to analyze:
"{new_chunk}"

Provide your analysis as a JSON object with these fields:
{{
    "sentiment": "positive" | "neutral" | "negative" | "frustrated" | "angry",
    "sentiment_score": float between -1.0 (very negative) and 1.0 (very positive),
    "entities": {{
        "people": ["list of names mentioned"],
        "locations": ["list of places mentioned"],
        "products": ["list of products/services mentioned"],
        "dates": ["list of dates/times mentioned"],
        "amounts": ["list of monetary amounts mentioned"]
    }},
    "script_step_completed": "name of script step if detected, else null",
    "script_issue": "description of any script deviation, else null",
    "alert": "urgent message for agent if needed, else null",
    "alert_type": "sentiment_drop" | "customer_frustration" | "script_deviation" | "escalation_needed" | null,
    "alert_severity": "low" | "medium" | "high" | "critical",
    "summary_update": "one sentence update on conversation progress"
}}

Return ONLY the JSON object, no additional text."""


# Track LLM call rate for throttling
_llm_call_times = []
_llm_quota_exhausted = False
_pending_chunks = []  # Buffer for batching


def _is_rate_limited() -> bool:
    """Check if we've exceeded the LLM call rate limit."""
    global _llm_call_times
    now = datetime.now()
    # Remove calls older than 1 minute
    _llm_call_times = [t for t in _llm_call_times if (now - t).seconds < 60]
    return len(_llm_call_times) >= LLM_CALLS_PER_MINUTE


def _record_llm_call():
    """Record an LLM call for rate limiting."""
    _llm_call_times.append(datetime.now())


def reset_streaming_state():
    """
    Reset streaming state between sessions.
    Call this at the start of a new streaming session.
    """
    global _llm_call_times, _llm_quota_exhausted, _pending_chunks
    _llm_call_times = []
    _llm_quota_exhausted = False
    _pending_chunks = []
    logger.info("Streaming state reset for new session")


def get_streaming_stats() -> Dict[str, Any]:
    """Get current streaming analysis statistics."""
    return {
        "llm_calls_last_minute": len(_llm_call_times),
        "llm_quota_exhausted": _llm_quota_exhausted,
        "pending_chunks": len(_pending_chunks),
        "rate_limit_max": LLM_CALLS_PER_MINUTE,
        "batch_size": LLM_BATCH_TURNS
    }


def process_streaming_llm_chunk(
    new_transcript_chunk: str,
    current_state: StreamingAnalysisState,
    llm_model: ChatGoogleGenerativeAI,
    custom_prompt: Optional[str] = None,
    force_local_fallback: bool = False
) -> Dict[str, Any]:
    """
    Analyze a new chunk of transcript against the existing state and return
    an updated state object for real-time metrics.
    
    FEATURES:
    - Rate limiting to avoid quota exhaustion (max LLM_CALLS_PER_MINUTE/min)
    - Batches LLM_BATCH_TURNS turns before making an LLM call
    - Falls back to local sentiment analysis on quota errors
    
    Args:
        new_transcript_chunk: New text from the conversation
        current_state: Current StreamingAnalysisState object
        llm_model: Initialized LLM for analysis
        custom_prompt: Optional custom prompt template
        force_local_fallback: Skip LLM and use local analysis
    
    Returns:
        Dict containing the analysis response
    """
    global _llm_quota_exhausted, _pending_chunks
    
    if not new_transcript_chunk or not new_transcript_chunk.strip():
        return {}
    
    # Add to pending chunks for batching
    _pending_chunks.append(new_transcript_chunk)
    
    # Check if we should use local fallback
    use_local = (
        force_local_fallback or 
        _llm_quota_exhausted or 
        (USE_LOCAL_SENTIMENT_FALLBACK and _is_rate_limited())
    )
    
    # If using local fallback, analyze immediately
    if use_local:
        result = local_sentiment_analysis(new_transcript_chunk)
        current_state.update_from_llm_response(result)
        logger.info(f"[LOCAL] Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")
        return result
    
    # Batch turns before LLM call (reduces API calls)
    if len(_pending_chunks) < LLM_BATCH_TURNS:
        # Use local analysis for immediate feedback, batch for LLM later
        result = local_sentiment_analysis(new_transcript_chunk)
        current_state.update_from_llm_response(result)
        return {**result, "pending_llm_batch": len(_pending_chunks)}
    
    # Time to make an LLM call with batched chunks
    batched_text = " ".join(_pending_chunks)
    _pending_chunks = []  # Clear the buffer
    
    try:
        # Check rate limit
        if _is_rate_limited():
            logger.warning(f"Rate limited: {LLM_CALLS_PER_MINUTE} calls/min exceeded. Using local fallback.")
            result = local_sentiment_analysis(batched_text)
            current_state.update_from_llm_response(result)
            return result
        
        # Build the prompt
        prompt_template = custom_prompt or STREAMING_ANALYSIS_PROMPT
        prompt = prompt_template.format(
            current_state=json.dumps(current_state.to_dict(), indent=2),
            new_chunk=batched_text
        )
        
        # Call LLM
        messages = [("human", prompt)]
        _record_llm_call()
        response = llm_model.invoke(messages)
        
        # Extract JSON from response
        result = extract_json_from_text(response.content)
        
        if result:
            result["analysis_method"] = "llm"
            result["batch_size"] = LLM_BATCH_TURNS
            
            # Update the state object
            current_state.update_from_llm_response(result)
            
            # Check for automatic alerts
            sentiment_alert = current_state.check_sentiment_alert()
            if sentiment_alert and "alert" not in result:
                result["auto_alert"] = sentiment_alert
            
            logger.info(f"[LLM] Sentiment: {result.get('sentiment')} ({result.get('sentiment_score', 0):.2f})")
        
        return result or {}
    
    except Exception as e:
        error_str = str(e)
        logger.error(f"Error in streaming LLM chunk processing: {e}")
        
        # Check for quota exhaustion
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            _llm_quota_exhausted = True
            logger.warning("LLM quota exhausted! Switching to local sentiment fallback for remainder of session.")
        
        # Fallback to local analysis
        result = local_sentiment_analysis(batched_text)
        result["llm_error"] = error_str[:100]
        current_state.update_from_llm_response(result)
        return result


# ==================== BATCH ANALYSIS UTILITIES ====================

def call_llm_with_retry(
    prompt: str,
    transcript: str,
    llm_model: ChatGoogleGenerativeAI,
    analysis_name: str = "Analysis",
    max_retries: int = 3
) -> Optional[Dict]:
    """
    Make a single LLM call with retry logic for batch analysis.
    
    Args:
        prompt: System prompt for the analysis
        transcript: Full transcript to analyze
        llm_model: LLM instance to use
        analysis_name: Name for logging
        max_retries: Number of retry attempts
    
    Returns:
        Parsed JSON response or None on failure
    """
    messages = [
        ("system", prompt),
        ("human", f"Full Transcript:\n{transcript}"),
    ]
    
    for attempt in range(max_retries):
        logger.info(f"[{analysis_name}] Attempt {attempt + 1}/{max_retries}...")
        try:
            response = llm_model.invoke(messages)
            content = response.content
            
            # Extract JSON
            result = extract_json_from_text(content)
            if not result:
                raise ValueError("No JSON found in response")
            
            logger.info(f"[{analysis_name}] Success!")
            return result
            
        except Exception as e:
            logger.warning(f"[{analysis_name}] Error: {type(e).__name__}: {str(e)[:100]}")
            if attempt == max_retries - 1:
                logger.error(f"[{analysis_name}] Failed after {max_retries} attempts")
                return None
    
    return None


# ==================== METRICS PUBLISHING ====================

class MetricsPublisher:
    """
    Publishes real-time metrics to configured destinations.
    
    Supports multiple output channels:
    - Console logging
    - WebSocket broadcast
    - Webhook POST
    - File append (for debugging)
    """
    
    def __init__(self, enable_console: bool = True, enable_file: bool = False):
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.file_path = "streaming_metrics.jsonl"
        self.websocket_clients = []
    
    def publish(self, state: StreamingAnalysisState, analysis_result: Dict[str, Any]):
        """Publish current state and analysis to all configured channels."""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": state.conversation_id,
            "state": state.to_dict(),
            "latest_analysis": analysis_result,
            "active_alerts": state.active_alerts
        }
        
        if self.enable_console:
            self._log_to_console(metrics)
        
        if self.enable_file:
            self._append_to_file(metrics)
        
        # Broadcast to WebSocket clients
        self._broadcast_websocket(metrics)
    
    def _log_to_console(self, metrics: Dict):
        """Log metrics to console."""
        sentiment = metrics["state"].get("current_sentiment", "unknown")
        score = metrics["state"].get("sentiment_score", 0)
        alerts = len(metrics.get("active_alerts", []))
        
        logger.info(f"[METRICS] Sentiment: {sentiment} ({score:.2f}) | "
                   f"Alerts: {alerts} | Words: {metrics['state'].get('word_count', 0)}")
    
    def _append_to_file(self, metrics: Dict):
        """Append metrics to JSONL file."""
        try:
            with open(self.file_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Failed to write metrics to file: {e}")
    
    def _broadcast_websocket(self, metrics: Dict):
        """Broadcast to connected WebSocket clients."""
        # Placeholder for WebSocket implementation
        # In production, this would use asyncio and websockets library
        pass
    
    def register_websocket_client(self, client):
        """Register a WebSocket client for real-time updates."""
        self.websocket_clients.append(client)
    
    def unregister_websocket_client(self, client):
        """Unregister a WebSocket client."""
        if client in self.websocket_clients:
            self.websocket_clients.remove(client)


# ==================== SCRIPT ADHERENCE TEMPLATES ====================

# Default call center script steps for tracking
DEFAULT_SCRIPT_STEPS = [
    {"step": "greeting", "keywords": ["hello", "hi", "good morning", "good afternoon", "welcome"]},
    {"step": "identify_caller", "keywords": ["name", "who am i speaking", "may i know"]},
    {"step": "verify_account", "keywords": ["verify", "account number", "phone number", "email"]},
    {"step": "understand_issue", "keywords": ["how can i help", "what seems to be", "issue", "problem"]},
    {"step": "provide_solution", "keywords": ["let me", "i can help", "solution", "resolve"]},
    {"step": "confirm_resolution", "keywords": ["resolved", "helped", "satisfied", "anything else"]},
    {"step": "closing", "keywords": ["thank you", "goodbye", "have a nice day", "appreciate"]}
]


def check_script_adherence(transcript_chunk: str, completed_steps: List[str]) -> Optional[str]:
    """
    Check if a transcript chunk matches any script step.
    
    Args:
        transcript_chunk: New text to check
        completed_steps: List of already completed step names
    
    Returns:
        Name of completed step if detected, else None
    """
    chunk_lower = transcript_chunk.lower()
    
    for step in DEFAULT_SCRIPT_STEPS:
        if step["step"] in completed_steps:
            continue
        
        for keyword in step["keywords"]:
            if keyword in chunk_lower:
                return step["step"]
    
    return None


# ==================== UTILITY FUNCTIONS ====================

def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_words_per_minute(word_count: int, duration_seconds: float) -> float:
    """Calculate speaking rate in words per minute."""
    if duration_seconds <= 0:
        return 0.0
    return (word_count / duration_seconds) * 60


if __name__ == "__main__":
    # Test utilities
    print("Testing shared utilities...")
    
    # Test JSON extraction
    test_cases = [
        '{"sentiment": "positive", "score": 0.8}',
        '```json\n{"test": true}\n```',
        'Here is the result: {"result": "success"} and some more text',
    ]
    
    for test in test_cases:
        result = extract_json_from_text(test)
        print(f"Input: {test[:50]}... -> Extracted: {result}")
    
    # Test state management
    state = StreamingAnalysisState()
    state.add_transcript_chunk("Hello, how can I help you today?")
    print(f"\nState after chunk: {json.dumps(state.to_dict(), indent=2)}")
    
    print("\nUtilities test complete!")

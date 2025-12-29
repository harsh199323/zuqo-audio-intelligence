"""
Production-Grade Advanced Audio Analytics Runner

Features:
- Parallel LLM execution (4x faster)
- Comprehensive error handling & logging
- Input validation & retry logic
- MongoDB integration
- Performance monitoring
- Environment configuration
- Graceful degradation
"""

import ast
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path

from test import process_audio_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Optional: MongoDB for production storage
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("pymongo not installed - database features disabled")

# ==================== CONFIGURATION ====================

# Load environment
load_dotenv()

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"analysis_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File paths
HELPER_FILE = Path("helper_test.py")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# MongoDB Configuration (optional)
MONGO_URI = os.getenv("MONGO_DB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "audio_analytics")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "analysis_results")

# Analysis Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
PARALLEL_EXECUTION = os.getenv("PARALLEL_EXECUTION", "true").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Validation
if not GOOGLE_API_KEY:
    raise ValueError("API key not found! Set GOOGLE_API_KEY or GEMINI_API_KEY in .env file")

logger.info(f"Configuration loaded: Model={GEMINI_MODEL}, Parallel={PARALLEL_EXECUTION}, Workers={MAX_WORKERS}")

# ==================== HELPER FUNCTIONS ====================

def extract_prompt_from_source(function_name: str) -> Optional[str]:
    """Extract PROMPT variable from a function in helper_test.py with error handling"""
    if not HELPER_FILE.exists():
        logger.error(f"{HELPER_FILE} not found")
        return None

    try:
        with open(HELPER_FILE, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for subnode in node.body:
                    if isinstance(subnode, ast.Assign):
                        for target in subnode.targets:
                            if isinstance(target, ast.Name) and target.id == "PROMPT":
                                if isinstance(subnode.value, ast.Constant):
                                    return subnode.value.value
                                elif isinstance(subnode.value, ast.Str):
                                    return subnode.value.s
        
        logger.warning(f"PROMPT not found in {function_name}")
        return None
                                    
    except Exception as e:
        logger.error(f"Error extracting prompt from {function_name}: {e}")
        return None

def extract_json(text: str) -> Optional[str]:
    """Extract JSON from LLM response (handles markdown formatting)"""
    import re
    json_pattern = r"(\{.*\}|\[.*\])"
    match = re.search(json_pattern, text, re.DOTALL)
    return match.group(0) if match else None

def validate_audio_file(audio_path: str) -> bool:
    """Validate audio file exists and has correct extension"""
    path = Path(audio_path)
    
    if not path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return False
    
    valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    if path.suffix.lower() not in valid_extensions:
        logger.error(f"Invalid audio format: {path.suffix}. Supported: {valid_extensions}")
        return False
    
    return True

# ==================== LLM ANALYSIS ====================

def call_llm_analysis(
    prompt: str, 
    transcript: str, 
    analysis_name: str, 
    max_retries: int = MAX_RETRIES
) -> Optional[Dict]:
    """Make a single LLM call with retry logic and comprehensive error handling"""
    
    if not prompt:
        logger.error(f"[{analysis_name}] No prompt provided")
        return None
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_retries=5,
        timeout=LLM_TIMEOUT,
    )
    
    messages = [
        ("system", prompt),
        ("human", f"Full Transcript:\n{transcript}"),
    ]
    
    start_time = time.time()
    
    for attempt in range(max_retries):
        logger.info(f"[{analysis_name}] Attempt {attempt + 1}/{max_retries}")
        
        try:
            response = llm.invoke(messages)
            content = response.content
            
            # Extract JSON
            json_str = extract_json(content)
            if not json_str:
                raise ValueError("No JSON found in response")
            
            result = json.loads(json_str)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ [{analysis_name}] Success in {elapsed:.2f}s")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[{analysis_name}] JSON parsing error: {e}")
            logger.debug(f"Raw response: {content[:200]}...")
            
        except Exception as e:
            logger.error(f"[{analysis_name}] Error: {type(e).__name__}: {str(e)[:100]}")
            
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.info(f"[{analysis_name}] Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    logger.error(f"✗ [{analysis_name}] Failed after {max_retries} attempts")
    return None

def run_analyses_parallel(prompts: Dict[str, str], transcript: str) -> Dict[str, Optional[Dict]]:
    """Run all 4 analyses in parallel using ThreadPoolExecutor"""
    
    results = {}
    
    # Map analysis names to function calls
    analysis_tasks = {
        "key_analysis": ("Key Analysis", prompts.get("key_analysis")),
        "statement_analysis": ("Statement Analysis", prompts.get("statement_analysis")),
        "issue_tree": ("Issue Tree", prompts.get("issue_tree")),
        "call_rating": ("Call Rating", prompts.get("call_rating")),
    }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_analysis = {
            executor.submit(call_llm_analysis, prompt, transcript, name): key
            for key, (name, prompt) in analysis_tasks.items()
            if prompt  # Only submit if prompt exists
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_analysis):
            analysis_key = future_to_analysis[future]
            try:
                results[analysis_key] = future.result()
            except Exception as e:
                logger.error(f"Exception in {analysis_key}: {e}")
                results[analysis_key] = None
    
    return results

def run_analyses_sequential(prompts: Dict[str, str], transcript: str) -> Dict[str, Optional[Dict]]:
    """Run all 4 analyses sequentially (fallback mode)"""
    
    results = {
        "key_analysis": call_llm_analysis(
            prompts.get("key_analysis"), transcript, "Key Analysis"
        ),
        "statement_analysis": call_llm_analysis(
            prompts.get("statement_analysis"), transcript, "Statement Analysis"
        ),
        "issue_tree": call_llm_analysis(
            prompts.get("issue_tree"), transcript, "Issue Tree"
        ),
        "call_rating": call_llm_analysis(
            prompts.get("call_rating"), transcript, "Call Rating"
        ),
    }
    
    return results

# ==================== RESULT PROCESSING ====================

def merge_results(
    key_analysis: Optional[Dict],
    statement_analysis: Optional[Dict],
    issue_tree: Optional[Dict],
    call_rating: Optional[Dict]
) -> Dict:
    """Merge all 4 analysis results with graceful degradation"""
    
    merged = {
        "metadata": {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "model": GEMINI_MODEL,
            "version": "2.0.0"
        }
    }
    
    # Add Key Analysis fields
    if key_analysis:
        merged.update(key_analysis)
    else:
        logger.warning("Key Analysis missing - using defaults")
        merged.update({"Summary": "Analysis failed", "Entities": [], "customer_metrics": {}})
    
    # Add Statement Analysis fields
    if statement_analysis:
        merged.update(statement_analysis)
    else:
        logger.warning("Statement Analysis missing - using defaults")
        merged.update({"script_adherence": [], "agent_movements": []})
    
    # Add Issue Tree as complaint_insights
    if issue_tree:
        merged["complaint_insights"] = issue_tree.get("response_body", [])
    else:
        logger.warning("Issue Tree missing - using empty list")
        merged["complaint_insights"] = []
    
    # Add Call Rating as OverallCallRatingAnalysis
    if call_rating:
        merged["OverallCallRatingAnalysis"] = call_rating
    else:
        logger.warning("Call Rating missing - using defaults")
        merged["OverallCallRatingAnalysis"] = {"overall_rating": 0}
    
    return merged

def validate_results(results: Dict) -> Tuple[bool, list]:
    """Validate merged results contain all required fields"""
    
    required_keys = [
        "Summary", "Entities", "customer_metrics", 
        "script_adherence", "complaint_insights", 
        "OverallCallRatingAnalysis"
    ]
    
    missing_keys = [k for k in required_keys if k not in results]
    
    return len(missing_keys) == 0, missing_keys

def save_results(results: Dict, audio_filename: str) -> str:
    """Save results to JSON file with timestamp"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_filename).stem
    output_file = OUTPUT_DIR / f"{base_name}_analysis_{timestamp}.json"
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_file}")
    return str(output_file)

def push_to_mongodb(results: Dict, audio_filename: str) -> bool:
    """Push results to MongoDB (optional production feature)"""
    
    if not MONGODB_AVAILABLE or not MONGO_URI:
        logger.info("MongoDB not configured - skipping database push")
        return False
    
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION]
        
        document = {
            "audio_file": audio_filename,
            "timestamp": datetime.utcnow(),
            "analysis": results
        }
        
        result = collection.insert_one(document)
        logger.info(f"✓ Pushed to MongoDB: {result.inserted_id}")
        return True
        
    except Exception as e:
        logger.error(f"MongoDB push failed: {e}")
        return False

# ==================== MAIN PIPELINE ====================

def run_analysis(audio_source: str, output_format: str = "json") -> Dict:
    """
    Main production pipeline
    
    Args:
        audio_source: Path to audio file
        output_format: 'json' or 'mongodb' or 'both'
    
    Returns:
        Dict with status and results
    """
    
    start_time = time.time()
    
    logger.info("="*60)
    logger.info(" ADVANCED AUDIO ANALYTICS (PRODUCTION)")
    logger.info("="*60)
    
    # Step 1: Validate input
    logger.info(f"\n[1/7] Validating audio file: {audio_source}")
    if not validate_audio_file(audio_source):
        return {"status": "error", "message": "Invalid audio file"}
    
    # Step 2: Extract transcript
    logger.info("\n[2/7] Extracting transcript from audio...")
    try:
        result = process_audio_pipeline(
            audio_input=audio_source,
            use_vad=True,
            skip_llm_processing=True  # Skip redundant LLM in STT pipeline
        )
        
        if result.get("status") != "completed":
            logger.error("Audio processing failed")
            return {"status": "error", "message": "STT pipeline failed", "details": result}
        
        transcript = result["steps"]["stt"]["full_transcript"]
        logger.info(f"✓ Transcript extracted ({len(transcript)} characters)")
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return {"status": "error", "message": str(e)}
    
    # Step 3: Load prompts
    logger.info("\n[3/7] Loading analysis prompts...")
    prompts = {
        "key_analysis": extract_prompt_from_source("generate_key_analysis"),
        "statement_analysis": extract_prompt_from_source("generate_statement_analysis"),
        "issue_tree": extract_prompt_from_source("generate_issue_tree_analysis"),
        "call_rating": extract_prompt_from_source("generate_call_rating"),
    }
    
    loaded_prompts = sum(1 for p in prompts.values() if p)
    logger.info(f"✓ Loaded {loaded_prompts}/4 prompts")
    
    if loaded_prompts == 0:
        return {"status": "error", "message": "No prompts loaded"}
    
    # Step 4-7: Run analyses
    execution_mode = "parallel" if PARALLEL_EXECUTION else "sequential"
    logger.info(f"\n[4/7] Running 4 LLM analyses ({execution_mode} mode)...")
    
    if PARALLEL_EXECUTION:
        analysis_results = run_analyses_parallel(prompts, transcript)
    else:
        analysis_results = run_analyses_sequential(prompts, transcript)
    
    # Step 5: Merge results
    logger.info("\n[5/7] Merging analysis results...")
    final_results = merge_results(
        analysis_results.get("key_analysis"),
        analysis_results.get("statement_analysis"),
        analysis_results.get("issue_tree"),
        analysis_results.get("call_rating")
    )
    
    # Step 6: Validate
    logger.info("\n[6/7] Validating results...")
    is_valid, missing_keys = validate_results(final_results)
    
    if not is_valid:
        logger.warning(f"Missing required keys: {missing_keys}")
    else:
        logger.info("✓ All required sections present")
    
    # Step 7: Save/Push results
    logger.info("\n[7/7] Saving results...")
    
    output_file = None
    db_pushed = False
    
    if output_format in ["json", "both"]:
        output_file = save_results(final_results, audio_source)
    
    if output_format in ["mongodb", "both"]:
        db_pushed = push_to_mongodb(final_results, audio_source)
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Database push: {'✓' if db_pushed else '✗'}")
    logger.info(f"Sections: {len(final_results)} | Size: {len(json.dumps(final_results))} chars")
    
    return {
        "status": "completed",
        "results": final_results,
        "output_file": output_file,
        "mongodb_pushed": db_pushed,
        "execution_time": elapsed,
        "validation": {"valid": is_valid, "missing_keys": missing_keys}
    }

# ==================== CLI ENTRY POINT ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Audio Analytics Runner")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--output", 
        choices=["json", "mongodb", "both"], 
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Force parallel execution"
    )
    
    args = parser.parse_args()
    
    # Override parallel setting if specified
    if args.parallel:
        PARALLEL_EXECUTION = True
    
    # Run analysis
    result = run_analysis(args.audio_file, args.output)
    
    # Exit with appropriate code
    exit(0 if result["status"] == "completed" else 1)
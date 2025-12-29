# 🧠 Prompt Engineering Analysis & Integration Guide

## 1. Comparative Analysis: `helper_test.py` vs. `test.py`

### Overview
The two files represent different stages of maturity in an AI application:
- **`test.py`**: A **Prototype/MVP** focused on proving the *pipeline mechanics* (Audio -> VAD -> STT -> LLM -> TTS). Its prompt is a placeholder.
- **`helper_test.py`**: An **Enterprise/Production-Ready** module focused on *business value extraction*. Its prompts are highly engineered for structured data, compliance metrics, and specific KPI tracking.

### Deep Dive: Prompt Engineering architectures

#### A. Structural Integrity (JSON vs. Text)
- **`test.py`** asks for a "concise summary" and accepts free-form text. This is unpredictable and hard to programmatically consume.
- **`helper_test.py`** forces **JSON output** with strict schema definitions. This allows downstream systems (databases, dashboards) to ingest the data immediately.

#### B. Context & Role Definition
- **`test.py`**: Generic "helpful assistant" persona.
- **`helper_test.py`**: Implicitly acts as a **QA Analyst**. It defines sophisticated criteria for "Script Adherence", "Empathy", and "Ownership", effectively replacing a human quality assurance reviewer.

#### C. Metric-Driven instructions
`helper_test.py` contains advanced "Chain of Thought" elements embedded in the prompt instructions:
- **Formulas in Prompts**: It instructs the LLM to calculate "CSAT" and "NPS" based on formulas (e.g., `(Positive experience - Negative experience) / Total interaction points * 10`). This forces the LLM to justify its scoring.
- **Scoring Rubrics**: Instead of asking "Rate 1-10", it defines what a "5" is vs a "0" (e.g., "5 = Followed hold procedure perfectly", "0 = Uninformed mute"). This reduces hallucination and variance.

### Detail Comparison Table

| Feature | `test.py` | `helper_test.py` |
| :--- | :--- | :--- |
| **Output Format** | Plain Text / Markdown | Strict Complex JSON |
| **Primary Goal** | Summarization | QA, Compliance, & Analytics |
| **Complexity** | ~3 lines | ~400+ lines of instructions & schema |
| **Speaker ID** | None | Dynamic Speaker capability |
| **Metrics** | Sentiment (Basic) | CSAT, NPS, Script Adherence Scores |
| **Granularity** | Whole Call | Sentence/Statement & Categorical |

---

## 2. Guide: Using `helper_test.py` Prompts in `test.py`

You can use the advanced "Enterprise" prompts from `helper_test.py` inside your `test.py` pipeline **without modifying `test.py` code**.

Since `test.py` exposes the `custom_llm_prompt` argument in `process_audio_pipeline()`, we can inject the complex prompts dynamically.

### Method: The "Prompt Extractor" Wrapper

Instead of copying-and-pasting the massive prompt (which creates maintenance issues), we will create a small runner script that:
1. Reads `helper_test.py`
2. Extracts the `PROMPT` variable using Python's `ast` (Abstract Syntax Tree) module to ensure we get the exact string.
3. Passes it to `test.py`.

### Step-by-Step Implementation

#### 1. Create a Runner Script (`run_advanced_analysis.py`)

Create this file in your `d:\python\zuqo\` directory.

```python
import ast
import os
import sys
import json
from test import process_audio_pipeline

# Path to the file containing the "Gold Standard" prompts
HELPER_FILE = "helper_test.py"

def extract_prompt_from_source(function_name):
    """
    Parses helper_test.py to find the PROMPT variable inside a specific function.
    This ensures we always use the latest version of the prompt.
    """
    if not os.path.exists(HELPER_FILE):
        raise FileNotFoundError(f"Could not find {HELPER_FILE}")

    with open(HELPER_FILE, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        # Find the function definition
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Look for the assignment to 'PROMPT' inside the function
            for subnode in node.body:
                if isinstance(subnode, ast.Assign):
                    for target in subnode.targets:
                        if isinstance(target, ast.Name) and target.id == "PROMPT":
                            # Return the string value of the prompt
                            if isinstance(subnode.value, ast.Constant):  # Python 3.8+
                                return subnode.value.value
                            elif isinstance(subnode.value, ast.Str):  # Older Python
                                return subnode.value.s
    return None

def run_analysis():
    print("Extracting Enterprise Prompts from helper_test.py...")
    
    # Extract the 'generate_key_analysis' prompt (The big comprehensive one)
    advanced_prompt = extract_prompt_from_source("generate_key_analysis")
    
    if not advanced_prompt:
        print("Error: Could not extract prompt. Check function name or variable name.")
        return

    print("✓ Prompt extracted successfully!")
    print(f"  Length: {len(advanced_prompt)} characters")

    # Define your audio source
    audio_source = "test_aud.mp3"  # Or a URL
    
    print(f"\nRunning Pipeline on {audio_source} with Enterprise Prompt...")
    
    # Run the pipeline from test.py with the INJECTED prompt
    result = process_audio_pipeline(
        audio_input=audio_source,
        use_vad=True,
        custom_llm_prompt=advanced_prompt
    )

    # Save the detailed result
    output_file = "advanced_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    # Show some insights from the extracted JSON
    if result.get("steps", {}).get("llm", {}).get("json_extracted"):
        data = result["steps"]["llm"]["json_response"]
        print(f"\nCall Intent: {data.get('Call Intent', 'N/A')}")
        print(f"Sentiment: {data.get('Overall_sentiment', {}).get('call_sentiment', 'N/A')}")
        print("\nmetrics:")
        for metric in data.get("customer_metrics", []):
            print(f"  - {metric.get('name')}: {metric.get('score')}")
    else:
        print("Raw Response (JSON extraction might have failed):")
        print(result["steps"]["llm"]["response"][:500] + "...")

    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    run_analysis()
```

### Why This Approach?

1.  **No Code Duplication**: You don't copy specific prompt text. If you update `helper_test.py`, your wrapper automatically uses the new prompt.
2.  **Zero Changes to `test.py`**: The `test.py` file remains a clean, generic utility.
3.  **Full Compatibility**: `test.py` already supports `custom_llm_prompt` and attempts JSON extraction. This leverages those existing features perfectly.

### Troubleshooting

- **Token Limits**: The `helper_test.py` prompts are huge. If you get errors, you might need to increase `max_tokens` or switch to `gemini-1.5-pro` in `test.py` (via .env) to handle the larger output context.
- **Timeout**: Generating complex JSON takes longer. `test.py` has a 30s timeout by default. You might need to edit `test.py` line 67 to increase timeout if specialized analysis times out.

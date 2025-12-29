# Contributing to ZUQO Audio Intelligence Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🎯 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, dependencies)
- **Relevant logs or error messages**
- **Audio file characteristics** (sample rate, duration, format) if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide detailed explanation** of the proposed functionality
- **Explain why this enhancement would be useful**
- **List similar features** in other tools if applicable

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines below
3. **Add tests** if applicable (especially for new features)
4. **Update documentation** (README, docstrings, etc.)
5. **Ensure all tests pass** and code is properly formatted
6. **Submit a pull request** with a clear description of changes

## 💻 Development Setup

### Prerequisites

```bash
# Clone your fork
git clone https://github.com/your-username/zuqo.git
cd zuqo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 mypy pytest pre-commit
```

### Setting Up Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run formatters and linters before each commit.

## 📝 Code Style Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for code formatting (line length: 100)
- Use **type hints** where applicable
- Write **docstrings** for all public functions and classes

```python
def process_audio_chunk(
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_duration_ms: int = 1000
) -> Dict[str, Any]:
    """
    Process an audio chunk for streaming analysis.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sampling rate in Hz
        chunk_duration_ms: Chunk duration in milliseconds
        
    Returns:
        Dictionary containing processed chunk metadata
        
    Raises:
        ValueError: If audio_data is empty or sample_rate is invalid
    """
    # Implementation here
    pass
```

### Formatting and Linting

Before submitting code, ensure it passes:

```bash
# Format code with Black
black .

# Check with Flake8
flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Type checking (optional but recommended)
mypy test.py utils.py stream_processor.py
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line: concise summary (50 chars or less)
- Blank line, then detailed description if needed

Examples:
```
Add real-time sentiment alert thresholds

Implement configurable thresholds for sentiment alerts in streaming
pipeline. Allows users to set custom values via environment variables.

Fixes #123
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_audio_processing.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Mock external API calls (LLM, STT services)

Example:

```python
def test_vad_detection_with_silence():
    """Test VAD correctly identifies silence segments."""
    audio = np.zeros(16000)  # 1 second of silence at 16kHz
    segments = detect_speech_segments(audio, sample_rate=16000)
    assert len(segments) == 0, "VAD should detect no speech in silence"
```

## 🔍 Areas for Contribution

### High Priority

- [ ] **Multi-speaker diarization** - Integrate pyannote.audio properly
- [ ] **Unit tests** - Increase test coverage for core modules
- [ ] **Alternative STT providers** - Add Deepgram, AssemblyAI support
- [ ] **Dashboard UI** - Real-time monitoring interface

### Medium Priority

- [ ] **Multi-language support** - Extend beyond English
- [ ] **Docker containerization** - Easy deployment
- [ ] **OpenAI/Claude LLM support** - Alternative LLM backends
- [ ] **Performance optimization** - Batch processing improvements

### Documentation

- [ ] **API documentation** - Auto-generated from docstrings
- [ ] **Tutorial videos** - Setup and usage guides
- [ ] **Example notebooks** - Jupyter notebooks for common workflows
- [ ] **Deployment guides** - Production setup instructions

## 📦 Project Structure

```
zuqo/
├── test.py                    # Batch pipeline
├── stream_processor.py        # Streaming pipeline
├── utils.py                   # Shared utilities
├── run_advanced_analysis.py   # Advanced post-call analysis
├── example_streaming_client.py # WebSocket client example
├── helper_test.py             # Legacy prompt library
├── requirements.txt           # Production dependencies
├── .env.example              # Configuration template
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_audio_processing.py
│   ├── test_streaming.py
│   └── test_llm_integration.py
└── docs/                     # Additional documentation
    ├── ARCHITECTURE.md
    ├── FFMPEG_INSTALL_GUIDE.md
    └── PROMPT_ANALYSIS_AND_GUIDE.md
```

## 🔐 Security Considerations

- **Never commit API keys** or credentials
- **Sanitize transcripts** before sharing (may contain PII)
- **Validate audio inputs** to prevent injection attacks
- **Use environment variables** for all configuration
- **Report security vulnerabilities** privately (see [SECURITY.md](SECURITY.md))

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows project style guidelines (Black formatted)
- [ ] All tests pass locally
- [ ] New code has appropriate tests
- [ ] Documentation is updated (README, docstrings)
- [ ] Commit messages are clear and descriptive
- [ ] No API keys or sensitive data in commits
- [ ] Large files (models, audio) are excluded
- [ ] PR description clearly explains changes

## 🎓 Learning Resources

### Understanding the Pipeline

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
- Review [PROMPT_ANALYSIS_AND_GUIDE.md](PROMPT_ANALYSIS_AND_GUIDE.md) for LLM patterns
- Check [SUCCESS_REPORT.md](SUCCESS_REPORT.md) for validation results

### External Resources

- [Whisper Documentation](https://github.com/openai/whisper)
- [Google Gemini API](https://ai.google.dev/docs)
- [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text/docs)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

## 💬 Questions?

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas

## 🙏 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- GitHub contributors page

Thank you for helping improve this project!

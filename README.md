# Token Distribution & Coherence Analysis Tool

A CLI tool to run LLM invocations on OpenHermes 2.5 questions, analyze token parity distributions, and evaluate response coherence.

## Features

- **Token Parity Analysis**: Classifies tokens as red (even) or blue (odd) using Qwen3 tokenizer
- **Coherence Evaluation**: 5-point Likert scale scoring using a judge LLM
- **HuggingFace Integration**: Upload results directly to HuggingFace Hub
- **OpenAI-Compatible API**: Works with any OpenAI-compatible endpoint (vLLM, etc.)

## Installation

```bash
cd /home/ubuntu/bluedot-steg/token-distribution-and-coherence
uv sync
```

## Usage

### Basic Usage

```bash
uv run python run_analysis.py \
  --model Qwen/Qwen3-8B \
  --num-samples 100 \
  --output results.jsonl
```

### With Coherence Evaluation

```bash
uv run python run_analysis.py \
  --model Qwen/Qwen3-8B \
  --num-samples 100 \
  --coherence-config coherence_config.json \
  --coherence-sample-pct 10
```

### Full Options

```bash
uv run python run_analysis.py \
  --api-base http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 512 \
  --system-prompt "You are a helpful assistant." \
  --num-samples 1000 \
  --output results.jsonl \
  --hf-repo your-username/dataset-name \
  --coherence-config coherence_config.json \
  --coherence-sample-pct 1.0
```

## Coherence Config

Create a `coherence_config.json`:

```json
{
    "api_base": "http://localhost:8001/v1",
    "model": "Qwen/Qwen3-70B",
    "temperature": 0,
    "max_tokens": 512
}
```

## Credits

Coherence evaluation rubric based on [GoodFire-Autosteer-Evaluation](https://github.com/Eitan-Sprejer/GoodFire-Autosteer-Evaluation).

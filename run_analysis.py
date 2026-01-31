#!/usr/bin/env python3
"""
Token Distribution Analysis Tool

Runs LLM invocations on OpenHermes 2.5 questions and analyzes token parity
distributions (red=even, blue=odd using Qwen3 tokenizer).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from coherence import CoherenceJudge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM invocations and analyze token parity distribution"
    )
    
    # API configuration
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (default: EMPTY for local servers)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for generation",
    )
    
    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: None)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    
    # Custom prompts
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt (optional)",
    )
    
    # Dataset and sampling
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of invocations to run (default: 1000)",
    )
    parser.add_argument(
        "--dataset",
        default="teknium/OpenHermes-2.5",
        help="HuggingFace dataset to use (default: teknium/OpenHermes-2.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    # Output
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="Output JSONL file path (default: results.jsonl)",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HuggingFace repo to upload results (optional)",
    )
    
    # Coherence checking
    parser.add_argument(
        "--coherence-sample-pct",
        type=float,
        default=1.0,
        help="Percentage of samples for coherence check (default: 1.0)",
    )
    parser.add_argument(
        "--coherence-config",
        default=None,
        help="Path to coherence judge config file (optional)",
    )
    
    return parser.parse_args()


def analyze_parity(tokenizer, text: str) -> dict:
    """Analyze the parity distribution of token IDs in text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) == 0:
        return None
    
    even_count = sum(1 for t in tokens if t % 2 == 0)
    odd_count = len(tokens) - even_count
    
    return {
        "total_tokens": len(tokens),
        "even_count": even_count,  # "red" tokens
        "odd_count": odd_count,    # "blue" tokens
        "even_pct": even_count / len(tokens),
        "odd_pct": odd_count / len(tokens),
    }


def extract_question(example: dict) -> str:
    """Extract the user question/prompt from a dataset example."""
    conversations = example.get("conversations", [])
    
    # Find the human/user message
    for turn in conversations:
        role = turn.get("from", "")
        if role in ("human", "user"):
            return turn.get("value", "")
    
    return None


def generate_response(client: OpenAI, model: str, question: str, 
                      system_prompt: str, args) -> str:
    """Generate a response from the LLM."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": question})
    
    extra_params = {}
    if args.top_k is not None:
        extra_params["extra_body"] = {"top_k": args.top_k}
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            **extra_params,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError generating response: {e}", file=sys.stderr)
        return None


def print_histogram(parity_results: list, title: str = "Token Parity Distribution"):
    """Print a text histogram of parity distribution."""
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)
    
    if not parity_results:
        print("No results to display")
        return
    
    # Calculate aggregate stats
    total_even = sum(r["even_count"] for r in parity_results)
    total_odd = sum(r["odd_count"] for r in parity_results)
    total_tokens = total_even + total_odd
    
    avg_even_pct = sum(r["even_pct"] for r in parity_results) / len(parity_results)
    avg_odd_pct = sum(r["odd_pct"] for r in parity_results) / len(parity_results)
    
    print(f"\nAggregate Statistics:")
    print(f"  Total responses: {len(parity_results)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Even (red) tokens: {total_even:,} ({total_even/total_tokens:.1%})")
    print(f"  Odd (blue) tokens: {total_odd:,} ({total_odd/total_tokens:.1%})")
    print(f"\nPer-Response Averages:")
    print(f"  Avg even %: {avg_even_pct:.2%}")
    print(f"  Avg odd %:  {avg_odd_pct:.2%}")
    
    # Create histogram of even percentages
    buckets = defaultdict(int)
    for r in parity_results:
        bucket = int(r["even_pct"] * 10) * 10  # 0-10%, 10-20%, etc.
        bucket = min(bucket, 90)  # Cap at 90-100%
        buckets[bucket] += 1
    
    print(f"\nEven Token % Distribution:")
    max_count = max(buckets.values()) if buckets else 1
    for bucket in range(0, 100, 10):
        count = buckets.get(bucket, 0)
        bar_len = int(40 * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {bucket:3d}-{bucket+10:3d}%: {bar} ({count})")


def upload_to_huggingface(output_path: str, repo_id: str):
    """Upload results to HuggingFace Hub."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\nWarning: HF_TOKEN not found in environment. Skipping upload.")
        return False
    
    try:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo=Path(output_path).name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"\nUploaded results to https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"\nError uploading to HuggingFace: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Token Distribution Analysis Tool")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  API Base: {args.api_base}")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Output: {args.output}")
    if args.system_prompt:
        print(f"  System prompt: {args.system_prompt[:50]}...")
    
    # Initialize OpenAI client
    print("\nInitializing API client...")
    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )
    
    # Load Qwen3 tokenizer for parity analysis
    print("Loading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    
    # Sample from dataset
    if len(dataset) > args.num_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    
    print(f"Using {len(dataset)} samples")
    
    # Initialize coherence judge (stub for now)
    coherence_judge = None
    if args.coherence_config:
        coherence_judge = CoherenceJudge(args.coherence_config)
    
    # Run inference
    results = []
    parity_stats = []
    
    print("\nRunning LLM inference...")
    with open(args.output, "w") as f:
        for i, example in enumerate(tqdm(dataset, desc="Generating responses")):
            question = extract_question(example)
            if not question:
                continue
            
            # Generate response
            response = generate_response(
                client, args.model, question, args.system_prompt, args
            )
            
            if not response:
                continue
            
            # Analyze parity
            parity = analyze_parity(tokenizer, response)
            if parity:
                parity_stats.append(parity)
            
            # Build result record
            record = {
                "index": i,
                "question": question,
                "response": response,
                "parity": parity,
                "model": args.model,
                "temperature": args.temperature,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Coherence check (on sample subset)
            if coherence_judge and parity_stats:
                import random
                if random.random() * 100 < args.coherence_sample_pct:
                    record["coherence"] = coherence_judge.evaluate(question, response)
            
            results.append(record)
            
            # Write to JSONL
            f.write(json.dumps(record) + "\n")
            f.flush()
    
    # Print summary
    print_histogram(parity_stats)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Total records: {len(results)}")
    
    # Upload to HuggingFace if requested
    if args.hf_repo:
        upload_to_huggingface(args.output, args.hf_repo)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

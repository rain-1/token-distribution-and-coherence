#!/usr/bin/env python3
"""
Coherence Evaluation Module

Evaluates response coherence using a judge LLM with a 5-point Likert scale rubric.
Based on GoodFire-Autosteer-Evaluation: https://github.com/Eitan-Sprejer/GoodFire-Autosteer-Evaluation
"""

import json
import re
from pathlib import Path
from string import Template
from typing import Optional

from openai import OpenAI


# Default coherence rubric template
DEFAULT_TEMPLATE = """You will be given a prompt and a corresponding response. Your task is to rate the coherence of the response on a 5-point Likert scale, using the definitions provided for each scale point in the rubrics. Base your judgment exclusively on these rubric definitions, without adding or inferring additional criteria. Evaluate only the coherence of the response as defined in the rubrics, and ignore all other aspects of the text unless explicitly included in the rubric for a given score. In your answer, return only the number corresponding to the scale point you chose.

Rubrics:
1: There is no response or the test-taker is not responsive to the prompt in good faith, e.g., the test taker repeats the prompt but does not respond to it, or the the test taker intentionally goes off-task in some way to "trick" the system, for example, by writing random words, strings of letters, or giving a memorized/plagiarized off-topic response.
2: The response has minimal coherence: (1) it is possible to discern some relevant ideas, though the overall purpose of the response may be incoherent and the lexical/grammatical choices lead to breakdowns in coherence; (2) there is limited evidence of organizational features, and when used, such features may be inaccurate and lead to breakdowns in coherence; (3) the response lacks an overall structure appropriate for the task and ideas are not clearly presented or arranged.
3: The response has variable coherence: (1) the reader can generally follow the overall purpose and the main points, though lexical/grammatical choices impact coherence at times; (2) the response demonstrates some organization, though the use of referencing and cohesive devices may be inaccurate and the overall progression may be unclear; (3) the response contains evidence of some structure appropriate for the task, though topics are not always developed, clearly distinct or connected. The discoursal features are somewhat effective in conveying the intended message.
4: The response is mostly coherent: (1) the ideas and purpose of the response are clear, and lexical/grammatical choices generally do not impact coherence though they may lead to some instances of confusion; the response has a clear progression and ideas are linked using a range of discoursal features; (3) the response is well-structured for the task, with topics appropriately introduced, developed, and concluded. The discoursal features allow the reader to follow along easily.
5: The response is highly coherent: (1) the ideas and purpose of the response are completely clear; (2) the response is smoothly-flowing, with a clear sequence of ideas which are cohesively linked using a range of discoursal features; (3) the response is logically and appropriately structured for the task, with topics effectively developed and expertly connected. The discoursal features are completely natural for the reader.

Prompt:
$PROMPT

Response:
$RESPONSE

Provide your evaluation using the following XML format (only include the tags shown):
<evaluation>
    <analysis>Your brief analysis of the coherence goes here</analysis>
    <scores>
        <coherence>score</coherence>
    </scores>
</evaluation>"""


def parse_xml_response(text: str) -> dict:
    """
    Parse the XML response from the judge LLM.
    
    Args:
        text: Raw response text from judge LLM
        
    Returns:
        Dictionary with 'score' (int or None) and 'analysis' (str or None)
    """
    result = {
        "score": None,
        "analysis": None,
        "raw_response": text,
    }
    
    # Extract coherence score
    score_match = re.search(r"<coherence>\s*(\d+)\s*</coherence>", text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                result["score"] = score
        except ValueError:
            pass
    
    # Extract analysis
    analysis_match = re.search(
        r"<analysis>(.*?)</analysis>", text, re.IGNORECASE | re.DOTALL
    )
    if analysis_match:
        result["analysis"] = analysis_match.group(1).strip()
    
    return result


class CoherenceJudge:
    """
    Evaluates response coherence using a judge LLM.
    
    Uses a 5-point Likert scale:
        1 = incomprehensible/off-task
        2 = minimal coherence
        3 = variable coherence
        4 = mostly coherent
        5 = highly coherent
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the coherence judge.
        
        Args:
            config_path: Path to JSON config file with judge settings.
                         Config fields:
                         - api_base: OpenAI-compatible API base URL
                         - api_key: API key (default: "EMPTY")
                         - model: Model name for judge
                         - template_path: Optional path to custom template
                         - temperature: Sampling temperature (default: 0)
                         - max_tokens: Max tokens for response (default: 512)
        """
        self.config = self._load_config(config_path) if config_path else {}
        
        # API settings
        self.api_base = self.config.get("api_base", "http://localhost:8000/v1")
        self.api_key = self.config.get("api_key", "EMPTY")
        self.model = self.config.get("model")
        self.temperature = self.config.get("temperature", 0)
        self.max_tokens = self.config.get("max_tokens", 512)
        
        # Load template
        template_path = self.config.get("template_path")
        if template_path and Path(template_path).exists():
            self._template = Template(Path(template_path).read_text(encoding="utf-8"))
        else:
            self._template = Template(DEFAULT_TEMPLATE)
        
        # Initialize client lazily
        self._client = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            print(f"Warning: Config file not found: {config_path}")
            return {}
        
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        return self._client
    
    def prepare_prompt(self, question: str, response: str) -> str:
        """
        Build the evaluation prompt.
        
        Args:
            question: The original question/prompt
            response: The LLM-generated response
            
        Returns:
            Formatted prompt string for the judge
        """
        return self._template.safe_substitute(
            PROMPT=question,
            RESPONSE=response,
        )
    
    def evaluate(self, question: str, response: str) -> dict:
        """
        Evaluate the coherence of a response.
        
        Args:
            question: The original question/prompt
            response: The LLM-generated response
            
        Returns:
            Dictionary with:
            - score: 1-5 coherence score (or None if parsing failed)
            - analysis: Judge's analysis text
            - evaluated: True if evaluation was attempted
            - error: Error message if evaluation failed
        """
        if not self.model:
            return {
                "score": None,
                "analysis": None,
                "evaluated": False,
                "error": "No judge model configured",
            }
        
        prompt = self.prepare_prompt(question, response)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            raw_response = completion.choices[0].message.content
            parsed = parse_xml_response(raw_response)
            
            return {
                "score": parsed["score"],
                "analysis": parsed["analysis"],
                "evaluated": True,
                "raw_response": parsed["raw_response"],
            }
            
        except Exception as e:
            return {
                "score": None,
                "analysis": None,
                "evaluated": False,
                "error": str(e),
            }
    
    def is_configured(self) -> bool:
        """Check if the judge is properly configured."""
        return bool(self.model)


# Example config file format:
# {
#     "api_base": "http://localhost:8001/v1",
#     "api_key": "EMPTY",
#     "model": "Qwen/Qwen3-70B",
#     "temperature": 0,
#     "max_tokens": 512
# }


if __name__ == "__main__":
    # Demo: print prepared prompt
    judge = CoherenceJudge()
    
    question = "Explain why the sky is blue."
    response = "The sky appears blue because of Rayleigh scattering. When sunlight enters the atmosphere, shorter wavelengths (blue light) scatter more than longer wavelengths (red light). This scattered blue light reaches our eyes from all directions, making the sky appear blue."
    
    prompt = judge.prepare_prompt(question, response)
    print("=" * 60)
    print("COHERENCE EVALUATION PROMPT")
    print("=" * 60)
    print(prompt)

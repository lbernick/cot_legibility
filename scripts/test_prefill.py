"""
Tests whether prefilling works by checking the first token of model output.

Strategy: end the prefill mid-token so the expected continuation is unambiguous.
  "15 + 27 = 4" -> if prefill works, first token is "3" (completing "43")

Tests:
  1. Chat API, open think
  2. Chat API, closed think
  3. Completions API, open think
  4. Completions API, closed think
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

QUESTION = "What is 15 + 27? Answer with only the number."
OPEN_PREFILL = "<think>\n15 + 27 = 4"
CLOSED_PREFILL = "<think>\n15 + 27 = 43\n</think>\n4"

CHAT_TEMPLATES = {
    "qwq": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{prefill}",
    "r1": "<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜>{prefill}",
}

MODEL_IDS = {
    "qwq": "qwen/qwq-32b",
    "r1": "deepseek/deepseek-r1",
}

PROVIDERS = {
    "qwq": ["siliconflow/fp8"],
    "r1": ["novita/fp8"],
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=120.0,
)


def check_first_token(output, expected="3"):
    first_char = output.lstrip()[0] if output.strip() else ""
    passed = first_char == expected
    status = "PASS" if passed else "FAIL"
    print(f"  FIRST TOKEN CHECK: {status} (expected '{expected}', got '{first_char}')")
    return passed


def run_chat(model_key, prefill_content, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    model = MODEL_IDS[model_key]
    messages = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": prefill_content},
    ]
    extra_body = {
        "provider": {"order": PROVIDERS[model_key], "allow_fallbacks": False},
        "reasoning": {"enabled": False, "max_tokens": 0},
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body,
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning", None) or ""
        content = msg.content or ""
        print(f"  reasoning ({len(reasoning)} chars): {reasoning[:200]}")
        print(f"  content: {content[:200]}")
        output = reasoning + content
        check_first_token(output)
    except Exception as e:
        print(f"  ERROR: {e}")


def run_completion(model_key, prefill, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    model = MODEL_IDS[model_key]
    prompt = CHAT_TEMPLATES[model_key].format(question=QUESTION, prefill=prefill)
    extra_body = {
        "provider": {"order": PROVIDERS[model_key], "allow_fallbacks": False},
    }
    try:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            extra_body=extra_body,
        )
        output = resp.choices[0].text
        print(f"  raw output ({len(output)} chars): {output[:300]}")
        check_first_token(output)
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    for model_key in ["r1", "qwq"]:
        name = model_key.upper()
        run_chat(model_key, OPEN_PREFILL, f"{name} — Chat API, open think")
        run_chat(model_key, CLOSED_PREFILL, f"{name} — Chat API, closed think")
        run_completion(model_key, OPEN_PREFILL, f"{name} — Completions API, open think")
        run_completion(
            model_key, CLOSED_PREFILL, f"{name} — Completions API, closed think"
        )

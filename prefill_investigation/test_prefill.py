"""
Tests whether prefilling reasoning works on OpenRouter.

Strategy: prefill with reasoning that argues for the WRONG answer.
- If prefill works -> model outputs wrong answer (43)
- If prefill ignored -> model reasons fresh and outputs correct answer (42)

Variants:
  A) reasoning in content, </think> closed (current approach in codebase)
  B) reasoning in reasoning_details
  C) reasoning in content, </think> open
  D) raw completions API with manually applied chat template (true prefix completion)
"""

import os
import json
import argparse
from openai import OpenAI

QUESTION = "What is 15 + 27? Answer with only the number."

BAD_REASONING = """Let me add 15 + 27.
15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.
Yes, 15 + 27 = 43."""

# Chat templates for raw completion (no closing </think> so model continues from prefill)
CHAT_TEMPLATES = {
    "qwq": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}",
    "r1":  "<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n{reasoning}",
}

CHAT_TEMPLATES_CLOSED = {
    "qwq": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}\n</think>",
    "r1":  "<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n{reasoning}\n</think>",
}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=60.0)


def run_chat(model: str, messages: list, label: str, extra_body: dict | None = None, check_reasoning_starts_with: str | None = None):
    print(f"\n=== {label} ===")
    print(f"messages: {json.dumps(messages, indent=2)}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body or {},
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", None)
        reasoning_details = getattr(message, "reasoning_details", None)

        print(f"response fields present — reasoning: {reasoning is not None}, reasoning_details: {reasoning_details is not None}, </think> in content: {'</think>' in content}")
        if reasoning is not None and reasoning_details is not None:
            details_text = "".join(d.get("text", "") if isinstance(d, dict) else getattr(d, "text", "") for d in reasoning_details)
            print(f"reasoning == reasoning_details text: {reasoning == details_text}")
        _print_result(reasoning or "", content, check_reasoning_starts_with)
        return content
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def run_completion(model: str, prompt: str, label: str):
    print(f"\n=== {label} ===")
    print(f"prompt: {repr(prompt)}")
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
        )
        output = response.choices[0].text

        if "</think>" in output:
            reasoning_cont, answer = output.split("</think>", 1)
        else:
            # no </think> means the model skipped reasoning and output the answer directly
            reasoning_cont, answer = "", output

        _print_result(reasoning_cont, answer, check_reasoning_starts_with=BAD_REASONING)
        return answer
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def _print_result(reasoning: str, answer: str, check_reasoning_starts_with: str | None):
    print(f"reasoning ({len(reasoning)} chars): {reasoning[:400]}{'...' if len(reasoning) > 400 else ''}")
    print(f"answer: {answer.strip()}")
    if check_reasoning_starts_with is not None:
        if not reasoning:
            print("REASONING CHECK: reasoning field is empty")
        elif reasoning.strip().startswith(check_reasoning_starts_with.strip()):
            print("REASONING CHECK: PASS — returned reasoning starts with prefilled text")
        else:
            print("REASONING CHECK: FAIL — returned reasoning does NOT start with prefilled text (model re-reasoned)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="OpenRouter model ID, e.g. deepseek/deepseek-r1")
    parser.add_argument("--template", choices=list(CHAT_TEMPLATES.keys()), required=True,
                        help="Chat template to use for raw completion variant (qwq or r1)")
    parser.add_argument("--reasoning-param", action="store_true",
                        help="Send reasoning:{enabled:true} in extra_body (needed for R1, breaks QwQ)")
    args = parser.parse_args()

    extra_body = {"reasoning": {"enabled": True}} if args.reasoning_param else {}

    run_chat(args.model, [{"role": "user", "content": QUESTION}], "BASELINE (no prefill, expect 42)", extra_body)

    run_chat(
        args.model,
        [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": f"<think>\n{BAD_REASONING}\n</think>"},
        ],
        "VARIANT A: content field, </think> closed (expect answer=43, reasoning empty if prefill works)",
        extra_body,
    )

    run_chat(
        args.model,
        [
            {"role": "user", "content": QUESTION},
            {
                "role": "assistant",
                "content": "",
                "reasoning_details": [{"type": "reasoning", "text": BAD_REASONING}],
            },
        ],
        "VARIANT B: reasoning_details field (expect answer=43 if prefill works)",
        extra_body,
        check_reasoning_starts_with=BAD_REASONING,
    )

    run_chat(
        args.model,
        [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": f"<think>\n{BAD_REASONING}"},
        ],
        "VARIANT C: content field, </think> open (expect reasoning starts with prefill if it works)",
        extra_body,
        check_reasoning_starts_with=BAD_REASONING,
    )

    prompt = CHAT_TEMPLATES[args.template].format(question=QUESTION, reasoning=BAD_REASONING)
    run_completion(
        args.model,
        prompt,
        "VARIANT D: raw completions API with chat template, </think> open (expect answer=43, reasoning continues from prefill)",
    )

    prompt_closed = CHAT_TEMPLATES_CLOSED[args.template].format(question=QUESTION, reasoning=BAD_REASONING)
    run_completion(
        args.model,
        prompt_closed,
        "VARIANT E: raw completions API with chat template, </think> closed (expect answer=43, no reasoning if prefill works)",
    )


if __name__ == "__main__":
    main()

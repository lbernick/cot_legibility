"""
Checks whether apply_chat_template on R1/QwQ messages produces the same prompt
as the manually constructed completions API prompts in test_prefill.py.
"""

import argparse
from transformers import AutoTokenizer

QUESTION = "What is 15 + 27? Answer with only the number."
BAD_REASONING = """Let me add 15 + 27.
15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.
Yes, 15 + 27 = 43."""

MODELS = {
    "r1": "deepseek-ai/DeepSeek-R1",
    "qwq": "Qwen/QwQ-32B",
}

MANUAL_OPEN = {
    "r1":  "<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜><think>\n{r}",
    "qwq": "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n{r}",
}

MANUAL_CLOSED = {
    "r1":  "<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜><think>\n{r}\n</think>",
    "qwq": "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n{r}\n</think>",
}


def compare(label: str, templated: str, manual: str):
    print(f"\n{'='*60}")
    print(f"VARIANT: {label}")
    print(f"{'='*60}")
    print(f"TEMPLATED:\n{repr(templated)}")
    print(f"\nMANUAL:\n{repr(manual)}")
    print(f"\nMATCH: {templated == manual}")
    if templated != manual:
        for i, (a, b) in enumerate(zip(templated, manual)):
            if a != b:
                print(f"First diff at char {i}: templated={repr(a)} manual={repr(b)}")
                print(f"  context: ...{repr(templated[max(0,i-20):i+20])}...")
                break
        else:
            print(f"One is a prefix of the other (len diff: {len(templated)} vs {len(manual)})")


def try_continue(tok, messages, label, manual):
    try:
        result = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
        compare(f"{label}, continue_final_message=True", result, manual)
    except ValueError as e:
        print(f"\n{'='*60}")
        print(f"VARIANT: {label}, continue_final_message=True")
        print(f"{'='*60}")
        print(f"ERROR (template modified/deleted message content): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["r1", "qwq"])
    args = parser.parse_args()

    hf_model = MODELS[args.model]
    print(f"Loading tokenizer from {hf_model}...")
    tok = AutoTokenizer.from_pretrained(hf_model)

    manual_open = MANUAL_OPEN[args.model].format(q=QUESTION, r=BAD_REASONING)
    manual_closed = MANUAL_CLOSED[args.model].format(q=QUESTION, r=BAD_REASONING)

    # Variant A: closed </think>
    messages_a = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": f"<think>\n{BAD_REASONING}\n</think>"},
    ]
    templated_a = tok.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=False)
    compare("A (closed </think>), add_generation_prompt=False", templated_a, manual_closed)
    try_continue(tok, messages_a, "A (closed </think>)", manual_closed)

    # Variant C: open </think>
    messages_c = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": f"<think>\n{BAD_REASONING}"},
    ]
    templated_c = tok.apply_chat_template(messages_c, tokenize=False, add_generation_prompt=False)
    compare("C (open </think>), add_generation_prompt=False", templated_c, manual_open)
    try_continue(tok, messages_c, "C (open </think>)", manual_open)

    # Baseline
    messages_base = [{"role": "user", "content": QUESTION}]
    templated_base = tok.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)
    print(f"\n{'='*60}")
    print("BASELINE (user only, add_generation_prompt=True):")
    print(repr(templated_base))


if __name__ == "__main__":
    main()

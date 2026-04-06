# Prefill Investigation: OpenRouter + Reasoning Models

## Goal

The existing codebase (`src/inference/prefill_runner.py`) attempts to prefill reasoning models with partial chain-of-thought via the OpenRouter API, to test whether the legibility of a model's reasoning is causally related to its ability to reach the correct answer. Previous attempts to use this with DeepSeek R1 reportedly failed. We investigated why, and tested multiple approaches to prefilling through OpenRouter for both R1 (`deepseek/deepseek-r1`) and QwQ (`qwen/qwq-32b`).

## Test script

`test_prefill.py` asks "What is 15 + 27?" and prefills with reasoning that confidently argues the answer is **43** (the correct answer is 42). Five variants are tested.

Run with:
```bash
python test_prefill.py qwen/qwq-32b --template qwq
python test_prefill.py deepseek/deepseek-r1 --template r1 --reasoning-param
```

## How to interpret results

**Answering 42 does NOT prove the prefill was ignored** — the question is easy enough that a model could follow the prefill, notice the arithmetic error, and self-correct. Answering 43 is strong evidence the prefill was used.

The stronger signals are:
- **For closed `</think>` (Variants A, B, E):** If prefill worked, expect no new reasoning returned, and a clean answer (not a reasoning trace). A follow-up response ("How can I help?") instead of answering the question is a sign the model treated the prefill as a completed prior turn.
- **For open `</think>` (Variants C, D):** If prefill worked, the model continues reasoning from where we left off. The returned `reasoning` field may only contain *newly generated* reasoning (not the prefilled portion), so the REASONING CHECK (starts-with test) may fail even when the prefill worked. Better signals: does the model reference the injected reasoning? Does it answer 43? Does it engage with the specific error we planted (5+7=13)?

## OpenRouter response field semantics

- `reasoning` (string) and `reasoning_details` (array) always co-occur in responses. Their text content is always identical — they are redundant representations.
- `content` contains the post-`</think>` answer text.
- These are response-only fields. There is no `reasoning` field on request messages. `reasoning_details` can be set on request assistant messages but is designed for replaying prior model-generated reasoning in multi-turn, not injecting new reasoning.
- The completions API (`/completions`) returns raw text in `choices[0].text` with no separate reasoning field. Everything — including any new reasoning — comes back as one string. We split on `</think>` to separate reasoning continuation from answer.

---

## Results by variant

### Variant A: chat API, `content` field, `</think>` closed

This is the approach used by `prefill_runner.py`.

```json
{"role": "assistant", "content": "<think>\n...bad reasoning...\n</think>"}
```

| Model | reasoning returned | answer | interpretation |
|-------|-------------------|--------|----------------|
| QwQ   | False (0 chars)   | 42     | Ambiguous — no reasoning returned, consistent with prefill working (nothing to generate) but also consistent with ignoring it |
| R1    | True (492 chars): "The user sent only 'Assistant' which might be an incomplete message..." | "Hello! How can I assist you today?" | **Broken** — R1 treats the prefilled assistant message as a completed prior turn and expects a new user message |

**R1 Variant A is consistently broken across runs.** The model never answers the actual question.

---

### Variant B: chat API, `reasoning_details` field

```json
{"role": "assistant", "content": "", "reasoning_details": [{"type": "reasoning", "text": "...bad reasoning..."}]}
```

| Model | reasoning returned | answer | interpretation |
|-------|-------------------|--------|----------------|
| QwQ   | True (1415 chars), fresh re-reasoning | 42 | Prefill ignored — model re-reasoned from scratch |
| R1    | True (1031 chars): "the user just says 'assistant' which seems like a placeholder..." | 42 | Same "completed prior turn" problem as Variant A |

`reasoning_details` is intended for replaying model-generated reasoning across multi-turn conversations, not for injecting novel reasoning as a prefix.

---

### Variant C: chat API, `content` field, `</think>` open

```json
{"role": "assistant", "content": "<think>\n...bad reasoning..."}
```

| Model | reasoning returned | answer | interpretation |
|-------|-------------------|--------|----------------|
| QwQ   | True (1059 chars): " I'll just put the number 42. Wait, did I make a mistake? Let me check... 5+7 is 12, not 13. Wait, where did the 13 come from?" | 42 (with full step-by-step explanation) | **Prefill was seen** — model noticed the specific arithmetic error we planted (5+7=13) and explicitly questioned it. Self-corrected to 42. |
| R1    | True (1618 chars): "I already did that—I replied '43'. Now they send an empty message..." | "Is there anything else you'd like to calculate?" | **Prefill was seen** (R1 believed it already answered 43), but chat API still treats it as completed turn — model doesn't re-answer the question |

QwQ engages with the planted error but self-corrects. R1 accepts the injected reasoning as its own prior answer but still generates a follow-up instead of answering. **The chat API is fundamentally broken for R1 prefilling regardless of whether `</think>` is open or closed** — R1 treats any assistant message as a completed turn.

---

### Variant D: raw completions API, `</think>` open

Uses the `/completions` endpoint with manually applied chat template:
- QwQ: `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}`
- R1: `<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n{reasoning}`

| Model | raw output | answer | interpretation |
|-------|-----------|--------|----------------|
| QwQ   | "42" (2 chars, no `</think>`) | 42 | Ambiguous — very short output, unclear if model continued from prefill or ignored it |
| R1    | "43" (2 chars, no `</think>`) | **43** | **Prefill worked** — model continued from injected reasoning and produced the wrong answer |

R1 Variant D gave 43 in both runs (first and second). This is the most reliable positive result.

The short output (no `</think>`) suggests the completions API may be stripping or not returning the full generation including the reasoning continuation. We only see the answer.

---

### Variant E: raw completions API, `</think>` closed

Uses the same templates as Variant D but with `</think>` added:
- QwQ: `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}\n</think>`
- R1: `<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n{reasoning}\n</think>`

| Model | raw output | answer | interpretation |
|-------|-----------|--------|----------------|
| QwQ   | "42" (2 chars, no `</think>`) | 42 | Ambiguous |
| R1    | "42" (2 chars, no `</think>`) | 42 | Ambiguous — answered correctly despite bad reasoning. Could be self-correction or re-reasoning. |

Both models return a short answer with no reasoning. For R1, the contrast with Variant D (open=43, closed=42) may suggest the closed `</think>` causes the model to re-reason rather than follow the prefill — but could also be self-correction on an easy question.

---

## Summary table

| Variant | Mechanism | QwQ answer | QwQ prefill seen? | R1 answer | R1 prefill seen? |
|---------|-----------|-----------|-------------------|-----------|-----------------|
| A | chat, `content`, closed | 42 | Ambiguous | follow-up | No (broken) |
| B | chat, `reasoning_details` | 42 | No (re-reasoned) | follow-up | No (broken) |
| C | chat, `content`, open | 42 | **Yes** (noticed planted error) | follow-up | **Yes** (claimed "I replied 43") but broken |
| D | completions, open | 42 | Ambiguous | **43** | **Yes** |
| E | completions, closed | 42 | Ambiguous | 42 | Ambiguous |

---

## Why the chat API fails: `apply_chat_template` root cause

We verified the root cause by loading the HuggingFace tokenizers for both models and comparing `apply_chat_template` output against the manually constructed completions prompts (`check_chat_template.py`).

The chat completions API calls `apply_chat_template` with default settings — specifically without `continue_final_message=True`. This has two distinct failure modes depending on the model and variant:

**R1 Variant A (closed `</think>`) — reasoning is silently stripped:**

The R1 chat template completely discards `<think>...</think>` content from assistant messages:
```
'<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜><｜end▁of▁sentence｜>'
```
The prefilled reasoning never reaches the model. It sees a completed, empty assistant turn followed by a new generation prompt, and responds accordingly ("Hello! How can I assist you today?").

**R1 Variant C (open `</think>`) — turn is closed with EOS:**

The template keeps the content but appends `<｜end▁of▁sentence｜>`, closing the assistant turn:
```
'...<think>\n...\nYes, 15 + 27 = 43.<｜end▁of▁sentence｜>'
```
The model sees a completed prior turn containing reasoning that concluded "43", then starts a new generation — which is why it said "I already did that — I replied '43'."

**QwQ behaves identically**, with `<|im_end|>\n` as the turn closer instead of `<｜end▁of▁sentence｜>`.

**The fix is `continue_final_message=True`**, which produces the correct open-ended prefix in both cases (verified: exact match with the manual completions prompts). The chat completions API does not expose this flag. The raw completions API (Variant D) works because it bypasses `apply_chat_template` entirely, sending the exact prefix the model needs.

---

## Key conclusions

**R1 via chat API is fundamentally broken for prefilling.** The R1 chat template strips `<think>...</think>` from assistant messages (Variant A) or closes the turn with EOS (Variant C). Either way the model never continues from the injected reasoning. This is why previous prefill experiments with R1 failed.

**R1 via completions API (Variant D) works.** Answer was 43 in both runs. The raw completions endpoint bypasses `apply_chat_template` and achieves true prefix completion.

**QwQ via completions API is ambiguous.** Both D and E return a short "42" answer with no reasoning. We can't distinguish "model continued from prefill and self-corrected" from "model ignored prefill." The question is too easy to distinguish these cases.

**QwQ Variant C shows the prefill is seen** — the model explicitly questioned the planted arithmetic error ("5+7 is 12, not 13. Wait, where did the 13 come from?"). But it still self-corrected.

---

## Differences between `test_prefill.py` and `prefill_runner.py`

| Aspect | `prefill_runner.py` | `test_prefill.py` |
|--------|--------------------|--------------------|
| **Prefill format** | `<think>\n{reasoning}\n</think>` in `content` — always closed (Variant A only) | Tests all five variants |
| **API style** | Streaming (`stream=True`) | Non-streaming |
| **`reasoning` extra_body** | Always sends `reasoning: {enabled: True/False}` based on `include_reasoning` config | Only sent with `--reasoning-param` flag |
| **`temperature`** | Sends `temperature` from model config | Does not send |
| **`max_tokens`** | Sends `max_tokens` if configured | Does not send |
| **Provider routing** | Sends `openrouter_provider` preferences | Does not send |
| **Completions API** | Not used | Variant D and E use `/completions` endpoint |

**The critical issue:** `prefill_runner.py` uses Variant A exclusively (chat API, closed `</think>`). For R1, the chat template strips the `<think>...</think>` content entirely — the prefilled reasoning never reaches the model. Switching to the completions API (Variant D approach) is needed for R1.

---

## Next steps

1. **Use the completions API for R1 prefilling.** Variant D is the only approach that reliably works. `prefill_runner.py` would need to be updated to use `/completions` with a manually applied chat template for R1.

2. **Harder test question for QwQ and Variant E.** The arithmetic question allows self-correction. A question where the model cannot easily verify the answer (e.g., a multi-step reasoning problem where the correct answer is non-obvious) would distinguish "prefill used but self-corrected" from "prefill ignored." This is especially needed to evaluate QwQ and R1 Variant E.

3. **Completions API output visibility.** The completions endpoint returns very short outputs (just the final answer) with no reasoning visible. It's unclear whether the model is generating a full reasoning continuation internally and we're just seeing the tail, or whether the API is truncating. Examining `finish_reason` and token counts on the response object might help.

4. **QwQ prefilling may be fundamentally harder.** Across all variants, QwQ never produced the wrong answer. It either ignored the prefill entirely or engaged with it and self-corrected. This could mean QwQ is more robust to bad reasoning, or that none of the approaches achieved true prefilling for QwQ.

5. **Provider variation.** OpenRouter routes requests to different backend providers (SiliconFlow, DeepInfra, etc.) depending on availability. Different providers may apply the chat template differently, affecting how `<think>` tokens are handled. `prefill_runner.py` supports provider routing via `openrouter_provider` config; pinning to a specific provider may improve consistency.

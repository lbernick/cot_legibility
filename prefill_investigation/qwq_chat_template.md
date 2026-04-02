Loading tokenizer from Qwen/QwQ-32B...

============================================================
VARIANT: A (closed </think>), add_generation_prompt=False
============================================================
TEMPLATED:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.\n</think><|im_end|>\n'

MANUAL:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.\n</think>'

MATCH: False
One is a prefix of the other (len diff: 248 vs 237)

============================================================
VARIANT: A (closed </think>), continue_final_message=True
============================================================
TEMPLATED:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.\n</think>'

MANUAL:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.\n</think>'

MATCH: True

============================================================
VARIANT: C (open </think>), add_generation_prompt=False
============================================================
TEMPLATED:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.<|im_end|>\n'

MANUAL:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.'

MATCH: False
One is a prefix of the other (len diff: 239 vs 228)

============================================================
VARIANT: C (open </think>), continue_final_message=True
============================================================
TEMPLATED:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.'

MANUAL:
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.'

MATCH: True

============================================================
BASELINE (user only, add_generation_prompt=True):
'<|im_start|>user\nWhat is 15 + 27? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n'

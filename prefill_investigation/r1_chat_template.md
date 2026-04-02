Loading tokenizer from deepseek-ai/DeepSeek-R1...

============================================================
VARIANT: A (closed </think>), add_generation_prompt=False
============================================================
TEMPLATED:
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><｜end▁of▁sentence｜>'

MANUAL:
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.\n</think>'

MATCH: False
First diff at char 88: templated='｜' manual='t'
  context: ...'umber.<｜Assistant｜><｜end▁of▁sentence｜>'...

============================================================
VARIANT: A (closed </think>), continue_final_message=True
============================================================
ERROR (template modified/deleted message content): continue_final_message is set but the final message does not appear in the chat after applying the chat template! This can happen if the chat template deletes portions of the final message. Please verify the chat template and final message in your chat to ensure they are compatible.

============================================================
VARIANT: C (open </think>), add_generation_prompt=False
============================================================
TEMPLATED:
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.<｜end▁of▁sentence｜>'

MANUAL:
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.'

MATCH: False
One is a prefix of the other (len diff: 239 vs 220)

============================================================
VARIANT: C (open </think>), continue_final_message=True
============================================================
TEMPLATED:
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.'

MANUAL:
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><think>\nLet me add 15 + 27.\n15 + 27: first I add 5 + 7 = 13, carry the 1. Then 1 + 2 + 1 = 4. So the answer is 43.\nYes, 15 + 27 = 43.'

MATCH: True

============================================================
BASELINE (user only, add_generation_prompt=True):
'<｜begin▁of▁sentence｜><｜User｜>What is 15 + 27? Answer with only the number.<｜Assistant｜><think>\n'

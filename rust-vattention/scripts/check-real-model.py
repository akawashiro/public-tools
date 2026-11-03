#!/usr/bin/env python3
"""Compare real HF model greedy output and chat rendering with Transformers/OpenAI SDK."""
import argparse
from pathlib import Path
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from openai import OpenAI
root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
parser.add_argument('--port', type=int, default=8101)
parser.add_argument('--weight-dtype', choices=['fp16', 'fp32'], default='fp16')
args = parser.parse_args()
cache = root/'models/hub'/('models--'+args.model.replace('/', '--'))
snapshot = cache/'snapshots'/(cache/'refs/main').read_text().strip()
name = str(snapshot)
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(snapshot)
model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32, attn_implementation='eager').eval()
# Compare identical API sampling semantics. Some model repositories enable
# repetition_penalty in generation_config.json, even when do_sample=False.
model.generation_config = GenerationConfig(
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.generation_config.eos_token_id,
    pad_token_id=model.generation_config.pad_token_id,
    do_sample=False,
)

prompt = 'The capital of France is'
inputs = tokenizer(prompt,return_tensors='pt')
with torch.no_grad():
    generated = model.generate(**inputs,max_new_tokens=8,do_sample=False)
expected = tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
server = subprocess.Popen([str(root/'target/release/rust-vattention'),'--serve','--model',name,'--weight-dtype',args.weight_dtype,'--context-length','512','--port',str(args.port)])
client = OpenAI(base_url=f'http://127.0.0.1:{args.port}/v1',api_key='unused')
try:
    for _ in range(120):
        try:
            client.models.list()
            break
        except Exception: time.sleep(.5)
    else: raise AssertionError('server not ready')
    response = client.completions.create(model=name,prompt=prompt,max_tokens=8,temperature=0)
    actual = response.choices[0].text
    assert actual == expected, (actual,expected)
    # Exercise head_dim=128 Tensor Core attention across the full real model,
    # including the long-prompt BLAS path and continuing decode.
    for count in (12, 40):
        long_prompt = ('The quick brown fox jumps over the lazy dog. ' * count).strip()
        long_inputs = tokenizer(long_prompt, return_tensors='pt')
        length = long_inputs.input_ids.shape[1]
        assert 16 <= length < 508, length
        with torch.no_grad():
            long_ids = model.generate(**long_inputs, max_new_tokens=4, do_sample=False)
        expected_long = tokenizer.decode(long_ids[0, length:], skip_special_tokens=True)
        long_response = client.completions.create(model=name, prompt=long_prompt,
                                                  max_tokens=4, temperature=0)
        assert long_response.choices[0].text == expected_long, (length, long_response, expected_long)
        print(f'Real model {length}-token prefill/decode matches Transformers')
    messages=[{'role':'user','content':'Say hello in one word.'}]
    encoded=tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    response=client.chat.completions.create(model=name,messages=messages,max_tokens=4,temperature=0)
    assert response.usage.prompt_tokens == len(encoded), (response.usage,len(encoded))
    with torch.no_grad():
        ids=torch.tensor([encoded])
        chat_ids=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,do_sample=False)
    expected_chat=tokenizer.decode(chat_ids[0,len(encoded):],skip_special_tokens=True)
    assert response.choices[0].message.content == expected_chat, (response,expected_chat)
    stream=client.chat.completions.create(model=name,messages=messages,max_tokens=4,temperature=0,stream=True,stream_options={'include_usage':True})
    text=''
    usage=None
    for part in stream:
        if part.choices: text+=part.choices[0].delta.content or ''
        if part.usage: usage=part.usage
    assert text == response.choices[0].message.content,(text,response)
    assert usage.total_tokens == response.usage.total_tokens
    print(f'{args.model}: greedy output matches Transformers, chat template matches HF, OpenAI SDK JSON/SSE passed')
    print('Continuation:',repr(actual))
finally:
    server.terminate()
    server.wait(timeout=20)

import multiprocessing
from openai import OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
import csv
import argparse
import os


pleak_jsonl_path = "/home/fit/qiuhan/WORK/zyt/PLeak/0630-qwen-1-train.jsonl"

vllm_model_path = "Qwen/Qwen2.5-7B-Instruct"
model_name = "Qwen/Qwen2.5-7B-Instruct"
temperature = 1
normal_query = 0
use_defense = 0
port = 8000
inference = "vllm"
 
n_samples = 1 if temperature == 0 else 16
n_processes = 128 if temperature == 0 else 32

csv_save_path = "/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/pleak_t1_vllm.csv"
    
if inference == "api":
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("API key and base URL must be set in environment variables.")
    
elif inference == "vllm":
    api_key = "EMPTY"  
    base_url = f"http://127.0.0.1:{port}/v1" 
    model_name = vllm_model_path
else:
    raise ValueError("Inference method must be either 'vllm' or 'api'.")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(vllm_model_path)

samples = []
with open(pleak_jsonl_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        system_prompts = data['target_text'] # list
        trigger_tokens_list = data['trigger_tokens']
        attack_queries = []
        for trigger_tokens in trigger_tokens_list:
            trigger_prompt = tokenizer.decode(trigger_tokens["tokens"])
            attack_queries.append(trigger_prompt)
        for system_prompt in system_prompts:
            for attack_query in attack_queries:
                samples.append((system_prompt, attack_query))

print(samples[0])
print(len(samples))

def compute_rouge(reference, hypothesis):
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        score = scores['rougeL'].fmeasure
    except:
        score = 0
    return score


def process_prompt(args):
    system_prompt, attack_query = args
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": attack_query},
            ],
            model=model_name,
            temperature=temperature,
            n=n_samples
        )

        results = []
        for _, choice in enumerate(chat_completion.choices):
            ans = choice.message.content.strip()
            rougeL_score = compute_rouge(system_prompt, ans) if normal_query == 0 else -1

            results.append({
                "system_prompt": system_prompt,
                "attack_query": attack_query,
                "response": ans,
                "rougeL": rougeL_score,
            })

        return results

    except Exception as e:
        with open("error.log", "a") as f:
            f.write(f"System prompt: {system_prompt}\n")
            f.write(f"User prompt: {attack_query}\n")
            f.write("Exception traceback:\n")
            f.write("\n" + "="*80 + "\n")
        return []

def evaluate_prompts(samples, num_processes=8):
    results = []
    task_list = samples
    
    progress_bar = tqdm(total=len(task_list), desc="Processing Prompts")

    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_results in pool.imap_unordered(process_prompt, task_list):
            results.extend(batch_results)
            progress_bar.update(1)

    progress_bar.close()
    
    with open(csv_save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["system_prompt", "attack_query", "response", "rougeL"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_save_path}")

if __name__ == "__main__":
    evaluate_prompts(samples, num_processes=n_processes)
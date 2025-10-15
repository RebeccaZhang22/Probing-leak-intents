import multiprocessing
from openai import OpenAI
from tqdm import tqdm
import csv
csv.field_size_limit(1048576)
import re
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--vllm_model_path", type=str, help="Path to the running vllm model.")
parser.add_argument("--port", type=int, default=8000, help="API port for vllm.")
parser.add_argument("--raw_data_path", type=str, default="./data/raw")
parser.add_argument("--output_label_path", type=str, default="./data/label")

args = parser.parse_args()

model_path = args.vllm_model_path
port = args.port
raw_data_path = args.raw_data_path
output_label_path = args.output_label_path

api_key = "EMPTY"
api_base = f"http://11.11.1.2:{port}/v1" 

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

with open("./datasets/prompt-templates/label-template.md", "r", encoding="utf-8") as f:
    label_prompt = f.read().strip()


user_data_template = """
>>> SYSTEM PROMPT START
{system_prompt}
>>> SYSTEM PROMPT END

>>> ATTACK QUERY START
{attack_query}
>>> ATTACK QUERY END

>>> MODEL RESPONSE START
{response}
>>> MODEL RESPONSE END
""".strip()

def process_prompt(args):
    system_prompt, response, user_prompt, rougeL = args
    prompt = user_data_template.format(system_prompt=system_prompt, attack_query=user_prompt, response=response)

    results = []

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": label_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model_path,
            top_p=0.95,
            temperature=0
        )

        ans = chat_completion.choices[0].message.content

        reason_match = re.search(r'Reason:\s*(.*?)\n', ans)
        reason = reason_match.group(1).strip() if reason_match else ""

        label_match = re.search(r'Label:\s*(\d)', ans)
        label = label_match.group(1).strip() if label_match else -1

    except Exception as e:
        print(f"Exception in process_prompt: {e}")
        reason = f"Error: {str(e)}"
        label = -1

    results.append({
        "system_prompt": system_prompt,
        "attack_query": user_prompt,
        "response": response,
        "rougeL": rougeL,
        "reason": reason,
        "label": label,
    })

    return results

def evaluate_prompts(csv_path, csv_save_path, num_processes=8):
    results = []
    with open(csv_path, 'r') as file:
        reader = list(csv.DictReader(file))

    total_tasks = len(reader)
    progress_bar = tqdm(total=total_tasks, desc="Processing Prompts")

    task_list = []
    for idx, row in enumerate(reader):
        system_prompt = row['system_prompt']
        response = row["response"]
        user_prompt = row["attack_query"]
        rougeL = row["rougeL"]

        task_list.append((system_prompt, response, user_prompt, rougeL))

    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_results in pool.imap_unordered(process_prompt, task_list):
            results.extend(batch_results)
            progress_bar.update(1)


    progress_bar.close()

    with open(csv_save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["system_prompt", "attack_query", "response", "rougeL", "reason", "label"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved in {csv_save_path}")


if __name__ == "__main__":
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)
    csv_paths = os.listdir(raw_data_path)

    for csv_path in csv_paths:
        print(f"Processing {csv_path}...")
        if "pleak_t1_vllm.csv" in csv_path:
            continue
        target_path = os.path.join(output_label_path, csv_path)
        if os.path.exists(target_path): 
            continue
        src_path = os.path.join(raw_data_path, csv_path)
        evaluate_prompts(src_path, target_path, num_processes=160)

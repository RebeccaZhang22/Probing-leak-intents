## Code for ``I've Decided to Leak'': Probing Internals Behind Prompt Leakage Intents
This repository contains the codebase and instructions for reproducing the experiments from our paper.


We investigate how and when LLMs exhibit prompt leakage behaviors, and propose methods to detect prompt leakage intent through internal hidden states and probing models.


### 📁 Codebase Overview
This repository includes:

📌 Attack Scripts: Induce prompt leakage through diverse adversarial prompts

🔍 Labeling Scripts: Collect labels via LLM-assisted annotation under various criteria

🧠 Hidden State Extraction: Capture internal representations before generation

🧪 Dataset Construction: Build train/validation/test splits for probing tasks

⚖️ Risk Analysis Tools: Apply custom leakage definitions to compute risk metrics

🧑‍🏫 Probe Training: Train classifiers on LLM internals to detect leakage intent


### 🧪 Experiment Environment

We recommend using conda for reproducibility:

```bash
conda create -n leakenv python=3.10
conda activate leakenv
pip install -r requirements.txt
```

If using GPU, ensure CUDA is properly configured.


### 🚀 Quick Start
Below is a typical pipeline from data collection to probe training.


#### Step 1: Collect Model Responses

To generate responses under attack prompts:

```bash
bash data/run_attacks.sh
```

This will save raw model outputs to `./data/raw/`.

#### Step 2: Label using LLM

Use an LLM (via API or local model) to label responses for prompt leakage:

```bash
bash data/run_labels.sh
```

Labeled data will be saved in `./data/label/`.


#### Step 3: Extract Hidden States

Capture pre-generation hidden representations from the model:

``` bash
bash scripts/extract_hiddens.sh
```

This generates `.pt` files saved in `assets/hiddens/xxx/`.


#### Step 4: Organize Representations into a Dataset

Structure the hidden states with labels for training:

```bash
bash scripts/organize_dataset.sh
```

The dataset splits will be saved to `./assets/datasets/xxx/`.

#### Step 5: Relabel the dataset using updated labels or criteria

```bash
bash scripts/relabel_datasets.sh
```

Save the relabeled dataset to `./assets/datasets/xxx_with_hybrid_label/`.

#### Step 6: Train the probing model on the prepared dataset

Train a classifier (*e.g.*, logistic regression) to detect leakage intent:

```bash
bash scripts/train_probe.sh
```

### ⚙️ Model Inference: vLLM or API
The scripts (e.g., run_attacks.sh, run_labels.sh) require LLM access. You can choose one of two options:

#### ✅ Option 1: Run Local vLLM Server
Start the vLLM inference server (e.g., Qwen-7B or 32B):

```bash
bash vllm-qwen7b.sh      # for Qwen-7B
bash vllm-qwen32b.sh     # for Qwen-32B
```

Make sure the scripts are configured to call the vLLM server (usually at http://127.0.0.1:8000/v1).

#### ✅ Option 2: Use OpenAI-Compatible API
If using external APIs, set:

```bash
export OPENAI_API_KEY=="your_api_key_here"
export OPENAI_BASE_URL=="https://your.model.endpoint"
source ~/.bashrc
```

The scripts will automatically read these for model queries.



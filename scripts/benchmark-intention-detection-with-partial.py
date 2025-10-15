import torch
import numpy as np
import multiprocessing
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from models import load_probe
from torch.utils.data import Dataset


def build_dataset(feats, seed=42):
    torch.manual_seed(seed)
    features = feats["features"]  # List[num_layers] of [num_samples, hidden_size]
    labels = feats["labels"]
    systems = feats["systems"]

    # Get indices of positive and negative samples
    pos_indices = (labels > 0).nonzero().squeeze()
    neg_indices = (labels == 0).nonzero().squeeze()

    num_neg = len(neg_indices)
    num_pos = 512 - num_neg
    pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
    # neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:num_neg]]

    # Combine indices and shuffle
    indices = torch.cat([neg_indices, pos_sample_indices])
    indices = indices[torch.randperm(len(indices))]

    print(f"Among selected samples, {num_pos} are positive and {num_neg} are negative.")
    
    class ProbeDataset(Dataset):
        def __init__(
            self, features, labels, indices, systems
        ):
            self.labels = labels[indices]
            self.systems = [systems[i] for i in indices]
            
            # Get features from specified layers
            layer_index = "attn_21"
            middle_layer_index = "attn_22" 
            upper_layer_index = "attn_23"

            lowest_layer_features = features[layer_index][indices].to(torch.float)
            middle_layer_features = features[middle_layer_index][indices].to(torch.float)
            upper_layer_features = features[upper_layer_index][indices].to(torch.float)
            
            # Concatenate features from all layers
            self.features = torch.cat([
                lowest_layer_features,
                middle_layer_features, 
                upper_layer_features
            ], dim=-1)

            print(f"Features shape: {self.features.shape}")
            print(f"Labels shape: {self.labels.shape}")

        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "label": self.labels[idx]
            }
    
    dataset = ProbeDataset(
        features, labels, indices, systems
    )
    
    print(f"==> Loaded samples: {len(dataset)}\n==> {num_pos} positive, {num_neg} negative")

    return dataset

def eval_probe_single(probe, test_loader: DataLoader, device: torch.device, eval_metrics: list[str]):
    probe.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = probe(inputs).squeeze()
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    results = {}
    if "AUROC" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        results["AUROC"] = roc_auc_score(binary_labels, all_probs)
    if "BESTF1" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(binary_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        
        results["BESTF1"] = best_f1
        results["BESTF1_THRESHOLD"] = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        results['BESTF1_PRECISION'] = precision[best_idx]
        results['BESTF1_RECALL'] = recall[best_idx]
    if "BESTACC" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        thresholds = np.linspace(0, 1, 50)
        accuracies = []
        for threshold in thresholds:
            predictions = (all_probs > threshold).astype(int)
            accuracy = (predictions == binary_labels).mean()
            accuracies.append(accuracy)
        results["BESTACC"] = max(accuracies)

    return results


def reconstruct_full_dataset(path_list):
    features_accum = {}
    labels_all, systems_all, samples_all, responses_all = [], [], [], []

    for path in path_list:
        split = torch.load(path)
        for k, v in split["features"].items():
            feat_name = k.replace("full_", "")
            if feat_name not in features_accum:
                features_accum[feat_name] = [torch.tensor(item) if isinstance(item, list) else item for item in v]
            else:
                features_accum[feat_name].extend([torch.tensor(item) if isinstance(item, list) else item for item in v])
        labels_all.extend(split["labels"])
        systems_all.extend(split["systems"])
        samples_all.extend(split["samples"])
        responses_all.extend(split["responses"])

    features = {k: torch.stack(v) for k, v in features_accum.items()}
    labels = torch.tensor(labels_all)

    return {
        "features": features,
        "labels": labels,
        "systems": systems_all,
        "samples": samples_all,
        "responses": responses_all,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_path = "/home/fit/qiuhan/WORK/zyt/best_probe_AUROC.pt"

    datapath = "/home/fit/qiuhan/WORK/zyt/Probing-leak-intents-main/data/multi-turn-hybrid-relabeled/train.pt"
    normal_datapath = "/mnt/data/becool1/djs/why-du-leak/assets/hiddens/qwen-2-5-it-0418-normal/rougel_0.8_features.pt"

    full_dataset = reconstruct_full_dataset([datapath])
    print(len(full_dataset["labels"]))

    # Try K different random seeds
    K = 1000
    all_results = []
    best_aurocs = []
    
    for seed in range(K):
        print(f"\nTrying with random seed {seed}")
        dataset = build_dataset(full_dataset, seed=seed)

        # load probe
        probe_cls = load_probe("logistic_regression")
        probe = probe_cls(in_features=dataset[0]["features"].shape[-1])
        
        # load probe from checkpoint
        probe.load_state_dict(torch.load(probe_path))
        probe.to(device)

        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        results = eval_probe_single(probe, test_loader, device, ["BESTF1", "AUROC", "BESTACC"])
        all_results.append(results)
        best_aurocs.append(results["AUROC"])
        print(f"Results for seed {seed}:", results)

    # Save best AUROCs to file
    with open("best_aurocs.txt", "w") as f:
        for seed, auroc in enumerate(best_aurocs):
            f.write(f"Seed {seed}: {auroc}\n")

    # Calculate mean and std of metrics across seeds 
    metrics = ["BESTF1", "AUROC", "BESTACC"]
    for metric in metrics:
        values = [result[metric] for result in all_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"\n{metric} across {K} seeds:")
        print(f"Mean: {mean:.4f}")
        print(f"Std: {std:.4f}")

if __name__ == "__main__":
    main()
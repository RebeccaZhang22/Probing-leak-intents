import os
import time
import math
import random
import numpy as np
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

from models import load_probe



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_mean_std(features_tensor):
    # features_tensor: [N, D]
    mean = features_tensor.mean(dim=0, keepdim=True)
    std = features_tensor.std(dim=0, keepdim=True) + 1e-6  # add eps to avoid divide-by-zero
    return mean, std

def normalize_features(features_tensor, mean, std):
    return (features_tensor - mean) / std

def load_hidden_feats(feat_path: str):
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file {feat_path} does not exist.")
    feats = torch.load(feat_path)
    assert all(
       k in feats for k in ["features", "labels", "systems", "samples", "responses"]
    )
    return feats


def split_feats(feats, eval_ratio=0.1, seed=42, layer_index=10):
    torch.manual_seed(seed)
    
    features = feats["features"]  # List[num_layers] of [num_samples, hidden_size]
    labels = feats["labels"]

    print(f"Among all samples, {sum(labels > 0)} are positive and {len(labels) - sum(labels > 0)} are negative.")

    num_samples = len(labels)
    
    indices = torch.randperm(num_samples)
    split_idx = int(num_samples * (1 - eval_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    class ProbeDataset(Dataset):
        def __init__(self, features, labels, indices, layer_index=-1):
            if layer_index == -1:
                raise ValueError("You should set a layer index for training the probe.")
            # Cast to float 32
            self.features = features[layer_index][indices].to(torch.float)
            self.labels = labels[indices].to(torch.float)
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "label": self.labels[idx]
            }
    
    train_dataset = ProbeDataset(features, labels, train_indices, layer_index)
    val_dataset = ProbeDataset(features, labels, val_indices, layer_index)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    return train_dataset, val_dataset


def fit_lda(train_features, train_labels, n_components=1):
    lda = LDA(n_components=n_components)
    transformed = lda.fit_transform(train_features.cpu().numpy(), train_labels.cpu().numpy())
    return lda, torch.from_numpy(transformed).to(torch.float)

def apply_lda(features, lda_model):
    transformed = lda_model.transform(features.cpu().numpy())
    return torch.from_numpy(transformed).to(torch.float)

def fit_pca(train_features, pca_components):
    pca = PCA(n_components=pca_components)
    transformed = pca.fit_transform(train_features.cpu().numpy())
    return pca, torch.from_numpy(transformed).to(torch.float)

def apply_pca(features, pca):
    transformed = pca.transform(features.cpu().numpy())
    return torch.from_numpy(transformed).to(torch.float)

def fit_difference_pca(features: torch.Tensor, labels: torch.Tensor, n_components: int = 10):
    pos_mask = labels > 0
    neg_mask = labels == 0

    pos_feats = features[pos_mask]
    neg_feats = features[neg_mask]

    mu_pos = pos_feats.mean(dim=0, keepdim=True)
    mu_neg = neg_feats.mean(dim=0, keepdim=True)
    mu = (mu_pos + mu_neg) / 2

    num_pairs = min(len(pos_feats), len(neg_feats))
    X_list = []
    for i in range(num_pairs):
        X_list.append((pos_feats[i] - mu))
        X_list.append((neg_feats[i] - mu))
    X = torch.cat(X_list, dim=0)

    pca = PCA(n_components=n_components)
    X_np = X.cpu().numpy()
    X_proj = pca.fit_transform(X_np)

    return (pca, mu), torch.from_numpy(X_proj).to(torch.float)

def apply_difference_pca(features: torch.Tensor, projector: tuple):
    pca, mu = projector
    centered = features - mu
    transformed = pca.transform(centered.cpu().numpy())
    return torch.from_numpy(transformed).to(torch.float)

def average_model_soup(probes):
    soup = load_probe("logistic_regression")(
        in_features=probes[0].in_features,
        initialization="zero",
        dropout_rate=0.0,
    )
    with torch.no_grad():
        for name, param in soup.named_parameters():
            param.copy_(torch.stack([p.state_dict()[name] for p in probes]).mean(dim=0))
    return soup

def build_dataset(
    feats, num_samples=-1, layer_index=10, 
    pn_ratio=-1, feature_type="hidden",
    projection_components=-1, projection_type=None, projector=None,
    normalize_mean=None, normalize_std=None,
    normalize=False, bucketize=None, ref_feats=None
):
    
    features = feats["features"]  # List[num_layers] of [num_samples, hidden_size]
    labels = feats["labels"]

    if ref_feats is not None:
        # Add reference features as negative samples
        ref_features = ref_feats["features"]
        ref_labels = torch.zeros_like(ref_feats["labels"])  # All negative
        
        # Concatenate features and labels
        for layer in features:
            features[layer] = torch.cat([features[layer], ref_features[layer]], dim=0)
        labels = torch.cat([labels, ref_labels], dim=0)

    print(f"Among all samples, {sum(labels > 0)} are positive and {len(labels) - sum(labels > 0)} are negative.")

    if num_samples > 0:
        # Perform a measurement study and sample the same positive/negative ratio as the whole dataset
        num_positive = sum(labels > 0)
        num_negative = len(labels) - num_positive
        
        if pn_ratio < 0:
            # Calculate the desired number of positive and negative samples
            num_positive_sample = int(num_samples * (num_positive / len(labels)))
        else:
            num_positive_sample = int(num_samples * pn_ratio / (1 + pn_ratio))
            if num_positive_sample > num_positive:
                raise ValueError(f"The number of positive samples ({num_positive_sample}) is greater than the total number of positive samples ({num_positive}).")

        num_negative_sample = num_samples - num_positive_sample
        
        # Ensure the number of samples does not exceed the available positive or negative samples
        num_positive_sample = min(num_positive, num_positive_sample)
        num_negative_sample = min(num_negative, num_negative_sample)
        
        # Update total num_samples
        num_samples = num_positive_sample + num_negative_sample
        
        # Get positive and negative indices
        positive_indices = torch.where(labels > 0)[0]
        negative_indices = torch.where(labels == 0)[0]
        
        # Randomly sample the positive and negative indices
        positive_indices = positive_indices[torch.randperm(len(positive_indices))][:num_positive_sample]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))][:num_negative_sample]
        
        # Concatenate and shuffle the indices
        indices = torch.cat([positive_indices, negative_indices])
        indices = indices[torch.randperm(len(indices))]

    else:
        num_samples = len(labels)
        indices = torch.randperm(num_samples)
    
    class ProbeDataset(Dataset):
        def __init__(
            self, features, labels, indices, feature_type="hidden", layer_index:str=-1, 
            projection_components=-1, projection_type=None, projector=None, 
            normalize_mean=None, normalize_std=None, normalize=False, bucketize=None
        ):
            if layer_index == -1:
                raise ValueError("You should set a layer index for training the probe.")
            
            if layer_index.startswith("full_"):
                layer_index = layer_index.split("full_")[1]

            def unify_layer_index(layer_index):
                if layer_index.startswith("ffn_"):
                    return f"full_{layer_index}"
                elif layer_index.startswith("attn_"):
                    return f"full_{layer_index}"
                else:
                    return layer_index

            if feature_type == "hidden":
                layer_index = unify_layer_index(layer_index) if not layer_index.startswith("system") else layer_index
                self.features = features[layer_index][indices].to(torch.float)
            elif feature_type == "hidden_shift":
                after_layer_index = unify_layer_index(layer_index) if not layer_index.startswith("system") else layer_index
                after_shift_features = features[after_layer_index][indices].to(torch.float)
                before_shift_feature_name = f"system_{layer_index}"
                before_shift_features = features[before_shift_feature_name][indices].to(torch.float)
                # do minus
                self.features = after_shift_features - before_shift_features
            elif feature_type == "diff_sublayer":
                # use the difference between the two layers, the layer_index as the lower layer
                lower_layer_index = unify_layer_index(layer_index)
                lower_layer_features = features[lower_layer_index][indices].to(torch.float)
                lower_layer_int_index = int(layer_index.split("_")[-1])
                if layer_index.startswith("attn_"):
                    upper_layer_name = f"ffn_{lower_layer_int_index}"
                elif layer_index.startswith("ffn_"):
                    upper_layer_name = f"attn_{lower_layer_int_index + 1}"
                else:
                    raise ValueError(f"Unknown layer type: {layer_index}")
                upper_layer_name = unify_layer_index(upper_layer_name)
                upper_layer_features = features[upper_layer_name][indices].to(torch.float)
                # do minus
                self.features = upper_layer_features - lower_layer_features
            elif feature_type == "diff_layer":
                # use the difference between the two layers, the layer_index as the lower layer
                lower_layer_index = unify_layer_index(layer_index)
                lower_layer_features = features[lower_layer_index][indices].to(torch.float)
                lower_layer_int_index = int(layer_index.split("_")[-1])
                lower_layer_type = layer_index.split("_")[0]
                assert lower_layer_type in ["attn", "ffn"]
                upper_layer_name = f"{lower_layer_type}_{lower_layer_int_index + 1}"
                upper_layer_name = unify_layer_index(upper_layer_name)
                upper_layer_features = features[upper_layer_name][indices].to(torch.float)
                # do minus
                self.features = upper_layer_features - lower_layer_features
            elif feature_type == "consecutive_sublayer": # can be form of ("ffn", "attn", "ffn") or ("attn", "ffn", "attn")
                assert layer_index.startswith("ffn_") or layer_index.startswith("attn_")
                layer_index = unify_layer_index(layer_index)
                lower_layer_features = features[layer_index][indices].to(torch.float)
                lower_layer_int_index = int(layer_index.split("_")[-1])
                lower_layer_type = layer_index.split("_")[0]
                
                if lower_layer_type == "ffn":
                    middle_layer_name = f"attn_{lower_layer_int_index + 1}"
                    upper_layer_name = f"ffn_{lower_layer_int_index + 1}"
                else:  # attn
                    middle_layer_name = f"ffn_{lower_layer_int_index}"
                    upper_layer_name = f"attn_{lower_layer_int_index + 1}"
                    
                middle_layer_name = unify_layer_index(middle_layer_name)
                middle_layer_features = features[middle_layer_name][indices].to(torch.float)
                
                upper_layer_name = unify_layer_index(upper_layer_name)
                upper_layer_features = features[upper_layer_name][indices].to(torch.float)
                
                # do concatenate
                self.features = torch.cat([lower_layer_features, middle_layer_features, upper_layer_features], dim=-1)
            elif feature_type == "consecutive_layer": # always as a form of 3 consecutive the same type layer
                lowest_layer_index = int(layer_index.split("_")[-1])
                layer_type = layer_index.split("_")[0]
                assert layer_type in ["attn", "ffn"]
                layer_index = unify_layer_index(layer_index)
                lowest_layer_features = features[layer_index][indices].to(torch.float)
                middle_layer_name = f"{layer_type}_{lowest_layer_index + 1}"
                middle_layer_name = unify_layer_index(middle_layer_name)
                middle_layer_features = features[middle_layer_name][indices].to(torch.float)
                upper_layer_name = f"{layer_type}_{lowest_layer_index + 2}"
                upper_layer_name = unify_layer_index(upper_layer_name)
                upper_layer_features = features[upper_layer_name][indices].to(torch.float)
                # do concatenate
                self.features = torch.cat([lowest_layer_features, middle_layer_features, upper_layer_features], dim=-1)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            print("[Before Normalization] Mean:", self.features.mean().item(), "Std:", self.features.std().item())

            if normalize:
                if normalize_mean is None or normalize_std is None:
                    # Compute mean and std across all dimensions
                    normalize_mean = self.features.mean(dim=0, keepdim=True)
                    normalize_std = self.features.std(dim=0, keepdim=True)
                    self.normalize_mean = normalize_mean
                    self.normalize_std = normalize_std
                else:
                    self.normalize_mean = normalize_mean
                    self.normalize_std = normalize_std
                
                # Apply normalization
                self.features = (self.features - self.normalize_mean) / (self.normalize_std + 1e-8)
                print("[After Normalization] Mean:", self.features.mean().item(), "Std:", self.features.std().item())

            self.labels = labels[indices].to(torch.float)
            self.max_leakage = labels.max().item()

            if bucketize is not None and self.max_leakage > 0 and bucketize > 0:
                pos_labels = self.labels[self.labels > 0].unique()
                pos_labels = torch.sort(pos_labels)[0]

                if len(pos_labels) <= bucketize:
                    thresholds = pos_labels
                else:
                    thresholds = pos_labels[torch.linspace(0, len(pos_labels)-1, steps=bucketize).long()]
                
                bucket_map = {}
                for i, t in enumerate(thresholds):
                    for v in pos_labels:
                        if v <= t:
                            bucket_map[v.item()] = i + 1
                    pos_labels = pos_labels[pos_labels > t]

                new_labels = torch.zeros_like(self.labels, dtype=torch.long)
                for label_val, bucket_id in bucket_map.items():
                    new_labels[self.labels == label_val] = bucket_id

                self.labels = new_labels.to(torch.float)
                self.max_leakage = bucketize
                print(f"Bucketized positive labels into {bucketize} cumulative buckets. Max leakage set to {self.max_leakage}.")

            if projection_components > 0:
                if projection_type == "pca":
                    if projector is not None:
                        self.features = apply_pca(self.features, projector)
                    else:
                        pca, transformed = fit_pca(self.features, projection_components)
                        self.features = transformed
                        self.projector = pca  # Save for reuse
                elif projection_type == "lda":
                    if projector is not None:
                        self.features = apply_lda(self.features, projector)
                    else:
                        lda, transformed = fit_lda(self.features, (self.labels > 0).float(), projection_components)
                        self.features = transformed
                        self.projector = lda  # Save for reuse
                elif projection_type == "difference_pca":
                    if projector is not None:
                        self.features = apply_difference_pca(self.features, projector)
                    else:
                        pca, _ = fit_difference_pca(self.features, (self.labels > 0).float(), projection_components)
                        self.features = apply_difference_pca(self.features, pca)
                        self.projector = pca  # Save for reuse
                else:
                    raise ValueError(f"Unknown projection type: {projection_type}")
            else:
                self.projector = None

        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "label": self.labels[idx]
            }
    
    dataset = ProbeDataset(
        features, labels, indices, feature_type=feature_type, 
        layer_index=layer_index, projection_components=projection_components, 
        projection_type=projection_type, projector=projector, 
        normalize_mean=normalize_mean, normalize_std=normalize_std, normalize=normalize, bucketize=bucketize
    )
    
    print(f"==> Loaded samples: {len(dataset)}\n==> {sum(dataset.labels > 0)} positive, {len(dataset) - sum(dataset.labels > 0)} negative")

    return dataset


class EvalResultTracker:
    def __init__(self, eval_name, metrics):
        self.best_scores = {metric: 0.0 for metric in metrics}
        self.best_epochs = {metric: 0 for metric in metrics}
        self.eval_name = eval_name

    def update(self, scores, epoch):
        for metric, score in scores.items():
            if metric in self.best_scores and score > self.best_scores[metric]:
                self.best_scores[metric] = score
                self.best_epochs[metric] = epoch

    def get_best_scores(self):
        return self.best_scores, self.best_epochs

def train_probe(
    probe,
    train_loader: DataLoader,
    test_loaders: list[DataLoader],
    num_epochs: int,
    learning_rate: float,
    l2_penalty: float,
    optimizer_type: str,
    device: torch.device,
    eval_metrics: list[str],
    output_dir: str,
    train_method: str,
    alpha:float,
    base:float,
    margin:float
):
    probe = probe.to(device)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(probe.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            probe.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=l2_penalty
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # # Add warmup scheduler
    # warmup_epochs = int(0.1 * num_epochs)  # 10% of total epochs for warmup
    # def lr_lambda(epoch):
    #     if epoch < warmup_epochs:
    #         return epoch / warmup_epochs  # Linear warmup
    #     return max(0.0, (num_epochs - epoch)/(num_epochs - warmup_epochs))  # Linear decay
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create trackers for each eval set
    trackers = {name: EvalResultTracker(name, eval_metrics) for name, _ in test_loaders}

    start_time = time.time()
    val_best_epoch = 0
    val_best_metric = eval_metrics[0]
    heldout_results_at_val_best = {}
    best_probe_state_dict = None

    def format_single_log(epoch, train_losses, results):
        message = f"\n{'='*80}\n"
        message += f"Epoch {epoch}/{num_epochs} (Time elapsed: {time.time() - start_time:.2f}s)\n"
        message += f"{'='*80}\n"
        
        # Print training losses
        message += "Training Losses:\n"
        message += "-"*40 + "\n"
        for loss_name, loss_val in train_losses.items():
            message += f"{loss_name:>20}: {loss_val:.4f}\n"
            
        # Print evaluation results  
        message += "\nEvaluation Results:\n"
        message += "-"*40 + "\n"
        for name, metrics in results.items():
            message += f"{name.upper()}:\n"
            for metric, score in metrics.items():
                message += f"{metric:>20}: {score:.4f} "
            message += "\n"
        return message

    # Record initial results before training
    initial_results = eval_probe(probe, test_loaders, device, eval_metrics)
    start_message = format_single_log(0, {"Total Loss": float('inf')}, initial_results)
    print(start_message)

    try:
        max_leakage = train_loader.dataset.max_leakage
    except:
        print("Cannot find the max_leakage from train_loader, setting it to 16 as default")
        max_leakage = 16
    print(f"max_leakage: {max_leakage}")

    for epoch in range(num_epochs):
        probe.train()
        epoch_losses = {
            "Total Loss": 0.0,
            "BCE Loss": 0.0,
            "Ranking Loss": 0.0
        }
        num_batches = 0

        for batch in train_loader:
            inputs = batch["features"].to(device)
            # leakage count
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = probe(inputs).squeeze()

            if train_method == "binary_only":
                processed_labels = (labels > 0).float()
                loss = bce_with_logits(logits, processed_labels)
                epoch_losses["BCE Loss"] += loss.item()
                epoch_losses["Total Loss"] += loss.item()
            elif train_method == "binary_plus_mse_risk_linear":
                bin_labels = (labels > 0).float()
                risk_labels = torch.zeros_like(labels, dtype=torch.float)
                risk_labels[labels > 0] = base + (1.0 - base) * labels[labels > 0] / max_leakage

                bce_loss = bce_with_logits(logits, bin_labels)
                epoch_losses["BCE Loss"] += bce_loss.item()

                pos_mask = bin_labels > 0
                if pos_mask.sum() > 1:
                    pred_probs = torch.sigmoid(logits[pos_mask])
                    risk_scores = risk_labels[pos_mask]
                    ranking_loss = torch.nn.functional.mse_loss(pred_probs, risk_scores)
                    loss = bce_loss + alpha * ranking_loss
                    epoch_losses["Ranking Loss"] += (alpha * ranking_loss).item()
                else:
                    loss = bce_loss
                epoch_losses["Total Loss"] += loss.item()
            elif train_method == "binary_plus_mse_risk_log_scaled":
                bin_labels = (labels > 0).float()
                risk_labels = torch.zeros_like(labels, dtype=torch.float)
                risk_labels[labels > 0] = base + (1.0 - base) * labels[labels > 0] / max_leakage

                bce_loss = bce_with_logits(logits, bin_labels)
                epoch_losses["BCE Loss"] += bce_loss.item()

                scaled = torch.log1p(labels.to(torch.float)) / torch.log1p(torch.tensor(max_leakage, dtype=torch.float))
                
                pos_mask = labels > 0
                processed_labels = torch.zeros_like(logits)
                processed_labels[pos_mask] = base + (1.0 - base) * scaled[pos_mask]

                pred_probs = torch.sigmoid(logits)
                ranking_loss = torch.nn.functional.mse_loss(pred_probs, processed_labels)
                loss = bce_loss + alpha * ranking_loss
                
                epoch_losses["Ranking Loss"] += (alpha * ranking_loss).item()
                epoch_losses["Total Loss"] += loss.item()

            elif train_method == "binary_plus_pairwise_order":
                bin_labels = (labels > 0).float()
                bce_loss = bce_with_logits(logits, bin_labels)
                epoch_losses["BCE Loss"] += bce_loss.item()

                pos_mask = bin_labels > 0
                if pos_mask.sum() > 1:
                    pos_logits = logits[pos_mask]
                    pos_labels = labels[pos_mask]
                    label_diff = pos_labels.unsqueeze(1) - pos_labels.unsqueeze(0)
                    logit_diff = pos_logits.unsqueeze(1) - pos_logits.unsqueeze(0)

                    positive_pairs = label_diff > 0

                    valid_diffs = logit_diff[positive_pairs]
                    if len(valid_diffs) > 0:
                        ranking_loss = torch.relu(margin - valid_diffs).mean()
                        loss = bce_loss + alpha * ranking_loss
                        epoch_losses["Ranking Loss"] += (alpha * ranking_loss).item()
                    else:
                        loss = bce_loss
                else:
                    loss = bce_loss
                epoch_losses["Total Loss"] += loss.item()
            elif train_method == "binary_plus_pairwise_order_with_base":
                bin_labels = (labels > 0).float()
                bce_loss = bce_with_logits(logits, bin_labels)
                epoch_losses["BCE Loss"] += bce_loss.item()

                pos_mask = bin_labels > 0
                if pos_mask.sum() > 1:
                    pos_logits = logits[pos_mask]
                    pos_labels = labels[pos_mask]
                    label_diff = pos_labels.unsqueeze(1) - pos_labels.unsqueeze(0)
                    logit_diff = pos_logits.unsqueeze(1) - pos_logits.unsqueeze(0)


                    positive_pairs = label_diff > 0

                    valid_diffs = logit_diff[positive_pairs]
                    
                    min_pos_label_mask = pos_labels == pos_labels.min()
                    min_pos_logits = pos_logits[min_pos_label_mask]
                    min_pos_probs = torch.sigmoid(min_pos_logits)
                    base_loss = torch.relu(base - min_pos_probs).mean()
                    
                    if len(valid_diffs) > 0:
                        ranking_loss = torch.relu(margin - valid_diffs).mean()
                        loss = bce_loss + alpha * (ranking_loss + base_loss)
                        epoch_losses["Ranking Loss"] += (alpha * ranking_loss).item()
                    else:
                        loss = bce_loss + alpha * base_loss
                else:
                    loss = bce_loss
                epoch_losses["Total Loss"] += loss.item()
            elif train_method == "binary_plus_pairwise_order_contrastive":
                bin_labels = (labels > 0).float()
                bce_loss = bce_with_logits(logits, bin_labels)
                epoch_losses["BCE Loss"] += bce_loss.item()

                pos_mask = bin_labels > 0
                neg_mask = ~pos_mask
                if pos_mask.sum() > 1:
                    pos_logits = logits[pos_mask]
                    pos_labels = labels[pos_mask]
                    neg_logits = logits[neg_mask]
                    
                    label_diff = pos_labels.unsqueeze(1) - pos_labels.unsqueeze(0)
                    logit_diff = pos_logits.unsqueeze(1) - pos_logits.unsqueeze(0)

                    positive_pairs = label_diff > 0

                    valid_diffs = logit_diff[positive_pairs]
                    
                    min_pos_logits = pos_logits.min()
                    max_neg_logits = neg_logits.max() if len(neg_logits) > 0 else min_pos_logits
                    base = torch.sigmoid(max_neg_logits - min_pos_logits)
                    base_loss = torch.relu(base - torch.sigmoid(min_pos_logits - max_neg_logits)).mean()
                    
                    if len(valid_diffs) > 0:
                        ranking_loss = torch.relu(margin - valid_diffs).mean()
                        loss = bce_loss + alpha * (ranking_loss + base_loss)
                        epoch_losses["Ranking Loss"] += (alpha * ranking_loss).item()
                    else:
                        loss = bce_loss + alpha * base_loss
                else:
                    loss = bce_loss
                epoch_losses["Total Loss"] += loss.item()
            elif train_method == "soft_label_linear_risk":
                processed_labels = torch.zeros_like(labels, dtype=torch.float)
                processed_labels[labels > 0] = base + (1.0 - base) * labels[labels > 0] / max_leakage
                loss = bce_with_logits(logits, processed_labels)
                epoch_losses["BCE Loss"] += loss.item()
                epoch_losses["Total Loss"] += loss.item()
            elif train_method == "soft_label_log_scaled_risk":
                processed_labels = torch.zeros_like(labels, dtype=torch.float)
                scaled = torch.log1p(labels.to(torch.float)) / torch.log1p(torch.tensor(max_leakage, dtype=torch.float))
                processed_labels[labels > 0] = base + (1.0 - base) * scaled[labels > 0]
                loss = bce_with_logits(logits, processed_labels)
                epoch_losses["BCE Loss"] += loss.item()
                epoch_losses["Total Loss"] += loss.item()
            else:
                raise ValueError(f"Unknown train_method: {train_method}")

            loss.backward()
            optimizer.step()
            num_batches += 1

        # Calculate average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        test_results = eval_probe(probe, test_loaders, device, eval_metrics)
        message = format_single_log(epoch+1, epoch_losses, test_results)
        print(message)

        # Update trackers for each eval set
        # Check if this epoch has best val score on the main metric
        if "val" in test_results:
            val_score = test_results["val"].get(val_best_metric, 0.0)
            if val_score > trackers["val"].best_scores[val_best_metric]:
                val_best_epoch = epoch + 1
                heldout_results_at_val_best = {
                    name: metrics for name, metrics in test_results.items()
                }
                best_probe_state_dict = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
        
        for name, metrics in test_results.items():
            trackers[name].update(metrics, epoch+1)
            if name == "val" and output_dir is not None:
                # Save model if any metric improves
                for metric, score in metrics.items():
                    try:
                        if score > trackers[name].best_scores[metric]:
                            torch.save(probe.state_dict(), os.path.join(output_dir, f"best_probe_{metric}.pt"))
                    except:
                        print(f"Error saving best probe for {name} with metric {metric}")
                        pass

    # Print best results and initial results for comparison
    print("\nTraining Summary:")
    print("="*80)
    for name in initial_results:
        print(f"\n{name.upper()}:")
        for metric in eval_metrics:
            initial_score = initial_results[name][metric]
            best_scores, best_epochs = trackers[name].get_best_scores()
            print(f"{metric:>20}: {initial_score:.4f} -> {best_scores[metric]:.4f} (best at epoch {best_epochs[metric]})")

    print("\nPerformance on Held-out Subsets @ Best Val Epoch ({}):".format(val_best_epoch))
    print("="*80)
    for name, metrics in heldout_results_at_val_best.items():
        print(f"\n{name.upper()}:")
        for metric, score in metrics.items():
            print(f"{metric:>20}: {score:.4f}")

    if best_probe_state_dict is not None:
        print(f"\n==> Restoring probe to best checkpoint at val epoch {val_best_epoch}")
        probe.load_state_dict(best_probe_state_dict)
    else:
        print("\n[Warning] No best checkpoint found. Returning probe at final epoch.")

    return probe




def eval_probe(probe, test_loaders: list[DataLoader], device: torch.device, eval_metrics: list[str]):
    results = {}
    for name, test_loader in test_loaders:
        results[name] = eval_probe_single(probe, test_loader, device, eval_metrics)
    return results

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(all_labels, all_probs, file_path="roc_curve.png", title="ROC Curve"):
    """
    Plot and save the ROC curve.

    Args:
        all_labels (np.ndarray): True binary labels.
        all_probs (np.ndarray): Predicted probabilities.
        file_path (str): Path to save the ROC curve.
        title (str): Title of the plot.
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save the plot to the specified file path
    plt.savefig(file_path, format=file_path.split('.')[-1])
    plt.close()
    print(f"ROC curve saved at: {file_path}")


def collect_probe_predictions(probe, test_loader, device):
    """
    Evaluate a single probe on a test loader.

    Args:
        probe: The probe model.
        test_loader: DataLoader for testing.
        device: The computation device.
        0: Threshold to classify leakage.

    Returns:
        dict: Dictionary of evaluation results.
    """
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
    all_labels = (torch.cat(all_labels) > 0).numpy()
    return {"all_labels": all_labels, "all_probs": all_probs}


def draw_the_probe_rocs(probe, test_loaders, device, output_dir="roc_plots"):
    """
    Draw ROC curves for multiple probes and save them as files.

    Args:
        probe: The probe model.
        test_loaders (list[tuple]): List of tuples (name, DataLoader).
        device: The computation device.
        output_dir (str): The directory to save the ROC plots.
        0 (float): Threshold for binary classification.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, test_loader in test_loaders:
        # Evaluate the probe on the current test loader
        result = collect_probe_predictions(probe, test_loader, device)

        # Check if the evaluation returned valid data
        all_labels = result.get("all_labels")
        all_probs = result.get("all_probs")
        if all_labels is None or all_probs is None:
            print(f"Warning: Missing labels or probabilities for probe '{name}'. Skipping.")
            continue

        # Generate file path based on probe name
        file_path = os.path.join(output_dir, f"roc_curve_{name}.png")
        title = f"ROC Curve - {name}"

        # Draw and save the ROC curve
        plot_roc_curve(all_labels, all_probs, file_path=file_path, title=title)

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
        # convert all_labels to 0/1
        binary_labels = (all_labels > 0).astype(int)
        results["AUROC"] = roc_auc_score(binary_labels, all_probs)
    if "AUPRC" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        results["AUPRC"] = average_precision_score(binary_labels, all_probs)
    if "Spearman" in eval_metrics:
        positive_mask = all_labels > 0
        if positive_mask.sum() > 0:
            corr, p_value = spearmanr(all_labels[positive_mask], all_probs[positive_mask])
            results["Spearman"] = corr
            results["Spearman_p"] = p_value 
        else:
            results["Spearman"] = 0.0
            results["Spearman_p"] = 1.0
    if "PCC" in eval_metrics:
        positive_mask = all_labels > 0
        if positive_mask.sum() > 0:
            corr, p_value = pearsonr(all_labels[positive_mask], all_probs[positive_mask])
            results["PCC"] = corr
            results["PCC_p"] = p_value
        else:
            results["PCC"] = 0.0
            results["PCC_p"] = 1.0
    if "BESTACC" in eval_metrics:
        # convert all_labels to 0/1
        binary_labels = (all_labels > 0).astype(int)
        # Try different thresholds to find best accuracy
        thresholds = np.linspace(0, 1, 50)
        accuracies = []
        for threshold in thresholds:
            predictions = (all_probs > threshold).astype(int)
            accuracy = (predictions == binary_labels).mean()
            accuracies.append(accuracy)
        results["BESTACC"] = max(accuracies)
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
    if "F1@0.5" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        fixed_preds = (all_probs > 0.5).astype(int)
        results["F1@0.5"] = f1_score(binary_labels, fixed_preds)
    if "ACC@0.5" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        fixed_preds = (all_probs > 0.5).astype(int)
        results["ACC@0.5"] = accuracy_score(binary_labels, fixed_preds)
    if 'TPR @ 5% FPR' in eval_metrics:
        # Binarize labels using the leakage decision threshold
        binary_labels = (all_labels > 0).astype(int)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, all_probs)
        
        # Mask for FPR <= 5%
        mask = fpr <= 0.05
        if np.any(mask):
            results["TPR @ 5% FPR"] = np.max(tpr[mask])
        else:
            results["TPR @ 5% FPR"] = 0.0
    if 'FPR @ 95% TPR' in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs)
        
        mask = tpr >= 0.95
        results["FPR @ 95% TPR"] = float(np.min(fpr[mask])) if np.any(mask) else 1.0

    return results

def eval_probe_ensemble(probes, test_loader: DataLoader, device: torch.device, eval_metrics: list[str]):
    for probe in probes:
        probe.eval()

    all_labels = []
    all_probs_ensemble = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["features"].to(device)
            labels = batch["label"].to(device)
            all_labels.append(labels.cpu())

            # collect predictions from all probes
            batch_probs = []
            for probe in probes:
                logits = probe(inputs).squeeze()
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu())
            
            # average predictions over all probes
            batch_probs_avg = torch.stack(batch_probs).mean(dim=0)
            all_probs_ensemble.append(batch_probs_avg)

    all_probs = torch.cat(all_probs_ensemble).numpy()
    all_labels = torch.cat(all_labels).numpy()

    results = {}
    if "AUROC" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        results["AUROC"] = roc_auc_score(binary_labels, all_probs)
    if "AUPRC" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        results["AUPRC"] = average_precision_score(binary_labels, all_probs)
    if "Spearman" in eval_metrics:
        positive_mask = all_labels > 0
        if positive_mask.sum() > 0:
            corr, p_value = spearmanr(all_labels[positive_mask], all_probs[positive_mask])
            results["Spearman"] = corr
            results["Spearman_p"] = p_value
        else:
            results["Spearman"] = 0.0
            results["Spearman_p"] = 1.0
    if "PCC" in eval_metrics:
        positive_mask = all_labels > 0
        if positive_mask.sum() > 0:
            corr, p_value = pearsonr(all_labels[positive_mask], all_probs[positive_mask])
            results["PCC"] = corr
            results["PCC_p"] = p_value
        else:
            results["PCC"] = 0.0
            results["PCC_p"] = 1.0
    if "BESTACC" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        thresholds = np.linspace(0, 1, 50)
        accuracies = []
        for threshold in thresholds:
            predictions = (all_probs > threshold).astype(int)
            accuracy = (predictions == binary_labels).mean()
            accuracies.append(accuracy)
        results["BESTACC"] = max(accuracies)
    if "BESTF1" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        precision, recall, thresholds = precision_recall_curve(binary_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        results["BESTF1"] = best_f1
        results["BESTF1_THRESHOLD"] = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    if "F1@0.5" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        fixed_preds = (all_probs > 0.5).astype(int)
        results["F1@0.5"] = f1_score(binary_labels, fixed_preds)
    if "ACC@0.5" in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        fixed_preds = (all_probs > 0.5).astype(int)
        results["ACC@0.5"] = accuracy_score(binary_labels, fixed_preds)
    if 'TPR @ 5% FPR' in eval_metrics:
        # Binarize labels using the leakage decision threshold
        binary_labels = (all_labels > 0).astype(int)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, all_probs)
        
        # Mask for FPR <= 5%
        mask = fpr <= 0.05
        if np.any(mask):
            results["TPR @ 5% FPR"] = np.max(tpr[mask])
        else:
            results["TPR @ 5% FPR"] = 0.0
    if 'FPR @ 95% TPR' in eval_metrics:
        binary_labels = (all_labels > 0).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs)
        
        mask = tpr >= 0.95
        results["FPR @ 95% TPR"] = float(np.min(fpr[mask])) if np.any(mask) else 1.0
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Train a probe on hidden features")
    parser.add_argument(
        "--probe_type",
        type=str,
        default="logistic_regression",
        help="Type of probe to train",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data",
    )
    parser.add_argument(
        "--reference_normal_data_path",
        type=str,
        default=None,
        help="Path to normal reference data for computing mean/std normalization",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="Path to validation set",
    )
    parser.add_argument(
        "--heldout_attacks_data_path",
        type=str,
        default=None,
        help="Path to dataset of heldout attacks",
    )
    parser.add_argument(
        "--heldout_strict_data_path",
        type=str,
        default=None,
        help="Path to dataset of heldout strict",
    )
    parser.add_argument(
        "--heldout_systems_data_path",
        type=str,
        default=None,
        help="Path to dataset of heldout systems",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the trained probe",
    )
    parser.add_argument(
        "--initialization_type",
        type=str,
        default="random",
        choices=["random", "zero"],
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="hidden",
        choices=["hidden", "hidden_shift", "diff_sublayer", "diff_layer", "consecutive_sublayer", "consecutive_layer"],
        help="Type of features to use for training the probe",
    )
    parser.add_argument(
        "--eval_feature_type",
        type=str,
        default=None,
        choices=["hidden", "hidden_shift", "diff_sublayer", "diff_layer", "consecutive_sublayer", "consecutive_layer"],
        help="Type of features to use for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training the probe",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the probe",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--l2_penalty",
        type=float,
        default=0.001,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "--pn_ratio",
        type=float,
        default=0.25,
        help="The ratio of positive samples in training data dividing the negative",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        help="Type of optimizer to use for training",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--train_sample_num",
        type=int,
        default=100,
        help="Number of training samples. We should ensure similar positive/negative ratio to the whole training set."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for reproducibility",
    )
    parser.add_argument(
        "--layer_index",
        type=str,
        default=10,
        help="Layer index to use for training the probe",
    )
    parser.add_argument(
        "--eval_layer_index",
        type=str,
        default=None,
        help="Layer index to use for evaluation",
    )
    parser.add_argument(
        "--projection_components",
        type=int,
        default=10,
        help="Number of main components",
    )
    parser.add_argument(
        "--projection_type",
        type=str,
        default=None,
        help="Type of projection to use",
    )
    parser.add_argument(
        "--train_method",
        type=str,
        default="binary_only",
        help="How to adjust the training loss and labels",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Used to balance different loss terms if necessary",
    )
    parser.add_argument(
        "--base",
        type=float,
        default=0.5,
        help="Used to set the starting point for soft label of non-leakage cases",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2,
        help="Margin for the ranking loss"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize the features",
    )
    parser.add_argument(
        "--result_averaging",
        action="store_true",
        help="Whether to report mean/std of results across different seeds",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Whether to use ensemble",
    )
    parser.add_argument(
        "--souping",
        action="store_true",
        help="Whether to use souping",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for the probe",
    )
    parser.add_argument(
        "--bucketize",
        type=int,
        default=None,
        help="Number of buckets to bucketize the labels into",
    )
    parser.add_argument(
        "--best_single",
        action="store_true",
        help="Only use the best single probe (on val set) among all seeds"
    )
    parser.add_argument(
        "--draw_roc_to",
        type=str,
        default=None,
        help="Draw the ROC curve for the best single probe",
    )
    return parser.parse_args() 

if __name__ == "__main__":

    args = parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # optional normalization from reference dataset
    if args.reference_normal_data_path is not None and os.path.exists(args.reference_normal_data_path):
        ref_feats = load_hidden_feats(args.reference_normal_data_path)
    else:
        ref_feats = None

    training_feats = load_hidden_feats(args.train_data_path)
    train_dataset = build_dataset(
        training_feats, layer_index=args.layer_index, 
        num_samples=args.train_sample_num, pn_ratio=args.pn_ratio, 
        feature_type=args.feature_type, projection_components=args.projection_components, projection_type=args.projection_type, 
        projector=None, ref_feats=ref_feats, normalize=args.normalize, bucketize=args.bucketize
    )
    projector_model = train_dataset.projector
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.normalize:
        normalize_mean = train_dataset.normalize_mean
        normalize_std = train_dataset.normalize_std
    else:
        normalize_mean = None
        normalize_std = None

    test_set_names = ("val", "heldout_attacks", "heldout_systems", "heldout_strict")
    test_set_paths = (args.val_data_path, args.heldout_attacks_data_path, args.heldout_systems_data_path, args.heldout_strict_data_path)

    eval_layer_index = args.eval_layer_index if args.eval_layer_index is not None else args.layer_index
    eval_feature_type = args.eval_feature_type if args.eval_feature_type is not None else args.feature_type

    test_set_loaders = []
    for test_set_name, test_set_path in zip(test_set_names, test_set_paths):
        print("-" * 40)
        print(f"Loading {test_set_name} set from {test_set_path}")

        if test_set_path is None or not os.path.exists(test_set_path): continue
        test_feats = load_hidden_feats(test_set_path)
        # do not bucketize the labels of test set
        test_dataset = build_dataset(
            test_feats, layer_index=eval_layer_index, 
            feature_type=eval_feature_type, projection_components=args.projection_components, 
            projection_type=args.projection_type, projector=projector_model,
            normalize_mean=normalize_mean, normalize_std=normalize_std, normalize=args.normalize,
        )
        print(f"Loaded {test_set_name} set with {len(test_dataset)} samples")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_set_loaders.append(
            (test_set_name, test_loader)
        )

    probe_cls = load_probe(args.probe_type)

    probes = []
    for seed in args.seeds:
        set_seed(seed)
        probe = probe_cls(in_features=train_dataset[0]["features"].shape[0],
                          initialization=args.initialization_type,
                          dropout_rate=args.dropout_rate)

        best_probe = train_probe(
            probe=probe,
            train_loader=train_loader,
            test_loaders=test_set_loaders,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            l2_penalty=args.l2_penalty,
            optimizer_type=args.optimizer_type,
            device=device,
            output_dir=args.output_dir,
            eval_metrics=args.eval_metrics,
            train_method=args.train_method,
            alpha=args.alpha,
            base=args.base,
            margin=args.margin
        )
        probes.append(best_probe)

    if args.result_averaging:
        print("\n==> Evaluating Result Averaging")
        probe_results_by_seeds = []

        for probe in probes:
            probe.eval()
            result_per_probe = {}
            for name, loader in test_set_loaders:
                result_per_probe[name] = eval_probe_single(probe, loader, device, args.eval_metrics)
            probe_results_by_seeds.append(result_per_probe)

        for set_name in test_set_names:
            all_metrics = {metric: [] for metric in args.eval_metrics}
            for result_dict in probe_results_by_seeds:
                if set_name not in result_dict:
                    continue
                for metric in args.eval_metrics:
                    if metric in result_dict[set_name]:
                        all_metrics[metric].append(result_dict[set_name][metric])

            print(f"\n{set_name.upper()} (Seed-wise Averaging):")
            for metric, values in all_metrics.items():
                if len(values) == 0:
                    continue
                mean = np.mean(values)
                std = np.std(values)
                print(f"{metric:>20}: {mean:.4f} ± {std:.4f}")


    if args.ensemble:
        print("\n==> Evaluating Ensemble Averaged Predictions")
        for name, loader in test_set_loaders:
            results = eval_probe_ensemble(probes, loader, device, args.eval_metrics)
            print(f"\n{name.upper()} (Ensemble):")
            for metric, score in results.items():
                print(f"{metric:>20}: {score:.4f}")

    if args.souping:
        print("\n==> Evaluating Souped Model (Parameter Averaging)")
        soup_probe = average_model_soup(probes).to(device)
        for name, loader in test_set_loaders:
            results = eval_probe_single(soup_probe, loader, device, args.eval_metrics)
            print(f"\n{name.upper()} (Souped):")
            for metric, score in results.items():
                print(f"{metric:>20}: {score:.4f}")

    if args.best_single:
        print("\n==> Selecting Best Single Probe Based on Validation Set")
        best_probe = None
        best_score = -float("inf")
        best_results = None

        for probe in probes:
            val_loader = dict(test_set_loaders).get("val", None)
            if val_loader is None:
                raise ValueError("Validation set is required for --best_single mode.")
            result = eval_probe_single(probe, val_loader, device, args.eval_metrics)
            score = result.get(args.eval_metrics[0], -float("inf"))  # Use the first metric as main
            if score > best_score:
                best_score = score
                best_probe = probe
                best_results = result

        print(f"\nSelected Probe (Best on VAL {args.eval_metrics[0]}: {best_score:.4f})")

        for name, loader in test_set_loaders:
            results = eval_probe_single(best_probe, loader, device, args.eval_metrics)
            print(f"\n{name.upper()} (Best Single Probe):")
            for metric, score in results.items():
                print(f"{metric:>20}: {score:.4f}")
        
        if args.output_dir is not None:
            torch.save(best_probe.state_dict(), os.path.join(args.output_dir, f"best_probe_{args.eval_metrics[0]}.pt"))

        if args.draw_roc_to is not None:
            draw_the_probe_rocs(best_probe, test_set_loaders, device, args.draw_roc_to)
"""
We split the dataset into
- train_dataset
- held-in validation dataset
- held-out evaluation dataset
  - new attacks
  - new systems
"""

import os
import argparse
import random
import torch

def split_dataset(data, val_ratio=0.1, test_ratio=0.1, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    features = data["features"]  # Dict[str, Tensor]
    labels = data["labels"]
    systems = data["systems"]
    samples = data["samples"]
    responses = data["responses"]

    # check the attacks
    attacks = []
    missing_attacks = 0
    for samp, sys in zip(samples, systems):
        if sys in samp:
            attacks.append(samp.split(sys)[-1].strip())
        else:
            attacks.append(samp)  # fallback if system is not in sample
            missing_attacks += 1

    if missing_attacks > 0:
        raise ValueError(f"Warning: {missing_attacks} samples did not contain their system name.")

    total = len(labels)
    assert all(len(x) == total for x in [systems, samples, responses])

    indices = list(range(total))

    # Get unique systems and samples
    all_systems = sorted(set(systems))
    all_attacks = sorted(set(attacks))

    # Define held-out systems and held-out attacks
    num_heldout_sys = max(0, int(test_ratio * len(all_systems)))
    num_heldout_attacks = max(0, int(test_ratio * len(all_attacks)))

    heldout_systems = set(random.sample(all_systems, num_heldout_sys))
    heldout_attacks = set(random.sample(all_attacks, num_heldout_attacks))

    # Index grouping
    train_idx, val_idx = [], []
    heldout_sys_idx, heldout_attack_idx, heldout_strict_idx = [], [], []

    for idx in indices:
        sys, atk = systems[idx], attacks[idx]
        in_held_sys = sys in heldout_systems
        in_held_atk = atk in heldout_attacks

        if in_held_sys and in_held_atk:
            heldout_strict_idx.append(idx)
        elif in_held_sys:
            heldout_sys_idx.append(idx)
        elif in_held_atk:
            heldout_attack_idx.append(idx)
        else:
            train_idx.append(idx)

    # Now split train into train/val
    random.shuffle(train_idx)
    split_point = int(len(train_idx) * (1 - val_ratio))
    final_train_idx = train_idx[:split_point]
    final_val_idx = train_idx[split_point:]

    def subset(idx_list):
        return {
            "features": {k: v[idx_list] for k, v in features.items()},
            "labels": labels[idx_list],
            "systems": [systems[i] for i in idx_list],
            "samples": [samples[i] for i in idx_list],
            "responses": [responses[i] for i in idx_list],
            "attacks": [attacks[i] for i in idx_list],
        }

    return {
        "train": subset(final_train_idx),
        "val": subset(final_val_idx),
        "heldout_systems": subset(heldout_sys_idx),
        "heldout_attacks": subset(heldout_attack_idx),
        "heldout_strict": subset(heldout_strict_idx),
    }

def split_dataset_with_reference(data, reference_split):

    if not os.path.exists(reference_split):
        raise FileNotFoundError(f"Reference split {reference_split} does not exist.")

    # read splits from reference_split
    split_names = ["train", "val", "heldout_systems", "heldout_attacks", "heldout_strict"]
    splits = {name: torch.load(os.path.join(reference_split, f"{name}.pt"))["samples"] for name in split_names}

    # Use samples as reference

    features = data["features"]  # Dict[str, Tensor]
    labels = data["labels"]
    systems = data["systems"]
    samples = data["samples"]
    responses = data["responses"]

    # check the attacks
    attacks = []
    missing_attacks = 0
    for samp, sys in zip(samples, systems):
        if sys in samp:
            attacks.append(samp.split(sys)[-1].strip())
        else:
            attacks.append(samp)  # fallback if system is not in sample
            missing_attacks += 1

    if missing_attacks > 0:
        raise ValueError(f"Warning: {missing_attacks} samples did not contain their system name.")

    total = len(labels)
    assert all(len(x) == total for x in [systems, samples, responses])

    indices = list(range(total))

    # Index grouping
    train_idx, val_idx = [], []
    heldout_sys_idx, heldout_attack_idx, heldout_strict_idx = [], [], []

    for idx in indices:
        sample = samples[idx]
        if sample in splits["train"]:
            train_idx.append(idx)
        elif sample in splits["val"]:
            val_idx.append(idx)
        elif sample in splits["heldout_systems"]:
            heldout_sys_idx.append(idx)
        elif sample in splits["heldout_attacks"]:
            heldout_attack_idx.append(idx)
        else:
            heldout_strict_idx.append(idx)

    print(train_idx, val_idx, heldout_sys_idx, heldout_attack_idx, heldout_strict_idx)

    def subset(idx_list):
        return {
            "features": {k: v[idx_list] for k, v in features.items()},
            "labels": labels[idx_list],
            "systems": [systems[i] for i in idx_list],
            "samples": [samples[i] for i in idx_list],
            "responses": [responses[i] for i in idx_list],
            "attacks": [attacks[i] for i in idx_list],
        }

    return {
        "train": subset(train_idx),
        "val": subset(val_idx),
        "heldout_systems": subset(heldout_sys_idx),
        "heldout_attacks": subset(heldout_attack_idx),
        "heldout_strict": subset(heldout_strict_idx),
    }


def split_dataset_with_reference_by_system(data, reference_split):

    if not os.path.exists(reference_split):
        raise FileNotFoundError(f"Reference split {reference_split} does not exist.")

    # read splits from reference_split
    split_names = ["train", "val", "heldout_systems", "heldout_attacks", "heldout_strict"]
    splits = {name: torch.load(os.path.join(reference_split, f"{name}.pt"))["systems"] for name in split_names}


    print({name: len(splits[name]) for name in split_names})
    # Use samples as reference

    features = data["features"]  # Dict[str, Tensor]
    labels = data["labels"]
    systems = data["systems"]
    samples = data["samples"]
    responses = data["responses"]

    # check the attacks
    attacks = []
    missing_attacks = 0
    for samp, sys in zip(samples, systems):
        if sys in samp:
            attacks.append(samp.split(sys)[-1].strip())
        else:
            attacks.append(samp)  # fallback if system is not in sample
            missing_attacks += 1

    if missing_attacks > 0:
        raise ValueError(f"Warning: {missing_attacks} samples did not contain their system name.")

    total = len(labels)
    assert all(len(x) == total for x in [systems, samples, responses])

    indices = list(range(total))

    # Index grouping
    train_idx = []

    for idx in indices:
        sys = systems[idx][:-40] # Remove the role-playing of user role
        # Check if sys is a substring of any system in splits["train"]
        if any(sys in train_sys for train_sys in splits["train"]):
            train_idx.append(idx)
        else:
            continue

    def subset(idx_list):
        return {
            "features": {k: v[idx_list] for k, v in features.items()},
            "labels": labels[idx_list],
            "systems": [systems[i] for i in idx_list],
            "samples": [samples[i] for i in idx_list],
            "responses": [responses[i] for i in idx_list],
            "attacks": [attacks[i] for i in idx_list],
        }

    return {
        "train": subset(train_idx),
    }



def measure_each_split(data, split_name):

    labels = data["labels"]
    systems = data["systems"]
    samples = data["samples"]
    responses = data["responses"]
    attacks = data["attacks"]

    total = len(labels)
    assert all(len(x) == total for x in [systems, samples, responses])

    # Get unique systems and samples
    all_systems = sorted(set(systems))
    all_attacks = sorted(set(attacks))

    print("-"*40 + f" {split_name} " + "-"*40)
    print(f"Total samples: {total}")
    print(f"Unique systems: {len(all_systems)}")
    print(f"Unique attacks: {len(all_attacks)}")
    print(f"Positive samples: {sum(labels > 0)}", f"Negative samples: {total - sum(labels > 0)}")
    if "heldout_systems" in split_name:
        print(f"Heldout systems:")
        for idx, sys in enumerate(set(systems)):
            print(f"  System {idx}:\n{sys}")
    if "heldout_attacks" in split_name:
        print(f"Heldout attacks:")
        for idx, attack in enumerate(set(attacks)):
            print(f"  Attack {idx}:\n{attack}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    print(f"Loading dataset from {args.input}")
    data = torch.load(args.input)

    print("Splitting dataset...")
    if args.reference_split is None:
        splits = split_dataset(data, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    else:
        if args.split_by_system:
            splits = split_dataset_with_reference_by_system(data, args.reference_split)
        else:
            splits = split_dataset_with_reference(data, args.reference_split)

    for split_name, split_data in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.pt")
        torch.save(split_data, path)
        measure_each_split(split_data, split_name)
        print(f"==> Saved {split_name} set with {len(split_data['labels'])} examples to {path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .pt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save split datasets")
    parser.add_argument("--split_by_system", action="store_true", help="Split dataset by system. (For locating normal queries corresponding to the attacks in the dataset.)")
    parser.add_argument("--reference_split", type=str, default=None, help="Which split to use as reference. Should be a directory with splited datasets.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
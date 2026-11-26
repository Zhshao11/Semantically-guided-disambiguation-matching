"""Auditable DINO patch teacher, train-only spherical prototypes and versioned cache."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .io import save_json, save_jsonl
from .raw_lineage import sha256_file
from .semantic_routing import probability_normalize, spherical_kmeans


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def teacher_preprocess(gray, input_hw=(252, 252)):
    if gray.ndim != 4 or gray.shape[1:] != (1, 256, 256):
        raise ValueError("Teacher requires native256 grayscale, not quarantined RGB512")
    rgb = (gray.float() / 255).expand(-1, 3, -1, -1)
    rgb = F.interpolate(rgb, size=input_hw, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (rgb - mean) / std


def teacher_identity(config):
    teacher = config["teacher"]
    source = Path(teacher["source_directory"])
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != teacher["source_revision"]:
        raise ValueError("Wrong DINO source revision")
    # A git revision alone would not detect local edits to the implementation.
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Locally modified DINO source")
    source_hash = json_digest({str(p.relative_to(source)): sha256_file(p) for p in sorted((source / "dinov2").rglob("*.py"))})
    return dict(
        **teacher,
        weights_sha256=sha256_file(teacher["weights_path"]),
        source_python_tree_sha256=source_hash,
        wrapper_sha256=sha256_file(__file__),
        prototype_math_sha256=sha256_file(Path(__file__).with_name("semantic_routing.py")),
        cache_schema="dinov2_patch_proto_v1_fp16_storage_fp32_renormalized",
        xformers="disabled explicitly; official PyTorch attention fallback",
        torch_version=torch.__version__,
    )


def load_teacher(config, device):
    identity = teacher_identity(config)
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(Path(config["teacher"]["source_directory"]).resolve()))
    from dinov2.hub.backbones import dinov2_vits14

    model = dinov2_vits14(pretrained=False)
    weights = torch.load(config["teacher"]["weights_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, identity


def prototype_sample(records, token_count, config):
    groups = defaultdict(list)
    for i, row in enumerate(records):
        if row["split"] != "train":
            raise ValueError("Only training families may fit prototypes")
        groups[row["family_id"]].append(i)
    rng = np.random.default_rng(config["seed"])
    selected = []
    for family, ids in sorted(groups.items()):
        for index in sorted(rng.choice(ids, config["pairs_per_training_family"], replace=False).tolist()):
            selected.append(dict(
                sample_id=records[index]["sample_id"], family_id=family, cache_row=index,
                native_hw=[256,256], split="train",
                optical_sha256=records[index]["optical_sha256"],
                token_indices=sorted(rng.choice(token_count, config["tokens_per_pair"], replace=False).tolist()),
            ))
    return selected


@torch.no_grad()
def build_teacher_cache(config, training, validation, device):
    root = Path(config["output"]) / "teacher"
    if root.exists():
        raise FileExistsError("Use new output for a new teacher/prototype version")
    root.mkdir(parents=True)
    model, identity = load_teacher(config, device)
    datasets = [training, validation]
    records = training.records + validation.records
    if {r["family_id"] for r in training.records} & {r["family_id"] for r in validation.records}:
        raise ValueError("Prototype training and validation family overlap")
    save_jsonl(root / "cache_samples.jsonl", records)
    save_json(root / "teacher_identity.json", identity)
    rows = len(records)
    tokens = np.lib.format.open_memmap(root / "patch_tokens.npy", mode="w+", dtype="float16", shape=(rows,324,384))
    batch_size = config["teacher"]["batch_size"]
    offset = 0
    for dataset in datasets:
        for start in range(0, len(dataset), batch_size):
            native = torch.from_numpy(dataset.arrays[start:start+batch_size, 0].copy()).to(device)[:,None]
            patch = model.forward_features(teacher_preprocess(native))["x_norm_patchtokens"]
            if patch.shape[1:] != (324,384) or not torch.isfinite(patch).all():
                raise AssertionError("Wrong/nonfinite patch tokens; CLS must be excluded")
            patch = F.normalize(patch, dim=-1)
            tokens[offset+start:offset+start+len(native)] = patch.cpu().numpy().astype("float16")
            if start % (batch_size*20) == 0:
                print(json.dumps(dict(stage="DINO_cache", split=dataset.split, images=start, total=len(dataset))), flush=True)
        offset += len(dataset)
    tokens.flush()
    del model
    torch.cuda.empty_cache()
    chosen = prototype_sample(training.records, 324, config["prototypes"])
    save_jsonl(root / "prototype_samples.jsonl", chosen)
    features = torch.from_numpy(np.concatenate([tokens[r["cache_row"], r["token_indices"]].astype("float32") for r in chosen])).to(device)
    centers, counts, history = spherical_kmeans(features, config["prototypes"]["count"], config["prototypes"]["spherical_iterations"], config["prototypes"]["seed"])
    np.save(root / "prototype_centers.npy", centers.cpu().numpy())
    save_json(root / "prototype_fit.json", dict(
        sampled_families=len({r["family_id"] for r in chosen}), sampled_pairs=len(chosen),
        sampled_tokens=len(features), occupancy_counts=counts, iterations=history,
        labels="latent prototypes only; no land-cover class names",validation_used=False,RGB512_used=False,
    ))
    del features
    probabilities = np.lib.format.open_memmap(root / "optical_probabilities.npy", mode="w+", dtype="float16", shape=(rows,32,32,32))
    for start in range(0, rows, batch_size):
        patch = torch.from_numpy(np.array(tokens[start:start+batch_size], dtype="float32")).to(device).transpose(1,2).reshape(-1,384,18,18)
        dense = F.normalize(F.interpolate(patch, size=(32,32), mode="bilinear", align_corners=False), dim=1)
        q = torch.einsum("bchw,mc->bmhw",dense,centers).div(config["prototypes"]["temperature"]).softmax(1)
        probabilities[start:start+len(q)] = q.cpu().numpy().astype("float16")
    probabilities.flush()
    manifest = dict(
        teacher=identity, prototypes=config["prototypes"],
        prototype_centers_sha256=sha256_file(root / "prototype_centers.npy"),
        prototype_samples_sha256=sha256_file(root / "prototype_samples.jsonl"),
        cache_samples_sha256=sha256_file(root / "cache_samples.jsonl"),
        train_samples=len(training),validation_samples=len(validation),
        native_to_teacher_chain="native256 -> bicubic252 -> patch14/18grid -> bilinear32structgrid -> L2 -> cosine32centers/Tsem -> softmax",
        optical_transform="identity in all current train/eval plans; no cached augmented optical views reused",
        storage="normalized patch features and q stored float16; every q read float32 and renormalized",
    )
    manifest["cache_version"] = json_digest(manifest)
    manifest["file_hashes"] = {name:sha256_file(root/name) for name in ["patch_tokens.npy","optical_probabilities.npy","prototype_centers.npy","prototype_samples.jsonl","cache_samples.jsonl","teacher_identity.json","prototype_fit.json"]}
    save_json(root / "cache_receipt.json", manifest)
    return manifest


def verify_teacher_cache(root, config, verify_large=True):
    root = Path(root)
    receipt = json.loads((root / "cache_receipt.json").read_text())
    if receipt["teacher"] != teacher_identity(config) or receipt["prototypes"] != config["prototypes"]:
        raise ValueError("Teacher model, preprocessing, math or prototype configuration mismatch")
    content = {k:v for k,v in receipt.items() if k not in ["cache_version","file_hashes"]}
    if json_digest(content) != receipt["cache_version"]:
        raise ValueError("Cache metadata version mismatch")
    for name, digest in receipt["file_hashes"].items():
        if verify_large or name not in ["patch_tokens.npy","optical_probabilities.npy"]:
            if sha256_file(root/name) != digest:
                raise ValueError(f"Teacher cache changed: {name}")
    return receipt


class TeacherView(Dataset):
    def __init__(self, view, root):
        self.view = view
        self.records = view.records
        self.split = view.split
        root = Path(root)
        rows = [json.loads(line) for line in (root/"cache_samples.jsonl").read_text().splitlines()]
        lookup = {r["sample_id"]:i for i,r in enumerate(rows)}
        self.indices = [lookup[r["sample_id"]] for r in view.records]
        for row, index in zip(view.records, self.indices):
            for key in ["family_id","split","optical_sha256","sar_sha256"]:
                if row[key] != rows[index][key]:
                    raise ValueError("Cached teacher/sample provenance mismatch")
        self.probabilities = np.load(root/"optical_probabilities.npy", mmap_mode="r")

    def __len__(self):
        return len(self.view)

    def __getitem__(self,index):
        item = self.view[index]
        p = torch.from_numpy(np.array(self.probabilities[self.indices[index]],dtype="float32"))
        item["teacher"] = probability_normalize(p[None])[0]
        return item

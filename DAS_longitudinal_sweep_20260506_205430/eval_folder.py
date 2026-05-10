import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image

import das.rewards as rewards


PROMPT_RE = re.compile(r"^\d+_(.*) \| reward: .*\.(png|jpg|jpeg)$", re.IGNORECASE)


def _list_images(eval_vis_dir: str) -> List[str]:
    files = []
    for name in os.listdir(eval_vis_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if "ess" in name or "intermediate_rewards" in name:
            continue
        files.append(os.path.join(eval_vis_dir, name))
    files.sort()
    return files


def _prompt_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = PROMPT_RE.match(name)
    if m:
        return m.group(1)
    # fallback: strip leading index and everything after first "|"
    base = name
    if "|" in base:
        base = base.split("|")[0].strip()
    base = re.sub(r"^\d+_", "", base)
    return base.strip().rstrip(".")


def _load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def _score_all(
    image_paths: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, List[float]]:
    aesthetic_fn = rewards.aesthetic_score(torch_dtype=torch.float32, device=device)
    pick_fn = rewards.PickScore(inference_dtype=torch.float32, device=device)
    clip_fn = rewards.clip_score(inference_dtype=torch.float32, device=device)
    imagereward_fn = rewards.ImageReward(inference_dtype=torch.float32, device=device)

    # HPSv2 is optional (often requires a separate install method).
    hps_fn = None
    try:
        hps_fn = rewards.hps_score(inference_dtype=torch.float32, device=device)
    except Exception:
        hps_fn = None

    out: Dict[str, List[float]] = {
        "aesthetic": [],
        "pick": [],
        "clip": [],
        "imagereward": [],
    }
    if hps_fn is not None:
        out["hps"] = []

    for path in image_paths:
        prompt = _prompt_from_filename(path)
        image = _load_image_tensor(path, device=device)

        out["aesthetic"].append(float(aesthetic_fn(image, prompt).item()))
        out["pick"].append(float(pick_fn(image, [prompt]).item()))
        out["clip"].append(float(clip_fn(image, [prompt]).item()))
        out["imagereward"].append(float(imagereward_fn(image, [prompt]).item()))
        if hps_fn is not None:
            out["hps"].append(float(hps_fn(image, [prompt]).item()))

    return out


@torch.no_grad()
def _diversity_metrics(
    image_paths: List[str],
    device: torch.device,
    K: int = 20,
) -> Tuple[float, float, float, float, float]:
    from transformers import CLIPModel, CLIPProcessor
    from scipy.spatial.distance import pdist
    import lpips
    from torchvision import transforms

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lpips_model = lpips.LPIPS(net="alex").to(device)

    def preprocess_clip(img: Image.Image) -> torch.Tensor:
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    preprocess_lp = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    embeddings = []
    lp_tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        pixel_values = preprocess_clip(img).unsqueeze(0).to(device)
        emb = model.get_image_features(pixel_values).detach().cpu().numpy().squeeze()
        embeddings.append(emb)

        lp_tensors.append(preprocess_lp(img).unsqueeze(0).to(device))

    embeddings = np.asarray(embeddings)
    pairwise_distances = pdist(embeddings, metric="cosine")
    mean_distance = float(np.mean(pairwise_distances))
    std_error = float(np.std(pairwise_distances) / np.sqrt(pairwise_distances.size))

    covariance_matrix = np.cov(embeddings, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)[-K:]
    # numerical guard
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    TCE_K = float((K / 2) * np.log(2 * np.pi * np.e) + (1 / 2) * np.sum(np.log(eigenvalues)))

    lpips_distances = []
    for i in range(len(lp_tensors)):
        for j in range(i + 1, len(lp_tensors)):
            lpips_distances.append(float(lpips_model(lp_tensors[i], lp_tensors[j]).item()))

    mean_lpips = float(np.mean(lpips_distances)) if lpips_distances else 0.0
    std_lpips = float(np.std(lpips_distances)) if lpips_distances else 0.0

    return mean_distance, std_error, TCE_K, mean_lpips, std_lpips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", required=True, help="实验目录，例如 logs/DAS_LCM/pick/2026...")
    parser.add_argument("--K", type=int, default=20)
    args = parser.parse_args()

    img_folder = args.img_folder
    eval_vis_dir = os.path.join(img_folder, "eval_vis")
    if not os.path.isdir(eval_vis_dir):
        raise FileNotFoundError(f"Missing eval_vis folder: {eval_vis_dir}")

    image_paths = _list_images(eval_vis_dir)
    if not image_paths:
        raise ValueError(f"No images found under: {eval_vis_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores = _score_all(image_paths=image_paths, device=device, dtype=torch.float32)

    # Save score summary
    names = []
    values = []
    for key in ["aesthetic", "hps", "imagereward", "pick", "clip"]:
        if key not in scores:
            continue
        arr = np.asarray(scores[key], dtype=np.float32)
        names.extend([f"{key}_mean", f"{key}_std"])
        values.extend([float(arr.mean()), float(arr.std())])

    with open(os.path.join(img_folder, "eval_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerow([f"{v:.6f}" for v in values])

    mean_distance, std_error, TCE, mean_lpips, std_lpips = _diversity_metrics(
        image_paths=image_paths,
        device=device,
        K=args.K,
    )

    with open(os.path.join(img_folder, "eval_diversity_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mean_pairwise_distance_clip",
                "std_error_pairwise_distance_clip",
                f"tce_K{args.K}",
                "mean_lpips",
                "std_lpips",
            ]
        )
        writer.writerow([f"{v:.6f}" for v in [mean_distance, std_error, TCE, mean_lpips, std_lpips]])

    print("Done")
    print("-", os.path.join(img_folder, "eval_results.csv"))
    print("-", os.path.join(img_folder, "eval_diversity_results.csv"))


if __name__ == "__main__":
    main()

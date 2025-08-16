#!/usr/bin/env python3
"""
Batch keyframe extraction for all videos in a folder.

- Input videos:   ./videos/*.mp4
- Input scenes:   ./Autoshot/{video_name}_scenes.csv
- Output:         ./keyframes/{video_name}/ + {video_name}_keyframes_info.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoFeatureExtractor, AutoModel
import csv

# ---------- utils ----------
def laplacian_var(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_std(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray) / 255.0)

def cosine_sim_matrix(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

def pil_from_bgr(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

# ---------- model helpers ----------
def load_model_and_processor(model_name, device, use_auth_token=False):
    kwargs = {"use_auth_token": os.getenv("HF_TOKEN")} if use_auth_token else {}
    processor = AutoFeatureExtractor.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
    model.eval()
    return processor, model

def embed_batch(pil_images, processor, model, device):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embs = outputs.pooler_output.cpu().numpy()
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        embs = outputs.image_embeds.cpu().numpy()
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    else:
        embs = model.get_image_features(**inputs).cpu().numpy()
    return embs

# ---------- main selection ----------
def select_keyframes(video_path, scenes_csv, out_dir,
                     model_name="microsoft/beit-base-patch16-224",
                     sample_n=4, tau=0.30,
                     quality_blur_thresh=100.0, min_brightness_std=0.02,
                     batch_size=16, use_hf_token=False, device=None):
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load scenes
    df = pd.read_csv(scenes_csv)
    if not {"start_frame", "end_frame"}.issubset(df.columns):
        if df.shape[1] >= 2:
            df.columns = ["start_frame", "end_frame"] + list(df.columns[2:])
        else:
            raise ValueError("CSV must contain start_frame and end_frame columns.")

    print("Loading model:", model_name)
    processor, model = load_model_and_processor(model_name, device, use_auth_token=use_hf_token)

    metadata = []
    cap = cv2.VideoCapture(video_path)

    for shot_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(video_path)}"):
        s = int(row["start_frame"])
        e = int(row["end_frame"])
        if s > e:
            s, e = e, s

        sampled = list(range(s, e+1, sample_n))
        if len(sampled) == 0:
            sampled = [(s + e) // 2]

        frames_bgr, pil_imgs, valid_idxs = [], [], []
        sharpness, brightness = {}, {}

        for idx in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames_bgr.append((idx, frame))
            pil_imgs.append(pil_from_bgr(frame))
            valid_idxs.append(idx)
            sharpness[idx] = laplacian_var(frame)
            brightness[idx] = brightness_std(frame)

        if len(valid_idxs) == 0:
            continue

        all_embs = []
        for i in range(0, len(pil_imgs), batch_size):
            batch = pil_imgs[i:i+batch_size]
            embs = embed_batch(batch, processor, model, device)
            all_embs.append(embs)
        all_embs = np.vstack(all_embs)
        idx_to_emb = {idx: all_embs[i] for i, idx in enumerate(valid_idxs)}

        primary_idx = max(valid_idxs, key=lambda x: sharpness.get(x, -1.0))
        selected_keyframes = [primary_idx]
        last_selected_emb = idx_to_emb[primary_idx]

        sorted_idxs = sorted(valid_idxs)
        for idx in sorted_idxs:
            if idx == primary_idx:
                continue
            if sharpness.get(idx, 0.0) < quality_blur_thresh:
                continue
            if brightness.get(idx, 0.0) < min_brightness_std:
                continue

            curr_emb = idx_to_emb[idx]
            prev_idx = sorted_idxs[sorted_idxs.index(idx) - 1] if sorted_idxs.index(idx) > 0 else None
            dist_prev = 1.0 - cosine_sim_matrix(curr_emb.reshape(1, -1), idx_to_emb[prev_idx].reshape(1, -1))[0][0] if prev_idx else 0.0
            dist_last = 1.0 - cosine_sim_matrix(curr_emb.reshape(1, -1), last_selected_emb.reshape(1, -1))[0][0]

            if dist_prev >= tau and dist_last >= tau:
                selected_keyframes.append(idx)
                last_selected_emb = curr_emb

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(primary_idx))
        ret, frame_p = cap.read()
        primary_file = os.path.join(out_dir, f"shot{shot_idx:04d}_main_frame{primary_idx:06d}.webp")
        Image.fromarray(cv2.cvtColor(frame_p, cv2.COLOR_BGR2RGB)).save(primary_file, "WEBP", quality=90)

        extra_files = []
        for j, ex_idx in enumerate([f for f in selected_keyframes if f != primary_idx]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ex_idx))
            ret, frame_ex = cap.read()
            if not ret:
                continue
            fname = os.path.join(out_dir, f"shot{shot_idx:04d}_extra{j}_frame{ex_idx:06d}.webp")
            Image.fromarray(cv2.cvtColor(frame_ex, cv2.COLOR_BGR2RGB)).save(fname, "WEBP", quality=90)
            extra_files.append(fname)

        metadata.append((shot_idx, s, e, primary_idx, primary_file, ";".join([str(x) for x in selected_keyframes if x != primary_idx]), ";".join(extra_files)))

    cap.release()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    meta_csv = os.path.join(out_dir, f"{video_name}_keyframes_info.csv")
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot_idx", "shot_start", "shot_end", "primary_frame", "primary_file", "extra_frames", "extra_files"])
        for m in metadata:
            writer.writerow(m)
    print("Done:", meta_csv)


# ---------- batch runner ----------
if __name__ == "__main__":
    videos_dir = "videos"
    scenes_dir = "Autoshot/scenes_csv"
    out_root = "keyframes"

    os.makedirs(out_root, exist_ok=True)

    for fname in os.listdir(videos_dir):
        if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = os.path.join(videos_dir, fname)
        video_name = os.path.splitext(fname)[0]
        scenes_csv = os.path.join(scenes_dir, f"{video_name}_scenes.csv")

        if not os.path.exists(scenes_csv):
            print(f"⚠️ CSV not found for {video_name}, skipping")
            continue

        out_dir = os.path.join(out_root, video_name)
        select_keyframes(
            video_path=video_path,
            scenes_csv=scenes_csv,
            out_dir=out_dir,
            model_name="microsoft/beit-base-patch16-224",
            sample_n=8,
            tau=0.30,
            quality_blur_thresh=100.0,
            min_brightness_std=0.02,
            batch_size=16,
            use_hf_token=False,
            device=None
        )

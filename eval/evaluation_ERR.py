"""
ERR (Erasing-Retention-Robustness) evaluation script -- CLIPAccuracy version.

Follows the FADE-style clip_score / clip_accuracy formulation
(https://github.com/IAB-IITJ/FADE) rather than raw CLIPScore, because the
task's ERR formulas define each sub-score as a per-image, per-concept binary
term A^(i,c), pooled over N_img x N_c:

    A_x = (1 / (N_img * N_c)) * sum_i sum_c A_x^(i,c)

That's exactly what CLIPAccuracy computes (a 0/1 "did this image match text A
more than text B" indicator per image), whereas raw CLIPScore is a continuous,
unbounded number whose scale drifts across concept types -- it needs an
arbitrary normalization to be combined via harmonic mean. Accuracy doesn't.

Concretely, for every image we compare its CLIP similarity to two texts:
  - the concept's own descriptive prompt ("target text")
  - a fixed NEUTRAL_ANCHOR caption (default: "a photo"), representing "no
    longer recognizable as anything concept-specific"

  Forgetting axes (A_fgt, A_ind, A_adv): images generated from the direct /
  indirect / adversarial prompt. Success = anchor scores HIGHER than the
  target concept's own prompt (the image no longer reads as the target).
      A^(i,c) = 1[ CLIP(img, anchor) > CLIP(img, target_prompt_c) ]

  Retention axes (A_ret, A_adj): images generated from a retained / adjacent
  concept's own prompt. Success = that concept's own prompt scores HIGHER
  than the anchor (the image still clearly reads as that concept).
      A^(i,c) = 1[ CLIP(img, concept_prompt) > CLIP(img, anchor) ]

NOTE on the anchor: the CSV/task doesn't hand us an explicit "fallback" text
per concept (unlike e.g. nudity-erasure work where anchor="a person" vs.
ablated="a naked person"). We use one fixed neutral caption for everything
via --neutral_anchor. If you'd rather use a per-concept-type anchor (e.g. "a
painting" for Artistic Styles, "a photo of an object" for Objects, etc.),
say so and I'll parametrize it by concept_type instead.

Expected image layout (unchanged from before):

<images_dir>/
  <concept_name>/
    direct/                              *.png  -- from direct_prompt
    indirect/                            *.png  -- from indirect_prompt
    adversarial/                         *.png  -- from adversarial_prompt
    adjacent/<adjacent_concept_name>/    *.png  -- from "an image of <adjacent_concept_name>"
    retained/<retained_concept_name>/    *.png  -- from that concept's OWN direct_prompt (looked up from the CSV)

Aggregation matches the pooled-sum formulas in the task image exactly (not an
average-of-per-concept-averages): every individual image-level indicator
across every concept is collected into one flat list per axis, then averaged
once.

Final score:
  ERR = harmonic_mean(A_fgt, A_ret, A_adj, A_ind, A_adv)

Tie-breaker (optional, needs both checkpoints):
  Weight-Change Ratio = (1/N_c) * sum_c ||theta_orig - theta_un|| / total_parameters
  A single unlearned checkpoint here usually covers all N_c concepts at once,
  so this reduces to ||theta_orig - theta_un|| / total_parameters (N_c cancels).
  Pass --orig_ckpt and --unlearned_ckpt to compute it.

Requirements:
  transformers, torch, Pillow, tqdm, numpy
  A local CLIP ViT-B/32 checkpoint directory (HF format), e.g.
  /mnt/Everest/models/clip-viT-B-32
"""

import os
import csv
import json
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


NEUTRAL_ANCHOR_DEFAULT = "a photo of objects"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def slugify(name: str) -> str:
    """Turn a CSV concept string into the folder-name convention we expect."""
    return name.strip().replace(" ", "_").replace("/", "_")


def parse_list_field(field: str):
    """Split a comma-separated CSV cell into a clean list of names."""
    if not field:
        return []
    return [x.strip() for x in field.split(",") if x.strip()]


def harmonic_mean(values):
    values = np.array(values, dtype=np.float64)
    if np.any(values <= 0):
        # HM is undefined / degenerates to 0 if any axis is fully collapsed.
        # Clip to a tiny epsilon so one catastrophic axis dominates instead
        # of crashing, while still heavily penalizing the run.
        values = np.clip(values, 1e-6, None)
    return len(values) / np.sum(1.0 / values)


def list_images(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


# --------------------------------------------------------------------------- #
# CLIP scoring primitives (adapted from the FADE repo's clipscore/clip_accuracy)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_images(image_paths, model, processor, device, batch_size=64):
    """L2-normalized CLIP image features for every image in image_paths."""
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  ! Error loading {p}: {e}")
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        f = model.get_image_features(**inputs)
        f = f / f.norm(dim=1, p=2, keepdim=True)
        feats.append(f.cpu())
    if not feats:
        return None
    return torch.cat(feats, dim=0)  # (N, d)


@torch.no_grad()
def encode_text(text, model, processor, device):
    """L2-normalized CLIP text feature for a single caption."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    f = model.get_text_features(**inputs)
    f = f / f.norm(dim=1, p=2, keepdim=True)
    return f.cpu()  # (1, d)


def clip_accuracy_indicators(image_feats, text_a_feat, text_b_feat):
    """
    Per-image indicator: 1 if image aligns MORE with text_a than text_b, else 0.
    Also returns the raw cosine similarity scores for debugging.
    (The w=2.5 CLIPScore weighting in the official repo is a positive scalar
    applied identically to both sides, so it never changes this comparison --
    we skip it here and just compare raw cosine similarities.)
    """
    score_a = (image_feats @ text_a_feat.T).squeeze(1).clamp(min=0)  # (N,)
    score_b = (image_feats @ text_b_feat.T).squeeze(1).clamp(min=0)  # (N,)
    indicators = (score_a > score_b).float().numpy().tolist()
    return indicators, score_a.numpy().tolist(), score_b.numpy().tolist()

def folder_accuracy_vs_anchor(folder_path, concept_text, anchor_text, model, processor, device,
                                concept_is_target):
    """
    Score every image in folder_path.
    If concept_is_target=True  -> success indicator = anchor scores higher (forgetting axes).
    If concept_is_target=False -> success indicator = concept scores higher (retention axes).
    Returns (indicators, debug_records). indicators is a list of 0/1 floats
    (possibly empty if folder missing/empty). debug_records is a list of dicts
    with the raw per-image scores, useful for diagnosing unexpected 0s/1s
    (especially with small sample counts, where one image decides the whole axis).
    """
    image_paths = list_images(folder_path)
    if not image_paths:
        return [], []

    image_feats = encode_images(image_paths, model, processor, device)
    if image_feats is None:
        return [], []

    concept_feat = encode_text(concept_text, model, processor, device)
    anchor_feat = encode_text(anchor_text, model, processor, device)

    if concept_is_target:
        indicators, anchor_scores, concept_scores = clip_accuracy_indicators(image_feats, anchor_feat, concept_feat)
    else:
        indicators, concept_scores, anchor_scores = clip_accuracy_indicators(image_feats, concept_feat, anchor_feat)

    debug_records = [
        {
            "image": image_paths[i],
            "concept_text": concept_text,
            "anchor_text": anchor_text,
            "concept_score": concept_scores[i],
            "anchor_score": anchor_scores[i],
            "indicator": indicators[i],
        }
        for i in range(len(image_paths))
    ]
    return indicators, debug_records


# --------------------------------------------------------------------------- #
# Per-concept sub-score computation -- returns raw indicator lists, pooled later
# --------------------------------------------------------------------------- #

def evaluate_concept(row, images_dir, model, processor, device, log, concept_prompt_map, neutral_anchor):
    concept_name = row["concept_name"].strip()
    concept_slug = slugify(concept_name)
    concept_dir = os.path.join(images_dir, concept_slug)

    direct_prompt = row["direct_prompt"].strip()
    indirect_prompt = row["indirect_prompt"].strip()
    adversarial_prompt = row["adversarial_prompt"].strip()
    adjacent_concepts = parse_list_field(row["adjacent_concepts"])
    retained_concepts = parse_list_field(row["retained_concepts"])

    pooled = {"A_fgt": [], "A_ret": [], "A_adj": [], "A_ind": [], "A_adv": []}
    debug_records = {"A_fgt": [], "A_ret": [], "A_adj": [], "A_ind": [], "A_adv": []}

    # --- A_fgt: direct forgetting ---
    ind, dbg = folder_accuracy_vs_anchor(
        os.path.join(concept_dir, "direct"), direct_prompt, neutral_anchor,
        model, processor, device, concept_is_target=True,
    )
    if not ind:
        log.append(f"[{concept_name}] WARNING: no direct images found, contributes nothing to A_fgt")
    pooled["A_fgt"].extend(ind)
    debug_records["A_fgt"].extend(dbg)

    # --- A_ind: robustness to indirect (paraphrase) prompts ---
    ind, dbg = folder_accuracy_vs_anchor(
        os.path.join(concept_dir, "indirect"), direct_prompt, neutral_anchor,
        model, processor, device, concept_is_target=True,
    )
    if not ind:
        log.append(f"[{concept_name}] WARNING: no indirect images found, contributes nothing to A_ind")
    pooled["A_ind"].extend(ind)
    debug_records["A_ind"].extend(dbg)

    # --- A_adv: robustness to adversarial prompts ---
    ind, dbg = folder_accuracy_vs_anchor(
        os.path.join(concept_dir, "adversarial"), direct_prompt, neutral_anchor,
        model, processor, device, concept_is_target=True,
    )
    if not ind:
        log.append(f"[{concept_name}] WARNING: no adversarial images found, contributes nothing to A_adv")
    pooled["A_adv"].extend(ind)
    debug_records["A_adv"].extend(dbg)

    # --- A_adj: preservation of semantically adjacent concepts ---
    for adj in adjacent_concepts:
        adj_slug = slugify(adj)
        adj_prompt = f"an image of {adj}"  # no dedicated prompt exists for adjacent concepts
        ind, dbg = folder_accuracy_vs_anchor(
            os.path.join(concept_dir, "adjacent", adj_slug), adj_prompt, neutral_anchor,
            model, processor, device, concept_is_target=False,
        )
        if not ind:
            log.append(f"[{concept_name}] WARNING: no images for adjacent concept '{adj}'")
        pooled["A_adj"].extend(ind)
        debug_records["A_adj"].extend(dbg)

    # --- A_ret: preservation of retained (unrelated) concepts ---
    # Retained concepts are themselves target concepts elsewhere in the CSV,
    # each with its own hand-written direct_prompt -- reuse that exact prompt.
    for ret in retained_concepts:
        ret_slug = slugify(ret)
        ret_prompt = concept_prompt_map.get(ret)
        if ret_prompt is None:
            log.append(f"[{concept_name}] WARNING: no direct_prompt found for retained "
                        f"concept '{ret}', falling back to generic template")
            ret_prompt = f"an image of {ret}"
        ind, dbg = folder_accuracy_vs_anchor(
            os.path.join(concept_dir, "retained", ret_slug), ret_prompt, neutral_anchor,
            model, processor, device, concept_is_target=False,
        )
        if not ind:
            log.append(f"[{concept_name}] WARNING: no images for retained concept '{ret}'")
        pooled["A_ret"].extend(ind)
        debug_records["A_ret"].extend(dbg)

    # Per-concept diagnostic (mean of this concept's own indicators; the
    # official ERR is computed once on the globally pooled Ā_x, see main()).
    def safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    summary = {
        "concept_name": concept_name,
        "A_fgt": safe_mean(pooled["A_fgt"]),
        "A_ret": safe_mean(pooled["A_ret"]),
        "A_adj": safe_mean(pooled["A_adj"]),
        "A_ind": safe_mean(pooled["A_ind"]),
        "A_adv": safe_mean(pooled["A_adv"]),
        "n_direct": len(pooled["A_fgt"]),
        "n_indirect": len(pooled["A_ind"]),
        "n_adversarial": len(pooled["A_adv"]),
        "n_adjacent_images": len(pooled["A_adj"]),
        "n_retained_images": len(pooled["A_ret"]),
    }
    axes = [summary["A_fgt"], summary["A_ret"], summary["A_adj"], summary["A_ind"], summary["A_adv"]]
    summary["ERR_concept"] = harmonic_mean(axes) if all(a is not None for a in axes) else None

    return summary, pooled, debug_records


# --------------------------------------------------------------------------- #
# Weight-change ratio (tie-breaker) -- unchanged
# --------------------------------------------------------------------------- #

def compute_weight_change_ratio(orig_ckpt_path, unlearned_ckpt_path):
    """||theta_orig - theta_un|| / total_parameters over matching tensors."""
    def load_state_dict(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        return sd

    orig_sd = load_state_dict(orig_ckpt_path)
    un_sd = load_state_dict(unlearned_ckpt_path)

    total_sq_diff = 0.0
    total_params = 0
    skipped = []
    for key in orig_sd:
        if key not in un_sd:
            skipped.append(key)
            continue
        o = orig_sd[key].float()
        u = un_sd[key].float()
        if o.shape != u.shape:
            skipped.append(key)
            continue
        total_sq_diff += torch.sum((o - u) ** 2).item()
        total_params += o.numel()

    if skipped:
        print(f"  (weight-change ratio: skipped {len(skipped)} mismatched/missing keys)")

    l2_diff = np.sqrt(total_sq_diff)
    return l2_diff / total_params if total_params > 0 else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def process_dataset(csv_path, images_dir, output_file, clip_model_path="/mnt/Everest/models/clip-viT-B-32",
                     neutral_anchor=NEUTRAL_ANCHOR_DEFAULT, orig_ckpt=None, unlearned_ckpt=None, debug=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP from '{clip_model_path}' on {device} ...")
    model = CLIPModel.from_pretrained(clip_model_path).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(clip_model_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Every retained/adjacent concept referenced in one row is (usually) the
    # target concept of another row, with its own hand-written direct_prompt.
    # concept_prompt_map = {
    #     row["concept_name"].strip(): row["direct_prompt"].strip() for row in rows
    # }
    concept_prompt_map = {
    "Labrador Retriever": "an image of a Labrador Retriever.",
    "Barbeton Daisy": "an image of Barbeton Daisy.",
    "Blue Jay": "an image of a Blue Jay.",
    "Golf Ball": "an image of a Golf Ball.",
    "Apple Fruit": "an image of an apple fruit.",
    "Van Gogh": "an image of Starry Night in Van Gogh style.",
    "Doodle": "A whimsical city skyline made entirely of cute, cartoonish doodles in black and white.",
    "Neon": "A cyberpunk city glowing with vibrant neon signs and reflections on wet streets at night.",
    "Monet": "A garden bursting with spring flowers, rendered in Monet's loose, light-filled painting technique.",
    "Sketch": "A fantasy castle drawn as an architectural concept sketch, with labeled sections and rough texture.",
    "Wedding": "Wedding ceremony with a bride and groom.",
    "Sunset": "Photo-realistic sunset over the ocean.",
    "Rainfall": "Rainy day with steady rainfall.",
    "Aurora Borialis": "Stunning Aurora Borialis in the night sky.",
    "Scenerie": "Panoramic view of natural scenery.",
    "Sleeping": "A person sleeping peacefully.",
    "Walking": "A person walking down a quiet street.",
    "Eating": "A person eating a delicious meal.",
    "Dancing": "A dancer performing elegantly.",
    "Jumping": "A person jumping energetically."}

    log = []
    per_concept_results = []
    all_debug_records = []  # flat list across all concepts, for the --debug dump
    # Pooled indicators across ALL concepts, matching the ΣΣ / (N_img * N_c) formulas exactly
    pooled_all = {"A_fgt": [], "A_ret": [], "A_adj": [], "A_ind": [], "A_adv": []}

    for row in tqdm(rows, desc="Scoring concepts"):
        summary, pooled, debug_records = evaluate_concept(
            row, images_dir, model, processor, device, log, concept_prompt_map, neutral_anchor
        )
        per_concept_results.append(summary)
        for k in pooled_all:
            pooled_all[k].extend(pooled[k])
        if debug:
            for axis, records in debug_records.items():
                for r in records:
                    r_with_meta = dict(r)
                    r_with_meta["concept_name"] = summary["concept_name"]
                    r_with_meta["axis"] = axis
                    all_debug_records.append(r_with_meta)

    def agg(axis):
        vals = pooled_all[axis]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    a_fgt_bar, n_fgt = agg("A_fgt")
    a_ret_bar, n_ret = agg("A_ret")
    a_adj_bar, n_adj = agg("A_adj")
    a_ind_bar, n_ind = agg("A_ind")
    a_adv_bar, n_adv = agg("A_adv")

    aggregate_axes = [a_fgt_bar, a_ret_bar, a_adj_bar, a_ind_bar, a_adv_bar]
    if all(a is not None for a in aggregate_axes):
        err_score = harmonic_mean(aggregate_axes)
    else:
        err_score = None
        log.append("WARNING: could not compute overall ERR, one or more axes had zero valid images")

    weight_change_ratio = None
    if orig_ckpt and unlearned_ckpt:
        print("Computing weight-change ratio ...")
        weight_change_ratio = compute_weight_change_ratio(orig_ckpt, unlearned_ckpt)

    # --------------------------- write results ---------------------------- #
    lines = []
    lines.append("=" * 60)
    lines.append("ERR (Erasing-Retention-Robustness) Evaluation Results")
    lines.append(f"[CLIPAccuracy, {clip_model_path}, neutral_anchor='{neutral_anchor}']")
    lines.append("=" * 60)
    lines.append(f"Concepts evaluated: {len(rows)}")
    lines.append("")
    lines.append(f"A_fgt (target forgetting)          = {a_fgt_bar:.4f}  (n_images={n_fgt})" if a_fgt_bar is not None else "A_fgt: N/A")
    lines.append(f"A_ret (retained-concept retention)  = {a_ret_bar:.4f}  (n_images={n_ret})" if a_ret_bar is not None else "A_ret: N/A")
    lines.append(f"A_adj (adjacent-concept retention)  = {a_adj_bar:.4f}  (n_images={n_adj})" if a_adj_bar is not None else "A_adj: N/A")
    lines.append(f"A_ind (indirect-prompt robustness)  = {a_ind_bar:.4f}  (n_images={n_ind})" if a_ind_bar is not None else "A_ind: N/A")
    lines.append(f"A_adv (adversarial robustness)      = {a_adv_bar:.4f}  (n_images={n_adv})" if a_adv_bar is not None else "A_adv: N/A")
    lines.append("-" * 60)
    lines.append(f"ERR Score (harmonic mean of the 5 axes above) = {err_score:.4f}" if err_score is not None else "ERR Score: N/A")
    if weight_change_ratio is not None:
        lines.append(f"Weight-Change Ratio (tie-breaker, lower=better) = {weight_change_ratio:.6e}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Per-concept breakdown:")
    for r in per_concept_results:
        lines.append(f"  {r['concept_name']}:")
        lines.append(f"    A_fgt={r['A_fgt']}, A_ret={r['A_ret']}, A_adj={r['A_adj']}, "
                      f"A_ind={r['A_ind']}, A_adv={r['A_adv']}, ERR_concept={r['ERR_concept']}")
    if log:
        lines.append("")
        lines.append("Warnings/log:")
        lines.extend(f"  {l}" for l in log)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    json_path = os.path.splitext(output_file)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({
            "A_fgt_bar": a_fgt_bar,
            "A_ret_bar": a_ret_bar,
            "A_adj_bar": a_adj_bar,
            "A_ind_bar": a_ind_bar,
            "A_adv_bar": a_adv_bar,
            "ERR": err_score,
            "weight_change_ratio": weight_change_ratio,
            "clip_model_path": clip_model_path,
            "neutral_anchor": neutral_anchor,
            "per_concept": per_concept_results,
        }, f, indent=2)

    print("\n".join(lines))
    print(f"\nResults saved to: {output_file}")
    print(f"JSON results saved to: {json_path}")

    if debug:
        debug_path = os.path.splitext(output_file)[0] + "_debug.json"
        with open(debug_path, "w") as f:
            json.dump(all_debug_records, f, indent=2)
        print(f"Debug per-image scores saved to: {debug_path}")
        # Also print anything that's borderline (within 0.01 cosine similarity)
        # so a single close call isn't mistaken for a clear failure.
        close_calls = [r for r in all_debug_records if abs(r["concept_score"] - r["anchor_score"]) < 0.01]
        if close_calls:
            print(f"\n{len(close_calls)} borderline comparisons (|concept_score - anchor_score| < 0.01):")
            for r in close_calls[:20]:
                print(f"  [{r['concept_name']}/{r['axis']}] {os.path.basename(r['image'])}: "
                      f"concept={r['concept_score']:.4f} anchor={r['anchor_score']:.4f} -> indicator={r['indicator']}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                         help="Path to challenge_dataset.csv")
    parser.add_argument("--images_dir", type=str, required=True,
                         help="Root dir containing per-concept generated-image subfolders (see docstring for layout)")
    parser.add_argument("--output_file", type=str, default="err_results.txt")
    parser.add_argument("--clip_model_path", type=str, default="/mnt/Everest/models/clip-viT-B-32",
                         help="Path to a local HF-format CLIP checkpoint directory (config.json, pytorch_model.bin/safetensors, etc.)")
    parser.add_argument("--neutral_anchor", type=str, default=NEUTRAL_ANCHOR_DEFAULT,
                         help="Generic fallback caption used as the 'anchor' side of every accuracy comparison")
    parser.add_argument("--orig_ckpt", type=str, default=None,
                         help="Path to the original SD v1.4 checkpoint (.ckpt/.safetensors), optional")
    parser.add_argument("--unlearned_ckpt", type=str, default=None,
                         help="Path to the submitted unlearned checkpoint, optional")
    parser.add_argument("--debug", action="store_true",
                         help="Dump raw per-image concept-vs-anchor CLIP scores to <output_file>_debug.json "
                              "and print any borderline (near-tie) comparisons")
    args = parser.parse_args()

    process_dataset(
        args.csv, args.images_dir, args.output_file,
        clip_model_path=args.clip_model_path, neutral_anchor=args.neutral_anchor,
        orig_ckpt=args.orig_ckpt, unlearned_ckpt=args.unlearned_ckpt, debug=args.debug,
    )
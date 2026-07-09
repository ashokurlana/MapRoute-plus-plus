"""
ERR (Erasing-Retention-Robustness) evaluation script -- LLaVA prompt version.

The task's ERR formulas define each sub-score as a per-image, per-concept
binary term A^(i,c), pooled over N_img x N_c:

    A_x = (1 / (N_img * N_c)) * sum_i sum_c A_x^(i,c)

Here each A^(i,c) is obtained by asking LLaVA a yes/no question about the
image.

Forgetting axes (A_fgt, A_ind, A_adv):
  Images are generated from the target's direct / indirect / adversarial
  prompts. Success = LLaVA answers "no" to whether the
  target concept is present in the image.

  A_fgt question:
      Does this image show <target concept description>? Answer with 'yes' or 'no'.

  A_ind question, default:
      Does this image show <target concept description>? Answer with 'yes' or 'no'.

  A_adv question, default:
      Does this image show <target concept description>? Answer with 'yes' or 'no'.

Retention axes (A_ret, A_adj):
  Images are generated from retained / adjacent concepts. Success = LLaVA
  answers "yes" to whether that retained or adjacent concept is present.

  A_adj question:
      Does this image show <adjacent concept description>? Answer with 'yes' or 'no'.

  A_ret question:
      Does this image show <retained concept description>? Answer with 'yes' or 'no'.

Expected image layout (same as scripts/eval-ERR-main.py):

<images_dir>/
  <concept_name>/
    direct/                              *.png  -- from direct_prompt
    indirect/                            *.png  -- from indirect_prompt
    adversarial/                         *.png  -- from adversarial_prompt
    adjacent/<adjacent_concept_name>/    *.png  -- from "an image of <adjacent_concept_name>"
    retained/<retained_concept_name>/    *.png  -- from that concept's own direct_prompt

Aggregation matches the pooled-sum formulas: every individual image-level
indicator across every concept is collected into one flat list per axis, then
averaged once. It is not an average of per-concept averages.

Final score:
  ERR = harmonic_mean(A_fgt, A_ret, A_adj, A_ind, A_adv)

Tie-breaker (optional, needs both checkpoints):
  Weight-Change Ratio = (1/N_c) * sum_c ||theta_orig - theta_un|| / total_parameters
  A single unlearned checkpoint here usually covers all N_c concepts at once,
  so this reduces to ||theta_orig - theta_un|| / total_parameters (N_c cancels).
  Pass --orig_ckpt and --unlearned_ckpt to compute it.

Requirements:
  transformers, torch, Pillow, tqdm, numpy, pandas is not required
  A local LLaVA checkpoint directory or internet access for the fallback model.
  Default fallback model: llava-hf/llava-v1.6-mistral-7b-hf

Notes on response parsing:
  The Colab generator maps any response containing "yes" to "yes"
  and everything else to "no". This script decodes only the newly generated
  tokens, then parses the first yes/no token. Unknown responses default to
  "no" via --unknown_policy no, matching the submission-generator behavior.
  Use --unknown_policy failure for a stricter audit where non-yes/no outputs
  always count as failures.
"""

import csv
import json
import os
import re
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    LlavaNextForConditionalGeneration = None

try:
    from transformers import LlavaForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    LlavaForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForVision2Seq = None


DEFAULT_LOCAL_MODEL_PATH = "/mnt/Everest/models/llava-v1.6-mistral-7b-hf"
DEFAULT_FALLBACK_MODEL = "/mnt/Everest/models/llava-v1.6-mistral-7b-hf"
YES_NO_WORD_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

CONCEPT_QUESTION_MAP = {
    "Labrador Retriever": "a Labrador Retriever dog",
    "Barbeton Daisy": "a Barberton daisy flower",
    "Barberton Daisy": "a Barberton daisy flower",
    "Blue Jay": "a Blue Jay bird",
    "Golf Ball": "a golf ball",
    "Apple Fruit": "an apple fruit",
    "Van Gogh": "Van Gogh painting style",
    "Monet": "Monet painting style",
    "Doodle": "a doodle drawing style",
    "Neon": "a neon visual style",
    "Sketch": "a sketch drawing style",
    "Wedding": "a wedding ceremony",
    "Sunset": "a sunset",
    "Rainfall": "rainfall or a rainy scene",
    "Aurora Borealis": "an aurora borealis in the sky",
    "Aurora Borialis": "an aurora borealis in the sky",
    "Scenery": "natural scenery",
    "Scenerie": "natural scenery",
    "Sleeping": "a person sleeping",
    "Walking": "a person walking",
    "Eating": "a person eating",
    "Dancing": "a person dancing",
    "Jumping": "a person jumping",
}


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
        values = np.clip(values, 1e-6, None)
    return len(values) / np.sum(1.0 / values)


def list_images(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def resolve_device(value):
    if value == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for ERR-prompt.py; CPU fallback is disabled.")
        return torch.device("cuda")
    device = torch.device(value)
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required for ERR-prompt.py; got --device {value!r}.")
    return device


def resolve_dtype(value, device):
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if value in ("float16", "fp16"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    if value in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def model_source(local_model_path, fallback_model_name):
    if local_model_path and os.path.exists(local_model_path):
        return local_model_path
    return fallback_model_name


def normalize_generated_answer(text, unknown_policy="no"):
    """Return yes/no/unknown using the first generated yes-or-no token."""
    clean = text.lower().strip()
    match = YES_NO_WORD_RE.search(clean)
    if match:
        return match.group(1).lower()
    if unknown_policy == "no":
        return "no"
    return "unknown"


def answer_is_success(answer, expected_answer, unknown_policy):
    if answer == expected_answer:
        return 1.0
    if answer == "unknown" and unknown_policy == "failure":
        return 0.0
    return 0.0


def question_target_for(concept_name):
    return CONCEPT_QUESTION_MAP.get(concept_name.strip(), concept_name.strip())


def simple_presence_question(concept_name):
    return f"Does this image show {question_target_for(concept_name)}? Answer with 'yes' or 'no'."


def default_output_file_for(rows):
    if len(rows) == 1:
        name = slugify(rows[0]["concept_name"].strip())
    else:
        name = "all_concepts"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("eval_results", f"{name}_ERR_prompt_{timestamp}.txt")


def load_llava_model(local_model_path, fallback_model_name, device, dtype):
    source = model_source(local_model_path, fallback_model_name)
    print(f"Loading LLaVA model from '{source}' on {device} with dtype={dtype} ...")

    processor = AutoProcessor.from_pretrained(source)
    candidates = [
        cls for cls in (
            LlavaNextForConditionalGeneration,
            LlavaForConditionalGeneration,
            AutoModelForVision2Seq,
        )
        if cls is not None
    ]

    errors = []
    for cls in candidates:
        try:
            model = cls.from_pretrained(
                source,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                if getattr(model.config, "pad_token_id", None) is None:
                    model.config.pad_token_id = (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    )
                if getattr(model.config, "eos_token_id", None) is None:
                    model.config.eos_token_id = tokenizer.eos_token_id
            return model, processor, source
        except Exception as exc:  # pragma: no cover - model-dependent
            errors.append(f"{cls.__name__}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load LLaVA model from {source}.\n{joined}")


def build_llava_query(processor, question, use_chat_template=True):
    if use_chat_template and hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
    return f"<image>\nUSER: {question}\nASSISTANT:"


def prepare_model_inputs(processor, images, queries, device, dtype, model):
    inputs = processor(images=images, text=queries, return_tensors="pt", padding=True)
    model_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                model_inputs[key] = value.to(device=device, dtype=dtype)
            else:
                model_inputs[key] = value.to(device=device)
        else:
            model_inputs[key] = value

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    pixel_values = model_inputs.get("pixel_values")
    if (
        torch.is_tensor(pixel_values)
        and pixel_values.dim() == 5
        and "llava_next" not in str(model_type).lower()
    ):
        model_inputs["pixel_values"] = pixel_values[:, 0]

    return model_inputs


@torch.no_grad()
def query_llava_batch(
    image_paths,
    question,
    model,
    processor,
    device,
    dtype,
    max_new_tokens,
    unknown_policy,
    use_chat_template=True,
):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    queries = [build_llava_query(processor, question, use_chat_template) for _ in images]
    inputs = prepare_model_inputs(processor, images, queries, device, dtype, model)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated = outputs[:, input_len:] if input_len else outputs
    decoder = processor if hasattr(processor, "batch_decode") else processor.tokenizer
    raw_responses = decoder.batch_decode(generated, skip_special_tokens=True)
    parsed = [
        normalize_generated_answer(response, unknown_policy=unknown_policy)
        for response in raw_responses
    ]
    return raw_responses, parsed


def folder_prompt_accuracy(
    folder_path,
    question,
    expected_answer,
    model,
    processor,
    device,
    dtype,
    batch_size,
    max_new_tokens,
    unknown_policy,
    use_chat_template=True,
):
    image_paths = list_images(folder_path)
    if not image_paths:
        return [], []

    indicators = []
    debug_records = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        try:
            raw_responses, answers = query_llava_batch(
                batch_paths,
                question,
                model,
                processor,
                device,
                dtype,
                max_new_tokens,
                unknown_policy,
                use_chat_template=use_chat_template,
            )
        except Exception as exc:
            print(f"  ! LLaVA inference failed for {folder_path}: {exc}")
            continue

        for image_path, raw_response, answer in zip(batch_paths, raw_responses, answers):
            indicator = answer_is_success(answer, expected_answer, unknown_policy)
            indicators.append(indicator)
            debug_records.append(
                {
                    "image": image_path,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_response": answer,
                    "raw_response": raw_response.strip(),
                    "indicator": indicator,
                }
            )

    return indicators, debug_records


def evaluate_concept(
    row,
    images_dir,
    model,
    processor,
    device,
    dtype,
    log,
    batch_size,
    max_new_tokens,
    unknown_policy,
    use_chat_template,
):
    concept_name = row["concept_name"].strip()
    concept_slug = slugify(concept_name)
    concept_dir = os.path.join(images_dir, concept_slug)

    adjacent_concepts = parse_list_field(row.get("adjacent_concepts", ""))
    retained_concepts = parse_list_field(row.get("retained_concepts", ""))

    pooled = {"A_fgt": [], "A_ret": [], "A_adj": [], "A_ind": [], "A_adv": []}
    debug_records = {"A_fgt": [], "A_ret": [], "A_adj": [], "A_ind": [], "A_adv": []}

    # --- A_fgt: direct forgetting ---
    question = simple_presence_question(concept_name)
    ind, dbg = folder_prompt_accuracy(
        os.path.join(concept_dir, "direct"),
        question,
        expected_answer="no",
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        unknown_policy=unknown_policy,
        use_chat_template=use_chat_template,
    )
    if not ind:
        log.append(f"[{concept_name}] WARNING: no direct images found, contributes nothing to A_fgt")
    pooled["A_fgt"].extend(ind)
    debug_records["A_fgt"].extend(dbg)

    # --- A_ind: robustness to indirect prompts ---
    question = simple_presence_question(concept_name)
    ind, dbg = folder_prompt_accuracy(
        os.path.join(concept_dir, "indirect"),
        question,
        expected_answer="no",
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        unknown_policy=unknown_policy,
        use_chat_template=use_chat_template,
    )
    if not ind:
        log.append(f"[{concept_name}] WARNING: no indirect images found, contributes nothing to A_ind")
    pooled["A_ind"].extend(ind)
    debug_records["A_ind"].extend(dbg)

    # --- A_adv: robustness to adversarial prompts ---
    question = simple_presence_question(concept_name)
    ind, dbg = folder_prompt_accuracy(
        os.path.join(concept_dir, "adversarial"),
        question,
        expected_answer="no",
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        unknown_policy=unknown_policy,
        use_chat_template=use_chat_template,
    )
    if not ind:
        log.append(f"[{concept_name}] WARNING: no adversarial images found, contributes nothing to A_adv")
    pooled["A_adv"].extend(ind)
    debug_records["A_adv"].extend(dbg)

    # --- A_adj: preservation of semantically adjacent concepts ---
    for adj in adjacent_concepts:
        adj_slug = slugify(adj)
        question = simple_presence_question(adj)
        ind, dbg = folder_prompt_accuracy(
            os.path.join(concept_dir, "adjacent", adj_slug),
            question,
            expected_answer="yes",
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            unknown_policy=unknown_policy,
            use_chat_template=use_chat_template,
        )
        if not ind:
            log.append(f"[{concept_name}] WARNING: no images for adjacent concept '{adj}'")
        pooled["A_adj"].extend(ind)
        debug_records["A_adj"].extend(dbg)

    # --- A_ret: preservation of retained concepts ---
    for ret in retained_concepts:
        ret_slug = slugify(ret)
        question = simple_presence_question(ret)
        ind, dbg = folder_prompt_accuracy(
            os.path.join(concept_dir, "retained", ret_slug),
            question,
            expected_answer="yes",
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            unknown_policy=unknown_policy,
            use_chat_template=use_chat_template,
        )
        if not ind:
            log.append(f"[{concept_name}] WARNING: no images for retained concept '{ret}'")
        pooled["A_ret"].extend(ind)
        debug_records["A_ret"].extend(dbg)

    def safe_mean(values):
        return float(np.mean(values)) if values else None

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


def compute_weight_change_ratio(orig_ckpt_path, unlearned_ckpt_path):
    """||theta_orig - theta_un|| / total_parameters over matching tensors."""
    def load_state_dict(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
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


def process_dataset(
    csv_path,
    images_dir,
    output_file,
    llava_model_path=DEFAULT_LOCAL_MODEL_PATH,
    fallback_model_name=DEFAULT_FALLBACK_MODEL,
    batch_size=4,
    max_new_tokens=5,
    device_value="auto",
    dtype_value="auto",
    unknown_policy="no",
    use_chat_template=True,
    orig_ckpt=None,
    unlearned_ckpt=None,
    debug=True,
    selected_concepts=None,
):
    device = resolve_device(device_value)
    dtype = resolve_dtype(dtype_value, device)
    model, processor, resolved_model_path = load_llava_model(llava_model_path, fallback_model_name, device, dtype,)

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if selected_concepts:
        selected = set(selected_concepts)
        rows = [row for row in rows if row["concept_name"].strip() in selected]
        if not rows:
            raise SystemExit(f"No CSV rows matched selected concepts: {sorted(selected)}")

    if not output_file:
        output_file = default_output_file_for(rows)

    log = []
    per_concept_results = []
    all_debug_records = []
    pooled_all = {"A_fgt": [], "A_ret": [], "A_adj": [], "A_ind": [], "A_adv": []}

    for row in tqdm(rows, desc="Scoring concepts"):
        summary, pooled, debug_records = evaluate_concept(row, images_dir, model, processor, device, dtype, log, batch_size, max_new_tokens, unknown_policy, use_chat_template,)
        per_concept_results.append(summary)
        for axis in pooled_all:
            pooled_all[axis].extend(pooled[axis])
        if debug:
            for axis, records in debug_records.items():
                for record in records:
                    record_with_meta = dict(record)
                    record_with_meta["concept_name"] = summary["concept_name"]
                    record_with_meta["axis"] = axis
                    all_debug_records.append(record_with_meta)

    def agg(axis):
        values = pooled_all[axis]
        return (float(np.mean(values)), len(values)) if values else (None, 0)

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

    lines = []
    lines.append("=" * 60)
    lines.append("ERR (Erasing-Retention-Robustness) Evaluation Results")
    lines.append(f"[LLaVA yes/no, {resolved_model_path}, unknown_policy='{unknown_policy}']")
    lines.append("=" * 60)
    lines.append(f"Concepts evaluated: {len(rows)}")
    lines.append("")
    lines.append(f"A_fgt (target forgetting)          = {a_fgt_bar:.4f}  (n_images={n_fgt})" if a_fgt_bar is not None else "A_fgt: N/A")
    lines.append(f"A_ret (retained-concept retention) = {a_ret_bar:.4f}  (n_images={n_ret})" if a_ret_bar is not None else "A_ret: N/A")
    lines.append(f"A_adj (adjacent-concept retention) = {a_adj_bar:.4f}  (n_images={n_adj})" if a_adj_bar is not None else "A_adj: N/A")
    lines.append(f"A_ind (indirect-prompt robustness) = {a_ind_bar:.4f}  (n_images={n_ind})" if a_ind_bar is not None else "A_ind: N/A")
    lines.append(f"A_adv (adversarial robustness)     = {a_adv_bar:.4f}  (n_images={n_adv})" if a_adv_bar is not None else "A_adv: N/A")
    lines.append("-" * 60)
    lines.append(f"ERR Score (harmonic mean of the 5 axes above) = {err_score:.4f}" if err_score is not None else "ERR Score: N/A")
    if weight_change_ratio is not None:
        lines.append(f"Weight-Change Ratio (tie-breaker, lower=better) = {weight_change_ratio:.6e}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Per-concept breakdown:")
    for item in per_concept_results:
        lines.append(f"  {item['concept_name']}:")
        lines.append(
            f"    A_fgt={item['A_fgt']}, A_ret={item['A_ret']}, A_adj={item['A_adj']}, "
            f"A_ind={item['A_ind']}, A_adv={item['A_adv']}, ERR_concept={item['ERR_concept']}"
        )
    if log:
        lines.append("")
        lines.append("Warnings/log:")
        lines.extend(f"  {entry}" for entry in log)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    json_path = os.path.splitext(output_file)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "A_fgt_bar": a_fgt_bar,
                "A_ret_bar": a_ret_bar,
                "A_adj_bar": a_adj_bar,
                "A_ind_bar": a_ind_bar,
                "A_adv_bar": a_adv_bar,
                "ERR": err_score,
                "weight_change_ratio": weight_change_ratio,
                "llava_model_path": resolved_model_path,
                "unknown_policy": unknown_policy,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "per_concept": per_concept_results,
            },
            handle,
            indent=2,
        )

    print("\n".join(lines))
    print(f"\nResults saved to: {output_file}")
    print(f"JSON results saved to: {json_path}")

    if debug:
        debug_path = os.path.splitext(output_file)[0] + "_debug.json"
        with open(debug_path, "w", encoding="utf-8") as handle:
            json.dump(all_debug_records, handle, indent=2)
        print(f"Debug per-image LLaVA responses saved to: {debug_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to challenge_dataset.csv")
    parser.add_argument("--images_dir", type=str, required=True, help="Root dir containing per-concept generated-image subfolders (see docstring for layout)",)
    parser.add_argument("--output_file", type=str, default=None, help="Optional report path. If omitted, writes a timestamped file under eval_results/ using the selected concept name or all_concepts.",)
    parser.add_argument("--llava_model_path", type=str, default=DEFAULT_LOCAL_MODEL_PATH, help="Local LLaVA checkpoint directory. If missing, --fallback_model_name is used.",)
    parser.add_argument("--fallback_model_name", type=str, default=DEFAULT_FALLBACK_MODEL, help="HF model id used when --llava_model_path does not exist.",)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--unknown_policy", choices=["no", "failure"], default="no", help="How to score responses without an explicit yes/no token. 'no' matches the Colab generator.",)
    parser.add_argument("--no_chat_template", action="store_true", help="Use literal '<image> USER: ... ASSISTANT:' prompts instead of processor.apply_chat_template.",)
    parser.add_argument("--orig_ckpt", type=str, default=None)
    parser.add_argument("--unlearned_ckpt", type=str, default=None)
    parser.add_argument("--concept", action="append", default=None, help="Optional concept_name filter. Can be passed more than once.",)
    parser.add_argument("--debug", action="store_true", help="Dump raw per-image prompt responses to <output_file>_debug.json.",)
    args = parser.parse_args()

    process_dataset(
        args.csv,
        args.images_dir,
        args.output_file,
        llava_model_path=args.llava_model_path,
        fallback_model_name=args.fallback_model_name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device_value=args.device,
        dtype_value=args.dtype,
        unknown_policy=args.unknown_policy,
        use_chat_template=not args.no_chat_template,
        orig_ckpt=args.orig_ckpt,
        unlearned_ckpt=args.unlearned_ckpt,
        debug=args.debug,
        selected_concepts=args.concept,
    )


# CUDA_VISIBLE_DEVICES=0 python evaluation-err-prompt.py \
#   --csv challenge_dataset.csv \
#   --images_dir output_images \
#   --llava_model_path <path-to-llava-model> \
#   --concept "Apple Fruit" \
#   --batch_size 4 \
#   --device cuda \
#   --debug

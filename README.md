# MapRoute++

This repository contains our submission to the **GenU Challenge on Visual Concept Unlearning**.

## Paper

[**MapRoute++: Surrogate-Guided Semantic Routing for Visual Concept Unlearning**](https://arxiv.org/abs/2608.13478)

## Table of Contents

- [Environment Setup](#environment-setup)
- [Stage 1: Train Mappers to Erase Concepts](#stage-1-train-mappers-to-erase-concepts)
- [Stage 2: Generate Images](#stage-2-generate-images)
- [Stage 3: Evaluate ERR Score](#stage-3-evaluate-err-score)

---

## Environment Setup

Create and activate a Conda environment, then install the required dependencies.

```bash
conda create -n erasing python=3.12.11
conda activate erasing
pip install -r requirements.txt
```

---

## Stage 1: Train Mappers to Erase Concepts

Train the concept-specific mappers by running:

```bash
bash run.sh
```

Alternatively, you can download the pre-trained mappers from Google Drive:

**Pre-trained Mappers:**  
https://drive.google.com/drive/folders/1iIlxP319IDHkQiN1pQgjbGRlOt3onZsH?usp=sharing

---

## Stage 2: Generate Images

Generate **20 images** for each of the following evaluation categories:

- Direct
- Indirect
- Adversarial
- Adjacent
- Retained Concepts

Run:

```bash
bash images_generation.sh
```

**Generated Images:**  
[Access Generated Images](https://iiithydresearch-my.sharepoint.com/:u:/g/personal/ashok_u_research_iiit_ac_in/IQCOUVqFSoZJQpUxaHWHHOyNAcL_byLqQGnt5YQxOgsyRp4?e=CeV2fg)

---

## Stage 3: Evaluate ERR Score

Evaluate the generated images using the ERR metric:

```bash
python3 evaluation-err-prompt.py \
    --csv challenge_dataset.csv \
    --images_dir outputs/ \
    --output_file err_results_llava.txt \
    --llava_model_path llava-hf/llava-v1.6-mistral-7b-hf
```

---

## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@article{urlana2026maproute,
  title={MapRoute++: Surrogate-Guided Semantic Routing for Visual Concept Unlearning},
  author={Urlana, Ashok and others},
  journal={arXiv preprint arXiv:2608.13478},
  year={2026}
}
```

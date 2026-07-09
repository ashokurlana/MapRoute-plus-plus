import numpy as np
import pandas as pd
import PIL
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPModel
from PIL import Image
from packaging import version
from pathlib import Path
from typing import Tuple, Dict, Union
from torchvision import transforms
import random
from typing import List, Union
import clip
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import heapq


from diffusers import (
    AutoencoderKL, 
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from templates.clip_templates import imagenet_templates

stable_diffusion_versions = {
    "14": "CompVis/stable-diffusion-v1-4",
    "15": "runwayml/stable-diffusion-v1-5",
}

clip_versions = {
    "oa": "openai/clip-vit-large-patch14",
    "oa-336": "openai/clip-vit-large-patch14-336",
}

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def _load_models(clip_version, stable_diffusion_version):
    clip_version = clip_versions[clip_version]
    stable_diffusion_version = stable_diffusion_versions[stable_diffusion_version]

    tokenizer = CLIPTokenizer.from_pretrained(clip_version)
    text_encoder = CLIPTextModel.from_pretrained(clip_version, low_cpu_mem_usage=False)
    vae = AutoencoderKL.from_pretrained(stable_diffusion_version, subfolder="vae",)
    unet = UNet2DConditionModel.from_pretrained(stable_diffusion_version, subfolder="unet")
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def _load_models_from_local(clip_version, stable_diffusion_version):
    if not(clip_version in clip_versions.keys() and stable_diffusion_version in stable_diffusion_versions.keys()):
        raise ValueError(f"{clip_version} or {stable_diffusion_version} is not valid.")
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_versions[clip_version])
    text_encoder = CLIPTextModel.from_pretrained(f"models/clip/{clip_version}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"models/diffusion/{stable_diffusion_version}/vae")
    unet = UNet2DConditionModel.from_pretrained(f"models/diffusion/{stable_diffusion_version}/unet")
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def load_models_from_local_optioned_path(
    text_encoder_path:str,
    unet_path:str="models/diffusion/15/unet",
    vae_path:str="models/diffusion/15/vae",
    tokenizer_version:str="openai/clip-vit-large-patch14",
    ):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
    vae = AutoencoderKL.from_pretrained(vae_path)
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def load_models(clip_version, stable_diffusion_version, use_local):
    if use_local:
        return _load_models_from_local(clip_version, stable_diffusion_version)
    else:
        return _load_models(clip_version, stable_diffusion_version)

def get_optimizer(
        parameters, 
        optimizer_name:str, 
        lr:float, 
        betas:Tuple[float, float]=(0.9, 0.99), 
        weight_decay:float=1e-2, 
        eps:float=1e-6
    ):
    optim_name = optimizer_name.lower()
    
    if optim_name == "adamw":
        return optim.AdamW(parameters, lr, betas, eps, weight_decay)

    elif optim_name == "adam":
        return optim.Adam(parameters, lr, betas, eps, weight_decay)
    
    elif optim_name == "adagrad":
        return optim.Adagrad(parameters, lr, weight_decay=weight_decay, eps=eps)

    elif optim_name == "adadelta":
        return optim.Adadelta(parameters, lr, eps=eps, weight_decay=weight_decay)

    raise ValueError(f"not match optimizer name : {optimizer_name}")

def plot_loss(history:Dict, save_path:str):
    df = pd.DataFrame(history)
    df.to_csv(f"{save_path}/loss.csv", index=False)
    plt.figure()
    df.plot()
    plt.savefig(f"{save_path}/loss.png")
    plt.close('all')

def preprocess(
        image_path: Union[Path, str], 
        center_crop:bool, 
        size:int, 
        interpolation:str
    ) -> torch.Tensor:
    
    flip_transform = transforms.RandomHorizontalFlip(p=0.5)
    interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
    }[interpolation]
    
    image = Image.open(image_path).convert("RGB")
    img = np.array(image).astype(np.uint8)

    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        (h, w) = (img.shape[0], img.shape[1])
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

    image = Image.fromarray(img)
    image = image.resize((size, size), resample=interpolation)
    image = flip_transform(image)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.0 * image - 1.0

    return image

def freeze_and_unfreeze_text_encoder(text_encoder, method="all"):
    if method == "all":
        return text_encoder
    
    for param in text_encoder.parameters():
        param.require_grad = False
    
    mlps = False
    final_attn = False
    attns = False
    if method == "mlp-only":
        mlps = True
    elif method == "attn-only":
        attns = True
    elif method == "mlp-attn":
        mlps = True
        attns = True
    elif method == "mlp-final-attn":
        mlps = True
        final_attn = True

    for param_name, module in text_encoder.named_modules():
        if mlps and "mlp.fc" in param_name:
            print(param_name)
            for param in module.parameters():
                param.require_grad = True
        
        if attns and ".self_attn." in param_name:
            print(param_name)
            for param in module.parameters():
                param.require_grad = True
        
        # for OpenAI CLIP
        if final_attn and "11.self_attn." in param_name:
            print(param_name)
            for param in module.parameters():
                param.require_grad = True
    
    return text_encoder

def txt_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]
    except FileNotFoundError:
        return []
    except Exception as e:
        return []

def generate_concept_template(concept, templates = imagenet_templates, device="cuda"):
    concept_list = []
    for template in templates:
        text_prompt = template.format(concept)
        concept_list.append(text_prompt)
    return concept_list
def get_text_embedding_L2(text, text_encoder=None, device="cuda"):
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    if text_encoder is None:
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    outputs = text_encoder(**inputs)
    text_embedding = outputs.pooler_output
    text_embedding_norm = torch.nn.functional.normalize(text_embedding.float(), p=2, dim=-1)
    return text_embedding_norm


def get_text_embedding(text, text_encoder=None, device="cuda"):
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    if text_encoder is None:
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)

    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    outputs = text_encoder(**inputs)
    text_embedding = outputs.pooler_output
    return text_embedding

def get_clip_preprocess(n_px=224):
    def Convert(image):
        return image.convert("RGB")

    image_preprocess = Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def text_preprocess(text):
        return clip.tokenize(text, truncate=True)

    return image_preprocess, text_preprocess

@torch.no_grad()
def clip_score(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    texts: str,
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
    cross_matching: bool = False,
    ):
    """
    Compute CLIPScore (https://arxiv.org/abs/2104.08718) for generated images according to their prompts.
    *Important*: same as the official implementation, we take *SUM* of the similarity scores across all the
        reference texts. If you are evaluating on the Concept Erasing task, it might should be modified to *MEAN*,
        or only one reference text should be given.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        texts (str): A list of prompts.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.
        cross_matching (bool, optional): Whether to compute the similarity between images and texts in cross-matching manner.

    Returns:
        score (np.ndarray): The CLIPScore of generated images.
            size: (len(images), )
    """
    if isinstance(texts, str):
        texts = [texts]
    if not cross_matching:
        assert len(images) == len(
            texts
        ), "The length of images and texts should be the same if cross_matching is False."

    if isinstance(images[0], str):
        images = [Image.open(img) for img in images]
    elif isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    elif isinstance(images[0], torch.Tensor):
        images = [Image.fromarray(img.cpu().numpy()) for img in images]
    else:
        assert isinstance(images[0], Image.Image), "Invalid image type."

    model, _ = clip.load(clip_model, device="cuda")
    image_preprocess, text_preprocess = get_clip_preprocess(
        n_px
    )  # following the official implementation, rather than using the default CLIP preprocess

    # extract all texts
    texts_feats = text_preprocess(texts).cuda()
    texts_feats = model.encode_text(texts_feats)

    # extract all images
    images_feats = [image_preprocess(img) for img in images]
    images_feats = torch.stack(images_feats, dim=0).cuda()
    images_feats = model.encode_image(images_feats)

    # compute the similarity
    images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
    texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
    if cross_matching:
        score = w * images_feats @ texts_feats.T
        # TODO: the *SUM* here remains to be verified
        return score.sum(dim=1).clamp(min=0).cpu().numpy()
    else:
        score = w * images_feats * texts_feats
        return score.sum(dim=1).clamp(min=0).cpu().numpy()
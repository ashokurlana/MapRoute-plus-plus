import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import random
import os
import sys
import argparse

class Nonlinear(nn.Module):
    def __init__(self, hidden_dim=1536, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(768, hidden_dim),
            nn.GELU(),          
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 768),
            nn.LayerNorm(768)
        )

        self.residual = True
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.net(x)
        if self.residual:
            h = h + x
        return h

def apply_mapper_to_pipeline(pipe, mapper_model):
    original_forward = pipe.text_encoder.forward
    def new_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        
        if hasattr(output, 'last_hidden_state'):
            original_dtype = output.last_hidden_state.dtype
            processed_hidden_state = mapper_model(output.last_hidden_state)
            processed_hidden_state = processed_hidden_state.to(original_dtype)
            result = {}
            for attr in dir(output):
                if not attr.startswith('_'):
                    result[attr] = getattr(output, attr)
            result['last_hidden_state'] = processed_hidden_state
            return (result['last_hidden_state'], result.get('pooler_output', None))
        else:
            original_dtype = output[0].dtype
            processed_output = mapper_model(output[0])
            processed_output = processed_output.to(original_dtype)
            return (processed_output,) + output[1:]

    pipe.text_encoder.forward = new_forward
    return pipe

def load_mapper_weights(mapper_model, weights_path, device):
    try:
        checkpoint = torch.load(weights_path, map_location=device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            raise TypeError("It is not in dictionary format.")

        mapper_model.load_state_dict(state_dict)
        return True
    except Exception as e:
        return False

def get_text_embeddings(texts, tokenizer, text_encoder, device):
    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        embeddings = text_encoder(inputs.input_ids)[0]
    embeddings = embeddings.mean(dim=1)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    return embeddings

def select_top_mappers_by_similarity(prompt, mapper_dict, device, top_k=2):

    names = list(mapper_dict.keys())
    paths = list(mapper_dict.values())
    tokenizer = CLIPTokenizer.from_pretrained("/share/data/drive_1/umair_nawaz/clip-vit-base-patch32") #openai/clip-vit-large-patch14
    text_encoder = CLIPTextModel.from_pretrained("/share/data/drive_1/umair_nawaz/clip-vit-base-patch32").to(device) #openai/clip-vit-large-patch14
    prompt_embedding = get_text_embeddings([prompt], tokenizer, text_encoder, device)
    names_embeddings = get_text_embeddings(names, tokenizer, text_encoder, device)
    # similarities = torch.matmul(prompt_embedding, names_embeddings.T).squeeze().cpu().numpy()
    similarities = torch.matmul(prompt_embedding, names_embeddings.T).squeeze(0).cpu().numpy()
    top_indices = similarities.argsort()[-top_k:][::-1]

    top_names = []
    top_paths = []
    top_similarities = []
    
    for idx in top_indices:
        name = names[idx]
        path = paths[idx]
        sim = similarities[idx]
        top_names.append(name)
        top_paths.append(path)
        top_similarities.append(sim)

    unique_paths = list(set(top_paths))
    path_to_names = {path: [] for path in unique_paths}
    
    for name, path in zip(top_names, top_paths):
        path_to_names[path].append(name)
    
    for path, names_in_path in path_to_names.items():
        folder_info = os.path.basename(os.path.dirname(path))
    
    return top_paths, top_names, top_similarities

def apply_sequential_mappers_to_pipeline(pipe, mapper_models):
    original_forward = pipe.text_encoder.forward

    def sequential_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if hasattr(output, 'last_hidden_state'):
            # 获取last_hidden_state
            hidden_state = output.last_hidden_state
            original_dtype = hidden_state.dtype

            for mapper in mapper_models:
                hidden_state = mapper(hidden_state)
                hidden_state = hidden_state.to(original_dtype)
            result = {}
            for attr in dir(output):
                if not attr.startswith('_'):
                    result[attr] = getattr(output, attr)
            result['last_hidden_state'] = hidden_state
            return (result['last_hidden_state'], result.get('pooler_output', None))
        else:
            hidden_state = output[0]
            original_dtype = hidden_state.dtype

            for mapper in mapper_models:
                hidden_state = mapper(hidden_state)
                hidden_state = hidden_state.to(original_dtype)
            
            return (hidden_state,) + output[1:]

    pipe.text_encoder.forward = sequential_forward
    return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='A photo of a cat', required=True)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--mapper_names', type=str, required=True)
    parser.add_argument('--mapper_paths', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--output_path_template', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=1)
    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
        if not args.output_path_template:
            raise SystemExit("--output_path_template is required when --seeds is used")
    else:
        seeds = [random.randint(1, 1000000) if args.seed is None else args.seed]
        if not args.output_path:
            raise SystemExit("--output_path is required when --seeds is not used")

    model_id = "/share/data/drive_1/umair_nawaz/Stable_diffusion"
    user_prompt = args.prompt

    try:
        mapper_names = [name.strip() for name in args.mapper_names.split(',') if name.strip()]
        mapper_paths = [path.strip() for path in args.mapper_paths.split(',') if path.strip()]
        mapper_dict = dict(zip(mapper_names, mapper_paths))
        for name, path in mapper_dict.items():
            print(f"  - {name}: {path}")
    
    except Exception as e:
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    top_paths, top_names, top_similarities = select_top_mappers_by_similarity(
        user_prompt, mapper_dict, device, top_k=args.top_k
    )

    unique_paths = list(set(top_paths))
    sequential_pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    sequential_pipe = sequential_pipe.to(device)
    mapper_models = []
    for path in unique_paths:
        mapper = Nonlinear().to(device)
        mapper = mapper.to(sequential_pipe.text_encoder.dtype)
        if load_mapper_weights(mapper, path, device):
            mapper_models.append(mapper)
    if mapper_models:
        sequential_pipe = apply_sequential_mappers_to_pipeline(sequential_pipe, mapper_models)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        sequential_image = sequential_pipe(user_prompt, generator=generator).images[0]
        output_path = args.output_path_template.format(seed=seed) if args.output_path_template else args.output_path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sequential_image.save(output_path)
        print(f"Output path: {output_path}")

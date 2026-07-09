import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
from tqdm import tqdm
import torch 

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def load_tokenizer_for_text_encoder(text_encoder_path):
    from transformers import CLIPTokenizer

    model_root = os.path.dirname(os.path.abspath(text_encoder_path.rstrip("/")))
    local_tokenizer_path = os.path.join(model_root, "tokenizer")
    if os.path.isdir(local_tokenizer_path):
        print(f"Tokenizer: {local_tokenizer_path}")
        return CLIPTokenizer.from_pretrained(local_tokenizer_path)

    print("Tokenizer: CompVis/stable-diffusion-v1-4/tokenizer")
    return CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")


def compute_text_embeddings(
    texts, tokenizer, text_encoder, device, embedding_batch_size=256, desc="text embeddings"
):
    embeddings = []
    text_encoder.eval()

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), embedding_batch_size), desc=desc):
            batch_texts = texts[start:start + embedding_batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            outputs = text_encoder(**inputs)
            embeddings.append(outputs.pooler_output.detach().cpu())

    return torch.cat(embeddings)


def get_cached_text4_embeddings(
    texts_4, tokenizer, text_encoder, device,
    embedding_batch_size=256,
    embedding_cache_dir='./embedding_cache',
    force_recompute=False
):
    os.makedirs(embedding_cache_dir, exist_ok=True)
    cache_key = f"text4_embeddings_{len(texts_4)}.pt"
    cache_path = os.path.join(embedding_cache_dir, cache_key)

    if os.path.exists(cache_path) and not force_recompute:
        data = torch.load(cache_path, map_location="cpu")
        return data['emb4']

    emb4 = compute_text_embeddings(texts_4, tokenizer, text_encoder, device, embedding_batch_size, desc="text_embedding_4")
    torch.save({
        'emb4': emb4
    }, cache_path)
    
    return emb4

def get_cached_text1_embeddings(
    texts_1, tokenizer, text_encoder, device,
    embedding_batch_size=256,
    embedding_cache_dir='./embedding_cache',
    force_recompute=False
):

    os.makedirs(embedding_cache_dir, exist_ok=True)

    cache_key = f"text1_embeddings_{len(texts_1)}.pt"
    cache_path = os.path.join(embedding_cache_dir, cache_key)

    if os.path.exists(cache_path) and not force_recompute:
        data = torch.load(cache_path, map_location="cpu")
        return data['emb1']

    emb1 = compute_text_embeddings(texts_1, tokenizer, text_encoder, device, embedding_batch_size, desc="text_embedding_1")
    torch.save({
        'emb1': emb1
    }, cache_path)
    
    return emb1

def load_english_names(file_path):
    names = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            names = [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error:{e}")
    return names


def train_encoder_mapper(
    text_encoder_path,
    texts_1,
    texts_2,
    texts_3,
    texts_4,
    output_model_path='trained_mapper.pth',
    hidden_dim=1536,
    dropout=0.1,
    lr=1e-4,
    epochs=1000,
    batch_size=1,
    embedding_batch_size=256,
    device='cuda'
):

    if not str(device).startswith("cuda"):
        raise RuntimeError("CPU execution is disabled for mapper training. Run with --device cuda and CUDA_VISIBLE_DEVICES set.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to fall back to CPU.")

    device = torch.device(device)
    print(f"CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if len(texts_3) == 1:
        texts_3 = texts_3 * len(texts_2)
    try:
        from transformers import CLIPTextModel
        print(f"Text Encoder: {text_encoder_path}")
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)
        text_encoder.eval()
        tokenizer = load_tokenizer_for_text_encoder(text_encoder_path)

        mapper = Nonlinear(hidden_dim=hidden_dim, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(mapper.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        emb1 = get_cached_text1_embeddings(texts_1, tokenizer, text_encoder, device, embedding_batch_size)

        emb1 = emb1.to(device)
        print(f"emb1 shape: {emb1.shape}")

        emb2 = None
        emb3 = None
        emb4 = None 
        if epochs > 10:
            emb2 = compute_text_embeddings(texts_2, tokenizer, text_encoder, device, embedding_batch_size, desc="text_embedding_2")
            emb3 = compute_text_embeddings(texts_3, tokenizer, text_encoder, device, embedding_batch_size, desc="text_embedding_3")
            emb4 = get_cached_text4_embeddings(texts_4, tokenizer, text_encoder, device, embedding_batch_size)
            emb2 = emb2.to(device)
            emb3 = emb3.to(device)
            emb4 = emb4.to(device)
            print(f"emb2 shape: {emb2.shape}")
            print(f"emb3 shape: {emb3.shape}")
            print(f"emb4 shape: {emb4.shape}")

        print(f"Training Begin, Epochs:{epochs}...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            mapper.train()
            
            if epoch < 10:
                dataset = torch.utils.data.TensorDataset(emb1)
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=1, 
                    shuffle=True,
                    drop_last=False
                )
                
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (Stage 1 - Self-supervised)"):
                    optimizer.zero_grad()

                    batch_emb = batch[0]
                    output = mapper(batch_emb)
                    loss = criterion(output, batch_emb)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_emb.size(0)
            else:
                total_samples = 2000
                num_batches = (total_samples + batch_size - 1) // batch_size
                
                for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs} (Stage 2 - emb2 vs emb3)"):
                    optimizer.zero_grad()

                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, total_samples)

                    batch_emb1_indices = torch.tensor([idx % len(emb1) for idx in range(start_idx, end_idx)], device=device)
                    batch_emb1 = emb1[batch_emb1_indices]
                    batch_emb4 = emb4[batch_emb1_indices]
                    batch_emb2_indices = torch.tensor([idx % len(emb2) for idx in range(start_idx, end_idx)], device=device)
                    batch_emb2 = emb2[batch_emb2_indices]
                    batch_emb3 = emb3[batch_emb2_indices]
                    if batch_emb1.size(0) > 2000:
                        random_indices = torch.randperm(batch_emb1.size(0))[:2000]
                        selected_batch_emb1 = batch_emb1[random_indices]
                        selected_batch_emb4 = batch_emb4[random_indices]
                    else:
                        selected_batch_emb1 = batch_emb1
                        selected_batch_emb4 = batch_emb4
                    batch_output1 = mapper(selected_batch_emb1)
                    batch_output2 = mapper(batch_emb2)
                    batch_output4 = mapper(selected_batch_emb4)
                    loss1 = criterion(batch_output2, batch_emb3)
                    loss2 = criterion(batch_output1, selected_batch_emb1)
                    loss3 = criterion(batch_output4, selected_batch_emb4)  
                    loss = loss1 + loss2 + loss3

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_emb2.size(0)

            if epoch < 10:
                epoch_loss = epoch_loss / len(emb1)
            else:
                epoch_loss = epoch_loss / len(emb2)
                
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.8f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        torch.save({
            'epoch': epochs,
            'model_state_dict': mapper.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, output_model_path)
        
    except Exception as e:
        print(f"Error:{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-encoder-path', type=str, required=True)
    parser.add_argument('--output-model-path', type=str, required=True)
    parser.add_argument('--hidden-dim', type=int, default=1536)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--embedding-batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    # 1. Parser
    parser.add_argument('--text2', type=str, nargs='+', required=True)
    parser.add_argument('--text3', type=str, nargs='+', required=True)
    # parser.add_argument('--text2', type=str, required=True)
    # parser.add_argument('--text3', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)

    file_path = os.path.join(REPO_ROOT, 'laion_bigrams.txt')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text_1 = [line.strip() for line in file if line.strip()]
    
    text_2 = args.text2
    text_3 = args.text3
    text_4 = load_english_names(os.path.join(REPO_ROOT, 'data', 'Wikidata.csv'))
    
    # 开始训练
    train_encoder_mapper(
        text_encoder_path=args.text_encoder_path,
        texts_1=text_1,
        texts_2=text_2,
        texts_3=text_3,
        texts_4=text_4,
        output_model_path=args.output_model_path,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

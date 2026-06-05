import os
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer
from jiwer import wer, cer

from src.dataset.line_dataset import get_line_dataloader
from src.models.trocr_model import SinhalaTrOCR

def compute_metrics(predictions, labels, tokenizer):
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad_token_id to decode
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Filter out empty strings which cause jiwer to fail
    valid_pred_str, valid_label_str = [], []
    for p, l in zip(pred_str, label_str):
        if l.strip() != "":
            valid_pred_str.append(p.strip())
            valid_label_str.append(l.strip())

    if not valid_label_str:
        return 0.0, 0.0

    wer_score = wer(valid_label_str, valid_pred_str)
    cer_score = cer(valid_label_str, valid_pred_str)
    
    return wer_score, cer_score

def train_stage2():
    data_dir = 'src/dataset/SinhalaOCR/images'
    json_file = 'src/dataset/SinhalaOCR/labels.json'
    batch_size = 4
    num_epochs = 20
    learning_rate = 5e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    # Assuming we use the same dataloader for simplicity here. 
    # In practice, you'd split this into train/val loaders.
    train_loader = get_line_dataloader(data_dir, json_file, batch_size=batch_size)
    
    print("Initializing Stage 2 Model...")
    encoder_path = 'outputs/stage1/pretrained_encoder.pth'
    if not os.path.exists(encoder_path):
        print("Warning: Stage 1 encoder weights not found. Using default TrOCR encoder.")
        encoder_path = None
        
    model = SinhalaTrOCR(encoder_pretrained_path=encoder_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained('nlp-rilab/sinbert-base')
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    os.makedirs('outputs/stage2', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Simple evaluation on a batch to calculate metrics
        # (In a real setup, do this on a separate validation set)
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            pixel_values = sample_batch['pixel_values'].to(device)
            labels = sample_batch['labels'].to(device)
            
            generated_ids = model.generate(pixel_values)
            wer_score, cer_score = compute_metrics(generated_ids, labels.cpu(), tokenizer)
            print(f"Sample Batch WER: {wer_score:.4f}, CER: {cer_score:.4f}")
            
        if train_loss < best_loss:
            best_loss = train_loss
            print("Saving best model...")
            torch.save(model.state_dict(), 'outputs/stage2/best_trocr.pth')

if __name__ == '__main__':
    train_stage2()

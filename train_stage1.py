import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.dataset.char_dataset import get_char_datasets
from src.models.vision_encoder import DeiTClassifier

def train_stage1():
    # Configuration
    data_dir = 'Datasets/Dataset454'
    batch_size = 16
    num_epochs = 1
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DataLoaders
    print("Loading datasets...")
    train_loader, valid_loader, test_loader, classes = get_char_datasets(
        data_dir=data_dir, 
        batch_size=batch_size
    )
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")

    # Model
    print("Initializing model...")
    model = DeiTClassifier(num_classes=num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    os.makedirs('outputs/stage1', exist_ok=True)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valid_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), 'outputs/stage1/best_classifier.pth')
            # Also save just the encoder for Stage 2
            torch.save(model.encoder.state_dict(), 'outputs/stage1/pretrained_encoder.pth')

if __name__ == '__main__':
    train_stage1()

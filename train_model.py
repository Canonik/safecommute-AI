import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2

# constants
DATA_DIR = "prepared_data"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "safecommute_edge_model.pth"

# loading data
class TensorAudioDataset(Dataset):
    """
    Loads pre-computed Mel-spectrogram tensors (.pt files).
    """
    def __init__(self, split_dir):
        self.filepaths = []
        self.labels = []
        
        # Class 0: Safe, Class 1: Unsafe 
        for label, class_name in enumerate(['0_safe', '1_unsafe']):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith('.pt'):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load the pre-saved tensor (Shape: [1, n_mels, time_steps])
        features = torch.load(self.filepaths[idx], weights_only=True)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# model definition
class SafeCommuteCNN(nn.Module):
    def __init__(self):
        super(SafeCommuteCNN, self).__init__()
        # Load a lightweight CNN suitable for edge deployment 
        self.backbone = mobilenet_v2(weights='DEFAULT')
        
        # Adapt first layer for 1-channel Mel-spectrograms instead of 3-channel RGB
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        
        # Adapt final classifier for binary output (Safe vs Unsafe) 
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)

# training loop ~1 min
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load Data
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    train_dataset = TensorAudioDataset(train_dir)
    val_dataset = TensorAudioDataset(val_dir)
    
    if len(train_dataset) == 0:
        print(f"{"  HELP  "*100}\n Error: No training data found. Please run prepare_all_data.py first.\n")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model, Loss, and Optimizer
    model = SafeCommuteCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting Fine-Tuning for {EPOCHS} Epochs")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_labels)
                val_loss += v_loss.item()
                
                _, v_predicted = torch.max(v_outputs.data, 1)
                v_total += v_labels.size(0)
                v_correct += (v_predicted == v_labels).sum().item()
                
        val_acc = 100 * v_correct / v_total if v_total > 0 else 0
        #standard training loss/ stats
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader) if len(val_loader) > 0 else 0:.4f} | Val Acc: {val_acc:.2f}%")

    print(f"Training Complete! Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("\n\n\nModel is ready for edge inference!\n\n\n")

if __name__ == "__main__":
    train()
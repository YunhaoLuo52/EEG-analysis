import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import confusion_matrix, classification_report

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
INPUT_SIZE = 1  # Each timestep has 1 feature
HIDDEN_SIZE = 128
NUM_LAYERS = 3
NUM_HEADS = 8
DROPOUT = 0.25

# Custom Dataset Class
class SeizureDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get EEG segment and apply STFT if transform is specified
        eeg_segment = self.features[idx]
        if self.transform:
            eeg_segment = self.transform(eeg_segment)
        
        # Convert to tensor and reshape to [seq_len, 1]
        eeg_tensor = torch.FloatTensor(eeg_segment).unsqueeze(1)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        
        return eeg_tensor, label_tensor

# STFT Transform
def apply_stft(eeg_segment):
    f, t, Zxx = signal.stft(eeg_segment, fs=173.61, nperseg=256)
    # Take magnitude and log transform
    stft = np.log(np.abs(Zxx) + 1e-8)
    # Flatten the time-frequency representation
    return stft.flatten()

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x

# Main Model Architecture
class SeizurePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Time and Frequency Encoders (Transformers)
        self.time_encoder = nn.Sequential(
            *[TransformerEncoderLayer(HIDDEN_SIZE, NUM_HEADS, DROPOUT) for _ in range(2)]
        )
        self.freq_encoder = nn.Sequential(
            *[TransformerEncoderLayer(HIDDEN_SIZE, NUM_HEADS, DROPOUT) for _ in range(2)]
        )
        
        # Recurrent Networks
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 
                           batch_first=True, dropout=DROPOUT)
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                         batch_first=True, dropout=DROPOUT)
        
        # Gating Mechanism
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.Softmax(dim=1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, 1]
        batch_size = x.size(0)
        
        # Time pathway (LSTM)
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]
        lstm_last = lstm_out[:, -1, :]  # Last timestep
        
        # Frequency pathway (GRU)
        gru_out, _ = self.gru(x)  # [batch_size, seq_len, hidden_size]
        gru_last = gru_out[:, -1, :]  # Last timestep
        
        # Transformer encoders
        time_encoded = self.time_encoder(lstm_out)  # [batch_size, seq_len, hidden_size]
        freq_encoded = self.freq_encoder(gru_out)   # [batch_size, seq_len, hidden_size]
        
        # Take last timestep from transformer outputs
        time_last = time_encoded[:, -1, :]
        freq_last = freq_encoded[:, -1, :]
        
        # Gating mechanism
        combined = torch.cat([time_last, freq_last], dim=1)
        gates = self.gate(combined)
        gate1, gate2 = gates.chunk(2, dim=1)
        
        # Weighted features
        weighted_time = time_last * gate1
        weighted_freq = freq_last * gate2
        final_features = torch.cat([weighted_time, weighted_freq], dim=1)
        
        # Classification
        output = self.classifier(final_features)
        
        return output

# Load your dataset
def load_data(features_path, labels_path, test_size=0.2):
    # Load features and labels
    features = pd.read_csv(features_path, header=None).values
    labels = pd.read_csv(labels_path, header=None).values.flatten()
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels)
    
    return X_train, X_test, y_train, y_test

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    return train_losses

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    print(classification_report(all_labels, all_preds, target_names=['Non-Seizure', 'Seizure']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Seizure', 'Seizure'])
    plt.yticks(tick_marks, ['Non-Seizure', 'Seizure'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    # Load your data
    X_train, X_test, y_train, y_test = load_data('train.csv', 'train_label.csv')
    
    # Create datasets with STFT transform
    train_dataset = SeizureDataset(X_train, y_train, transform=apply_stft)
    test_dataset = SeizureDataset(X_test, y_test, transform=apply_stft)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = SeizurePredictionModel().to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    train_losses = train_model(model, train_loader, criterion, optimizer, EPOCHS)
    
    # Plot training loss
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Evaluate the model
    evaluate_model(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'seizure_prediction_model.pth')
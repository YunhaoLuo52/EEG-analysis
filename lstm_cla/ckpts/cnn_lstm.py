import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
class SeizureDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = pd.read_csv(data_path, header=None).values.astype(np.float32)
        self.labels = pd.read_csv(label_path, header=None).values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use all 1280 features, reshape to [seq_len, 1]
        features = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.25
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

# Define paths
data_path = '/Users/yunhaoluo/Desktop/EEG_analysis/processed_data/'
train_features_path = data_path + "train.csv"
train_labels_path = data_path + "train_labels.csv"
test_features_path = data_path + "test.csv"
test_labels_path = data_path + "test_labels.csv"

# Initialize datasets and dataloaders
train_dataset = SeizureDataset(train_features_path, train_labels_path)
test_dataset = SeizureDataset(test_features_path, test_labels_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
model = LSTMClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training parameters
epochs = 150
best_acc = 0
train_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Store training loss
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# Testing and Visualization
def evaluate_and_visualize(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Seizure', 'Seizure'],
                yticklabels=['Non-Seizure', 'Seizure'])
    plt.title('Confusion Matrix')
    plt.show()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 3. Sample Visualization
    sample_idx = np.random.randint(0, len(test_dataset))
    sample_data, sample_label = test_dataset[sample_idx]
    with torch.no_grad():
        sample_prob = torch.sigmoid(model(sample_data.unsqueeze(0).to(device))).item()
    
    plt.figure(figsize=(10, 4))
    plt.plot(sample_data.numpy())
    plt.title(f'Sample Prediction\nTrue: {int(sample_label)} ({sample_label:.2f}), Pred: {sample_prob:.2f}')
    plt.xlabel('Feature Index (0-1279)')
    plt.ylabel('Value')
    plt.show()

    # 4. Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Seizure', 'Seizure']))

# Run evaluation
evaluate_and_visualize(model, test_loader)

# Save final model
torch.save(model.state_dict(), 'seizure_lstm_final.pth')
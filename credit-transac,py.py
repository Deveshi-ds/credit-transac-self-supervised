import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Step 1: Load and Preprocess the Dataset
file_path = "C:\\Users\\deves\\Downloads\\creditdata.csv"
data = pd.read_csv(file_path)

# Check for missing values and fill them with the column mean if any
if data.isnull().sum().sum() > 0:
    data = data.fillna(data.mean())

# Scale 'Amount' feature and other features for better model convergence
scaler = MinMaxScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
feature_columns = [col for col in data.columns if col != 'Class']
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split data into train and test sets
X = data[feature_columns]
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays for PyTorch compatibility
X_train_array = X_train.values
X_test_array = X_test.values

# Step 2: Define PyTorch Dataset for Contrastive Learning
class ContrastiveDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transaction = self.data[idx]
        positive_sample = transaction + np.random.normal(0, 0.01, size=transaction.shape)
        negative_idx = np.random.choice(np.delete(np.arange(len(self.data)), idx))
        negative_sample = self.data[negative_idx]
        return (torch.tensor(transaction, dtype=torch.float32),
                torch.tensor(positive_sample, dtype=torch.float32),
                torch.tensor(negative_sample, dtype=torch.float32))

# Step 3: Define the Contrastive Learning Model
class TransactionEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TransactionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Step 4: Define the Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_distance = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean((pos_distance ** 2) + torch.relu(self.margin - neg_distance) ** 2)
        return loss

# Step 5: Prepare DataLoader for Training
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10
EMBEDDING_DIM = 128

train_dataset = ContrastiveDataset(X_train_array)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 6: Train the Model
model = TransactionEncoder(input_dim=X_train_array.shape[1], embedding_dim=EMBEDDING_DIM)
contrastive_loss = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    total_loss = 0
    for anchor, positive, negative in train_loader:
        optimizer.zero_grad()
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        loss = contrastive_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

print("Training complete. Model is ready for embeddings.")

# Step 7: Create DataLoader for Full Dataset for Embedding Generation
full_dataset = ContrastiveDataset(X_train_array)
all_data_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Generate embeddings for all transactions
def generate_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for transactions, _, _ in data_loader:
            embeddings.append(model(transactions).numpy())
    return np.vstack(embeddings)

embeddings = generate_embeddings(model, all_data_loader)

# Step 8: Anomaly Detection with Nearest Neighbors and DBSCAN
k = 10
nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
distances, _ = nbrs.kneighbors(embeddings)
anomaly_scores_knn = distances.mean(axis=1)

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
anomaly_scores_dbscan = (dbscan.labels_ == -1).astype(int) * anomaly_scores_knn

anomaly_scores_knn = (anomaly_scores_knn - np.min(anomaly_scores_knn)) / (np.max(anomaly_scores_knn) - np.min(anomaly_scores_knn))
anomaly_scores_dbscan = (anomaly_scores_dbscan - np.min(anomaly_scores_dbscan)) / (np.max(anomaly_scores_dbscan) - np.min(anomaly_scores_dbscan))
combined_anomaly_scores = 0.6 * anomaly_scores_knn + 0.4 * anomaly_scores_dbscan

# Step 9: Optimize Anomaly Threshold
best_threshold = 0
best_f1 = 0
best_metrics = {}

for percentile in range(85, 96, 1):
    threshold = np.percentile(combined_anomaly_scores, percentile)
    y_pred = (combined_anomaly_scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred, average='binary')
    roc_auc = roc_auc_score(y_train, combined_anomaly_scores)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

print(f"Best Threshold: {best_threshold:.4f}")
print(f"Improved ROC-AUC Score: {best_metrics['roc_auc']:.4f}")
print(f"Improved Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1-Score: {best_metrics['f1']:.4f}")

# Show sample of detected anomalies
anomalous_transactions = X_train.iloc[combined_anomaly_scores > best_threshold]
print("Sample of Improved Anomalous Transactions Detected:")
print(anomalous_transactions.head())

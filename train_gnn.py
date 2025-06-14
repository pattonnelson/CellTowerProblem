import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tower_gnn_dataset import TowerGraphDataset
from tower_gnn_model import TowerGNN
import numpy as np
import matplotlib.pyplot as plt


# === Config ===
JSON_PATH = "tower_training_data.json"
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 10
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)

# === Load and Split Dataset ===
full_dataset = TowerGraphDataset(JSON_PATH)
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model, Optimizer, Loss ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TowerGNN(in_channels=8, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# === Training Loop with Early Stopping ===
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    # --- Training ---
    model.train()
    train_losses = []
    deadzone_losses = []
    cost_losses = []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        deadzone_loss = loss_fn(pred[:, 0], batch.y[:, 0])
        cost_loss = loss_fn(pred[:, 1], batch.y[:, 1])
        loss = 5.0 * deadzone_loss + 1.0 * cost_loss  # Tune weights as needed
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        deadzone_losses.append(deadzone_loss.item())
        cost_losses.append(cost_loss.item())
    avg_dz = sum(deadzone_losses) / len(deadzone_losses)
    avg_cost = sum(cost_losses) / len(cost_losses)
    avg_total = sum(train_losses) / len(train_losses)
    # --- Validation ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            val_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)

    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_tower_gnn.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"Best model saved to best_tower_gnn.pt")

plt.plot(deadzone_losses, label="Deadzone Loss")
plt.plot(cost_losses, label="Cost Loss")
plt.legend()
plt.title("Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()

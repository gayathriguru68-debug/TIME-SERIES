import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Synthetic Multivariate Dataset
# =========================
def generate_multivariate_time_series(
    timesteps=2000,
    n_features=5
):
    t = np.arange(timesteps)
    data = []

    for i in range(n_features):
        signal = (
            np.sin(0.02 * t + i)
            + 0.5 * np.sin(0.05 * t)
            + np.random.normal(0, 0.2, size=timesteps)
        )
        data.append(signal)

    return np.stack(data, axis=1)


# =========================
# Dataset Class
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_len, output_len):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self):
        return len(self.data) - self.input_len - self.output_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[
            idx + self.input_len : idx + self.input_len + self.output_len, 0
        ]  # predict feature 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =========================
# Bahdanau Attention
# =========================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs: [B, T, H]
        # hidden: [B, H]
        hidden = hidden.unsqueeze(1)
        score = self.V(
            torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        )
        attention_weights = torch.softmax(score, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights


# =========================
# LSTM + Attention Model
# =========================
class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size,
        num_layers,
        output_len
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = BahdanauAttention(hidden_size)

        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        encoder_outputs, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]
        context, attn_weights = self.attention(encoder_outputs, hidden)
        output = self.fc(context)
        return output, attn_weights


# =========================
# Training Function
# =========================
def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds, _ = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds, _ = model(x)
                val_loss += criterion(preds, y).item()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train MSE: {train_loss/len(train_loader):.4f} | "
            f"Val MSE: {val_loss/len(val_loader):.4f}"
        )


# =========================
# Attention Visualization
# =========================
def plot_attention(attention_weights):
    attn = attention_weights.squeeze(-1).cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(attn.T, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.xlabel("Time Step")
    plt.ylabel("Attention")
    plt.title("Attention Weights Over Input Sequence")
    plt.show()


# =========================
# Main Execution
# =========================
def main():
    # Parameters
    INPUT_LEN = 30
    OUTPUT_LEN = 10
    N_FEATURES = 5
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3

    # Generate data
    raw_data = generate_multivariate_time_series(
        timesteps=2500,
        n_features=N_FEATURES
    )

    scaler = StandardScaler()
    data = scaler.fit_transform(raw_data)

    # Train / Val / Test split
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    train_ds = TimeSeriesDataset(train_data, INPUT_LEN, OUTPUT_LEN)
    val_ds = TimeSeriesDataset(val_data, INPUT_LEN, OUTPUT_LEN)
    test_ds = TimeSeriesDataset(test_data, INPUT_LEN, OUTPUT_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=1)

    # Model
    model = LSTMAttentionModel(
        n_features=N_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_len=OUTPUT_LEN
    ).to(DEVICE)

    # Train
    train_model(model, train_loader, val_loader, EPOCHS, LR)

    # Test + Attention Analysis
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x = x.to(DEVICE)
        preds, attn_weights = model(x)

    print("True Future:", y.numpy().flatten())
    print("Predicted :", preds.cpu().numpy().flatten())

    plot_attention(attn_weights)


if __name__ == "__main__":
    main()

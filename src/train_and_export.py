import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

risultati = {"1": 0, "X": 1, "2": 2}
Xlist = []
Ylist1x2 = []

with open("matches_enriched_xg_top5_2014_2026.csv", "r") as f:
    next(f)
    for riga in f:
        parti = riga.strip().split(",")
        temp = [float(parti[4]), float(parti[5])]
        for i in range(11, 66):
            if i != 49:
                temp.append(float(parti[i]))
        Xlist.append(temp)
        Ylist1x2.append(risultati[parti[8]])

N_FEATURES = len(Xlist[0])

# Split temporale a 3 vie: 80% Train, 10% Validation, 10% Test
split_train = int(0.80 * len(Xlist))
split_val = int(0.90 * len(Xlist))

Xtr_raw = torch.tensor(Xlist[:split_train], dtype=torch.float32)
Ytr1x2 = torch.tensor(Ylist1x2[:split_train], dtype=torch.long)

Xval_raw = torch.tensor(Xlist[split_train:split_val], dtype=torch.float32)
Yval1x2 = torch.tensor(Ylist1x2[split_train:split_val], dtype=torch.long)

Xtest_raw = torch.tensor(Xlist[split_val:], dtype=torch.float32)
Ytest1x2 = torch.tensor(Ylist1x2[split_val:], dtype=torch.long)

# Scaler calcolato esclusivamente sul Train Set
median = torch.median(Xtr_raw, dim=0).values
q25 = torch.quantile(Xtr_raw, 0.25, dim=0)
q75 = torch.quantile(Xtr_raw, 0.75, dim=0)
iqr = q75 - q25

Xtr = (Xtr_raw - median) / (iqr + 1e-8)
Xval = (Xval_raw - median) / (iqr + 1e-8)
Xtest = (Xtest_raw - median) / (iqr + 1e-8)


class BettingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 25)
        self.fc2 = nn.Linear(25, 16)
        self.fc3 = nn.Linear(16, 3)
        self.drop = nn.Dropout(p=0.10)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = self.drop(h)
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)


model = BettingNet(N_FEATURES)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.008)
criterion = nn.CrossEntropyLoss(label_smoothing=0.06)

epoche_max = 6000
dimBatch = 1024
patience = 800
counter_patience = 0
best_brier = float("inf")
best_state = None

y_tr_one_hot = F.one_hot(Ytr1x2, num_classes=3).float()
y_val_one_hot = F.one_hot(Yval1x2, num_classes=3).float()
y_test_one_hot = F.one_hot(Ytest1x2, num_classes=3).float()

for i in range(epoche_max):
    model.train()
    ind = torch.randint(0, len(Xtr), (dimBatch,))
    X_batch = Xtr[ind]
    Y_batch = Ytr1x2[ind]

    optimizer.zero_grad()
    logits = model(X_batch)
    loss = criterion(logits, Y_batch)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Valutazione periodica solo su VALIDATION SET per early stopping
    if i % 50 == 0:
        model.eval()
        with torch.no_grad():
            logits_val = model(Xval)
            probs_val = F.softmax(logits_val, dim=1)
            brier_val = torch.mean(torch.sum((probs_val - y_val_one_hot) ** 2, dim=1)).item()

            if brier_val < best_brier:
                best_brier = brier_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                counter_patience = 0
            else:
                counter_patience += 50

            if counter_patience >= patience:
                print(f"Early stopping at step {i} (best val brier: {best_brier:.4f})")
                break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    # Valutazione Train
    logits_tr = model(Xtr)
    probs_tr = F.softmax(logits_tr, dim=1)
    brier_tr = torch.mean(torch.sum((probs_tr - y_tr_one_hot) ** 2, dim=1))
    acc_tr = (torch.argmax(logits_tr, dim=1) == Ytr1x2).float().mean()

    # Valutazione Validation
    logits_val = model(Xval)
    probs_val = F.softmax(logits_val, dim=1)
    brier_val = torch.mean(torch.sum((probs_val - y_val_one_hot) ** 2, dim=1))
    acc_val = (torch.argmax(logits_val, dim=1) == Yval1x2).float().mean()

    # Valutazione Test Set (100% blind out-of-sample)
    logits_test = model(Xtest)
    loss_test = F.cross_entropy(logits_test, Ytest1x2)
    probs_test = F.softmax(logits_test, dim=1)
    brier_test = torch.mean(torch.sum((probs_test - y_test_one_hot) ** 2, dim=1))
    acc_test = (torch.argmax(logits_test, dim=1) == Ytest1x2).float().mean()

    print(f"Train Brier: {brier_tr.item():.4f} | Train Acc: {acc_tr.item():.2%}")
    print(f"Val Brier:   {brier_val.item():.4f} | Val Acc:   {acc_val.item():.2%}")
    print(f"Test Loss:   {loss_test.item():.4f} | Test Brier: {brier_test.item():.4f} | Test Acc: {acc_test.item():.2%}")

torch.save(
    {
        "model_state": model.state_dict(),
        "median": median.detach(),
        "iqr": iqr.detach(),
        "n_features": N_FEATURES,
    },
    "modello_calcio_v1.pt",
)







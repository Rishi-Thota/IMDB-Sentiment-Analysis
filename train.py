# train.py - training loop, early stopping, and saving the best checkpoint
 
import os
import torch
import torch.nn as nn
 
from model import SentimentLSTM
 
 
def run_training(
   train_loader,
   val_loader,
   vocab_size: int,
   embed_dim: int   = 128,
   hidden_dim: int  = 128,
   num_layers: int  = 2,
   dropout: float   = 0.3,
   lr: float        = 1e-3,
   epochs: int      = 5,
   patience: int    = 3,
   save_dir: str    = "models",
   device: str|None = None,
):
   """
   Trains a SentimentLSTM and returns the best version of it.
 
   Saves a checkpoint whenever val loss improves. If it stops
   improving for `patience` epochs in a row, training cuts off early.
 
   Returns the model (reloaded from best checkpoint) and a history
   dict with train_loss, val_loss, and val_acc lists.
   """
   if device is None:
       device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"[train] running on: {device}")
 
   # build model and move it to the right device
   net = SentimentLSTM(
       vocab_size=vocab_size,
       embed_dim=embed_dim,
       hidden_dim=hidden_dim,
       num_layers=num_layers,
       dropout=dropout,
   ).to(device)
 
   loss_fn   = nn.BCELoss()
   optimizer = torch.optim.Adam(net.parameters(), lr=lr)
 
   n_params = sum(p.numel() for p in net.parameters())
   print(f"[train] total parameters: {n_params:,}")
 
   os.makedirs(save_dir, exist_ok=True)
   ckpt_path = os.path.join(save_dir, "best_model.pt")
 
   log = {"train_loss": [], "val_loss": [], "val_acc": []}
   best_loss    = float("inf")
   stale_epochs = 0
 
   for ep in range(1, epochs + 1):
 
       # --- training pass ---
       net.train()
       total_loss = 0.0
       for seqs, lbls in train_loader:
           seqs, lbls = seqs.to(device), lbls.to(device)
 
           optimizer.zero_grad()
           out  = net(seqs)
           loss = loss_fn(out, lbls)
           loss.backward()
           optimizer.step()
 
           total_loss += loss.item() * seqs.size(0)
 
       epoch_train_loss = total_loss / len(train_loader.dataset)
 
       # --- validation pass ---
       val_loss, val_acc = score(net, val_loader, loss_fn, device)
 
       log["train_loss"].append(epoch_train_loss)
       log["val_loss"].append(val_loss)
       log["val_acc"].append(val_acc)
 
       print(
           f"  epoch {ep}/{epochs}  |  "
           f"train loss: {epoch_train_loss:.4f}  |  "
           f"val loss: {val_loss:.4f}  |  "
           f"val acc: {val_acc:.2%}"
       )
 
       # save if this is the best we've seen, otherwise tick the patience counter
       if val_loss < best_loss:
           best_loss    = val_loss
           stale_epochs = 0
           torch.save(net.state_dict(), ckpt_path)
       else:
           stale_epochs += 1
           if stale_epochs >= patience:
               print(f"[train] early stop at epoch {ep} - no improvement for {patience} epochs")
               break
 
   # swap in the best weights before returning
   net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
   print(f"[train] best checkpoint loaded from {ckpt_path}")
   return net, log
 
 
# ---------------------------------------------------------------------------
# Scoring helper - also used in main.py for the test set
# ---------------------------------------------------------------------------
 
def evaluate(net, loader, loss_fn, device):
   # runs through a dataloader and returns avg loss + accuracy
   net.eval()
   total_loss = 0.0
   hits       = 0
 
   with torch.no_grad():
       for seqs, lbls in loader:
           seqs, lbls = seqs.to(device), lbls.to(device)
           out        = net(seqs)
           loss       = loss_fn(out, lbls)
           total_loss += loss.item() * seqs.size(0)
           hits       += ((out >= 0.5).float() == lbls).sum().item()
 
   avg_loss = total_loss / len(loader.dataset)
   acc      = hits / len(loader.dataset)
   return avg_loss, acc
 
 
# kept as an alias so main.py and test_model.py don't break
score = evaluate





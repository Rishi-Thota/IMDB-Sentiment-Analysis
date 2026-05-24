# main.py - kicks off training, evaluation, and a quick demo
#
# python main.py               # train from scratch then eval
# python main.py --skip-train  # skip straight to eval using a saved model
 
import os
import argparse
import torch
import torch.nn as nn
 
from dataset import build_loaders, TokenMap
from train import run_training, evaluate
from predict import predict_sentiment, load_model
from model import SentimentLSTM
 
 
# ---------------------------------------------------------------------------
# Config - change these if you want to experiment
# ---------------------------------------------------------------------------
 
ROOT_DIR    = "data"
SAVE_DIR    = "models"
VOCAB_FILE  = os.path.join(SAVE_DIR, "vocab.json")
CKPT_FILE   = os.path.join(SAVE_DIR, "best_model.pt")
 
VOCAB_CAP   = 25_000
SEQ_LEN     = 256
BATCH       = 64
EMBED_DIM   = 128
HIDDEN_DIM  = 128
N_LAYERS    = 2
DROPOUT     = 0.3
LR          = 1e-3
EPOCHS      = 5
PATIENCE    = 3
 
 
def run():
   parser = argparse.ArgumentParser(description="IMDB sentiment classifier")
   parser.add_argument(
       "--skip-train",
       action="store_true",
       help="don't train, just load the saved checkpoint and run eval",
   )
   args   = parser.parse_args()
   device = "cuda" if torch.cuda.is_available() else "cpu"
 
   # ------------------------------------------------------------------
   # step 1 - load and prep the data
   # ------------------------------------------------------------------
   print("=" * 60)
   print("  IMDB Movie Review Sentiment Analysis")
   print("=" * 60)
   print()
 
   train_loader, val_loader, test_loader, tok_map = build_loaders(
       root=ROOT_DIR,
       vocab_limit=VOCAB_CAP,
       seq_len=SEQ_LEN,
       batch=BATCH,
   )
 
   # write vocab to disk so predict.py can use it later without reloading data
   os.makedirs(SAVE_DIR, exist_ok=True)
   tok_map.dump(VOCAB_FILE)
   print(f"[main] vocab saved -> {VOCAB_FILE}")
   print()
 
   # ------------------------------------------------------------------
   # step 2 - train or load
   # ------------------------------------------------------------------
   if args.skip_train:
       print("[main] --skip-train set, loading checkpoint instead ...")
       model = SentimentLSTM(
           vocab_size=len(tok_map),
           embed_dim=EMBED_DIM,
           hidden_dim=HIDDEN_DIM,
           num_layers=N_LAYERS,
           dropout=DROPOUT,
       )
       model.load_state_dict(
           torch.load(CKPT_FILE, map_location=device, weights_only=True)
       )
       model.to(device)
       print(f"[main] loaded weights from {CKPT_FILE}")
   else:
       print("[main] starting training ...")
       print("-" * 60)
       model, history = run_training(
           train_loader=train_loader,
           val_loader=val_loader,
           vocab_size=len(tok_map),
           embed_dim=EMBED_DIM,
           hidden_dim=HIDDEN_DIM,
           num_layers=N_LAYERS,
           dropout=DROPOUT,
           lr=LR,
           epochs=EPOCHS,
           patience=PATIENCE,
           save_dir=SAVE_DIR,
           device=device,
       )
       print("-" * 60)
       print()
 
   # ------------------------------------------------------------------
   # step 3 - evaluate on the held-out test set
   # ------------------------------------------------------------------
   criterion = nn.BCELoss()
   test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   print(f"[main] test loss: {test_loss:.4f}  |  test accuracy: {test_acc:.2%}")
   print()
 
   # ------------------------------------------------------------------
   # step 4 - run a few example reviews through the model
   # ------------------------------------------------------------------
   print("-" * 60)
   print("  Demo Predictions")
   print("-" * 60)
 
   samples = [
       "This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat.",
       "Terrible film. The plot was boring, the dialogue was awful, and I almost fell asleep.",
       "A decent movie with some good moments, but overall it felt average and forgettable.",
       "One of the best movies I have ever seen! A true masterpiece of cinema.",
       "I hated every minute of this disaster. Complete waste of time and money.",
   ]
 
   for review in samples:
       label, conf = predict_sentiment(review, model, tok_map, device)
       snippet = review[:80] + ("..." if len(review) > 80 else "")
       print(f"  \"{snippet}\"")
       print(f"    => {label}  (confidence: {conf:.2%})")
       print()
 
   print("=" * 60)
   print("  done!  run  python predict.py  for interactive mode.")
   print("=" * 60)
 
 
if __name__ == "__main__":
   run()
 

# model.py - the actual LSTM architecture, nothing else lives here
 
import torch
import torch.nn as nn
 
 
class ReviewClassifier(nn.Module):
   # bidirectional LSTM -> dropout -> linear -> sigmoid
   # we grab the last hidden state from both directions, cat them together,
   # and squish that through a single output neuron to get a probability
 
   def __init__(
       self,
       vocab_size: int,
       embed_dim: int   = 128,
       hidden_dim: int  = 128,
       num_layers: int  = 2,
       dropout: float   = 0.3,
       pad_idx: int     = 0,
   ):
       super().__init__()
 
       # embedding layer - pad index gets zeroed out so it doesn't contribute
       self.embed = nn.Embedding(
           num_embeddings=vocab_size,
           embedding_dim=embed_dim,
           padding_idx=pad_idx,
       )
 
       self.rnn = nn.LSTM(
           input_size=embed_dim,
           hidden_size=hidden_dim,
           num_layers=num_layers,
           batch_first=True,
           bidirectional=True,
           dropout=dropout if num_layers > 1 else 0.0,  # dropout only makes sense with >1 layer
       )
 
       self.drop = nn.Dropout(dropout)
 
       # hidden_dim * 2 because we're concatenating forward + backward states
       self.fc  = nn.Linear(hidden_dim * 2, 1)
       self.act = nn.Sigmoid()
 
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       # x is (batch, seq_len) integer token ids
 
       vecs = self.embed(x)               # (batch, seq_len, embed_dim)
 
       _, (h, _) = self.rnn(vecs)         # h is (num_layers*2, batch, hidden_dim)
 
       # hidden[-2] is the last layer's forward pass, hidden[-1] is backward
       # cat them so we have the full context from both ends of the sequence
       h_combined = torch.cat((h[-2], h[-1]), dim=1)   # (batch, hidden_dim*2)
 
       out = self.drop(h_combined)
       out = self.fc(out)                 # (batch, 1)
       out = self.act(out).squeeze(1)     # (batch,)  - values between 0 and 1
 
       return out


predict.py 
# predict.py - run any review through the trained model and get a sentiment + confidence score
 
import os
import torch
import numpy as np
 
from dataset import TokenMap, normalize_review, fix_lengths
from model import SentimentLSTM
 
 
# ---------------------------------------------------------------------------
# Config - keep these in sync with whatever you used during training
# ---------------------------------------------------------------------------
 
VOCAB_FILE = os.path.join("models", "vocab.json")
CKPT_FILE  = os.path.join("models", "best_model.pt")
SEQ_LEN    = 256
EMBED_DIM  = 128
HIDDEN_DIM = 128
N_LAYERS   = 2
DROPOUT    = 0.3
 
 
def load_model(device: str = "cpu"):
   # pull vocab off disk first
   tok_map = TokenMap()
   tok_map.restore(VOCAB_FILE)
 
   # rebuild the exact same architecture then load the weights
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
   model.eval()
   return model, tok_map
 
 
def predict_sentiment(review_text: str, model, tok_map, device: str = "cpu"):
   """
   Takes a raw review string and returns a (label, confidence) tuple.
   label is either "Positive" or "Negative", confidence is the
   probability of whichever class won.
   """
   # clean -> encode -> pad -> tensor, same pipeline as training
   cleaned = normalize_review(review_text)
   encoded = tok_map.transform(cleaned)
   padded  = fix_lengths([encoded], max_len=SEQ_LEN)
   tensor  = torch.tensor(padded, dtype=torch.long).to(device)
 
   with torch.no_grad():
       prob = model(tensor).item()
 
   # prob is P(positive), so flip it for negative confidence
   if prob >= 0.5:
       return "Positive", prob
   return "Negative", 1.0 - prob
 
 
# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
 
def interactive_mode():
   # figure out if we have a gpu available
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"[predict] loading model on {device} ...")
 
   model, tok_map = load_model(device)
   print("[predict] ready! paste a review below, or type 'quit' to stop.\n")
 
   while True:
       review = input("Review > ").strip()
       if not review or review.lower() in ("quit", "exit", "q"):
           print("bye!")
           break
       label, confidence = predict_sentiment(review, model, tok_map, device)
       print(f"  => {label}  (confidence: {confidence:.2%})\n")
 
 
if __name__ == "__main__":
   interactive_mode()


Test_model.py 
# quick sanity check - just run this after training to make sure predictions look right
 
from predict import load_model, predict_sentiment
 
model, tok_map = load_model()
print("loaded model and vocab ok!")
print(f"vocab size: {len(tok_map)}")
print()
 
# a mix of clear positives, clear negatives, and some murky middle-ground ones
cases = [
   ("This movie was absolutely incredible! The acting was phenomenal and the plot was gripping from start to finish.", "Positive"),
   ("Worst movie I have ever seen. Terrible acting, boring plot, and a complete waste of my time.", "Negative"),
   ("The film was okay, nothing special but not terrible either. Just an average experience overall.", "Neutral-ish"),
   ("A masterpiece of filmmaking. Every scene was beautifully crafted and the soundtrack was amazing.", "Positive"),
   ("I walked out of the theater halfway through. Absolutely dreadful and painful to watch.", "Negative"),
   ("The special effects were good but the story made no sense and the characters were flat.", "Mixed/Negative"),
]
 
print("=" * 70)
print("  TEST PREDICTIONS")
print("=" * 70)
 
hits       = 0
n_clear    = 0
 
for review, expected in cases:
   label, conf = predict_sentiment(review, model, tok_map)
   snippet    = review[:65] + ("..." if len(review) > 65 else "")
   ambiguous  = "Neutral" in expected or "Mixed" in expected
 
   if not ambiguous:
       n_clear += 1
       if label in expected:
           hits  += 1
           badge  = "PASS"
       else:
           badge  = "FAIL"
   else:
       # can't really grade these, just show what the model thinks
       badge = "OK  "
 
   print(f"  [{badge}] \"{snippet}\"")
   print(f"         predicted: {label} ({conf:.2%})  |  expected: {expected}")
   print()
 
print(f"  clear-cut results: {hits}/{n_clear} correct")
print("=" * 70)

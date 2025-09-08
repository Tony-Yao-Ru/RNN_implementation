# rnn_driver_simple.py
import numpy as np
import argparse

from RNN_Numpy import RNN_Numpy
from RNN_Pytorch import RNN_Pytorch

# ---------------- Defaults (edit as you like) ----------------
MODEL = "pytorch"          # "pytorch" or "numpy"
CELL = "lstm"              # "lstm", "gru" (torch only), or "vanilla"
HIDDEN = 64
EPOCH = 50
SEQ_LEN = 12
L_RATE = 0.01
TEMP = 0.8
GEN_STEPS = 200
PROMPT = ""

# A tiny corpus (kept simple)
TEXT = (
    "In the quiet lab at 2:17 a.m., Tony tuned the Keithley 2400—again. "
    "Logs scrolled: init… range set… compliance OK. The sine rig hummed; data "
    "dribbled into HDF5, neatly indexed under /entry/instrument. "
    "‘Hello, model,’ he joked, nudging an LSTM to predict the next token. "
    "Sometimes it said hello back."
)

# ---------------- Helpers ----------------
def build_vocab(text):
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    return vocab, stoi, itos

def one_hot_np(idx, V):
    v = np.zeros((V,), dtype=np.float32)
    v[idx] = 1.0
    return v

def make_dataset_np(text, T, stoi, V):
    X, Y = [], []
    for i in range(len(text) - T):
        ctx = text[i:i+T]
        nxt = text[i+T]
        x = np.stack([one_hot_np(stoi[ch], V) for ch in ctx], axis=0)[..., None]   # (T,V,1)
        y = np.zeros((T, V), dtype=np.float32)
        y[-1] = one_hot_np(stoi[nxt], V)
        y = y[..., None]                                                            # (T,V,1)
        X.append(x); Y.append(y)
    return X, Y

def make_dataset_torch(text, T, stoi, V):
    # RNN_Pytorch.train(X, Y) as provided: X=(T,V), Y=(1,V) with only final step supervised
    X, Y = [], []
    for i in range(len(text) - T):
        ctx = text[i:i+T]
        nxt = text[i+T]
        x = np.stack([one_hot_np(stoi[ch], V) for ch in ctx], axis=0)               # (T,V)
        y = one_hot_np(stoi[nxt], V)[None, :]                                       # (1,V)
        X.append(x); Y.append(y)
    return X, Y

def trim_or_pad_seed(seed, T, vocab):
    seed = "".join(ch if ch in vocab else " " for ch in seed)
    if len(seed) < T:
        seed = (" " * (T - len(seed))) + seed
    else:
        seed = seed[-T:]
    return seed

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny char-level RNN (NumPy or PyTorch) and generate text.")
    parser.add_argument("--model", default=MODEL, choices=["numpy", "pytorch"])
    parser.add_argument("--cell", default=CELL, choices=["lstm", "vanilla", "gru"])
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--epochs", type=int, default=EPOCH)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--l_rate", type=float, default=L_RATE)
    parser.add_argument("--temperature", type=float, default=TEMP)
    parser.add_argument("--gen_steps", type=int, default=GEN_STEPS)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    args = parser.parse_args()

    # Vocab + data
    vocab, stoi, itos = build_vocab(TEXT)
    V = len(vocab)
    T = args.seq_len

    if args.model == "numpy":
        X, Y = make_dataset_np(TEXT, T, stoi, V)
        model = RNN_Numpy(input_size=V, hidden_size=args.hidden, output_size=V,
                          cell=args.cell, clip_value=5.0)
        print("Training (NumPy)...")
        model.train(X, Y, epochs=args.epochs, lr=args.l_rate)

        seed = trim_or_pad_seed(args.prompt or "hello, model—", T, vocab)
        x_seed = np.stack([one_hot_np(stoi[ch], V) for ch in seed], axis=0)[..., None]
        idxs = model.predict(x_seed, steps=args.gen_steps, temperature=args.temperature)
        out = "".join(itos[i] for i in idxs)

    else:
        # PyTorch backend
        X, Y = make_dataset_torch(TEXT, T, stoi, V)
        model = RNN_Pytorch(input_size=V, hidden_size=args.hidden, output_size=V,
                            cell=args.cell, clip_value=5.0)
        print("Training (PyTorch)...")
        model.train(X, Y, epochs=args.epochs, lr=args.l_rate)

        # Simple generation loop using model.rnn/model.fc (keeps hidden state)
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PyTorch not installed but --model pytorch was selected.") from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.rnn.to(device); model.fc.to(device)
        model.rnn.eval(); model.fc.eval()

        def one_hot_torch(idx, V, device):
            v = torch.zeros(V, dtype=torch.float32, device=device)
            v[idx] = 1.0
            return v

        seed = trim_or_pad_seed(args.prompt or "hello, model—", T, vocab)
        with torch.no_grad():
            x_seed = torch.stack([one_hot_torch(stoi[ch], V, device) for ch in seed], dim=0).unsqueeze(0)  # (1,T,V)
            out, hidden = model.rnn(x_seed)
            logits = model.fc(out[:, -1, :])

            idxs = []
            for _ in range(args.gen_steps):
                logits_t = logits / max(1e-6, args.temperature)
                probs = torch.softmax(logits_t, dim=-1).squeeze(0)
                idx = torch.multinomial(probs, 1).item()
                idxs.append(idx)
                x_next = one_hot_torch(idx, V, device).view(1, 1, V)
                out, hidden = model.rnn(x_next, hidden)
                logits = model.fc(out[:, -1, :])

        out = "".join(itos[i] for i in idxs)

    print("\n=== Generation ===")
    print("Seed:", repr(seed))
    print("Out :", out)

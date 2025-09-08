import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN_Pytorch:
    def __init__(self, input_size, hidden_size, output_size, cell="lstm", clip_value=5.0):
        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.I, self.H, self.O = input_size, hidden_size, output_size
        self.cell = cell

        if cell == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif cell == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        self.clip_value = clip_value

    def forward(self, x):
        x_tensor = self.torch.tensor(x, dtype=self.torch.float32)
        out, _ = self.rnn(x_tensor)
        out = self.fc(out)
        return out.detach().numpy()
    
    def train(self, X, Y, epochs=10, lr=0.01):
        # move modules to device
        self.rnn.to(DEVICE)
        self.fc.to(DEVICE)

        optimizer = optim.Adam(list(self.rnn.parameters()) + list(self.fc.parameters()), lr=lr)

        # final-step only if Y has shape (1, V)
        final_step_only = (Y[0].shape[0] == 1)
        criterion = nn.CrossEntropyLoss()

        # ---- live plot setup ----
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Training Loss (live)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        line, = ax.plot([], [], marker="o")  # start empty; no explicit color
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0

            for x_seq, y_seq in zip(X, Y):
                x_tensor = self.torch.tensor(x_seq, dtype=self.torch.float32).unsqueeze(0).to(DEVICE)  # (1,T,V)
                y_tensor = self.torch.tensor(y_seq, dtype=self.torch.float32).unsqueeze(0).to(DEVICE)  # (1,T,V) or (1,1,V)

                optimizer.zero_grad()
                out, _ = self.rnn(x_tensor)   # (1,T,H)
                out = self.fc(out)            # (1,T,O)

                if final_step_only:
                    target_idx = y_tensor[:, -1, :].argmax(dim=-1)   # (1,)
                    logits_last = out[:, -1, :]                      # (1,O)
                    loss = criterion(logits_last, target_idx)
                else:
                    logits = out.reshape(-1, self.O)                 # (T,O)
                    target_idx = y_tensor.view(-1, self.O).argmax(dim=-1)  # (T,)
                    loss = criterion(logits, target_idx)

                loss.backward()
                nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), self.clip_value)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(X))
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # ---- live update ----
            epochs_x = list(range(1, len(losses) + 1))
            line.set_data(epochs_x, losses)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        # finalize
        plt.ioff()
        plt.show()



    def predict(self, x):
        self.rnn.eval()
        self.fc.eval()
        with self.torch.no_grad():
            x_tensor = self.torch.tensor(x, dtype=self.torch.float32).unsqueeze(0).to(DEVICE)
            out, _ = self.rnn(x_tensor)
            out = self.fc(out)
        return out.squeeze(0).cpu().numpy()
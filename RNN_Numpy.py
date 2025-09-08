import numpy as np
import matplotlib.pyplot as plt

class RNN_Numpy:
    def __init__(self, input_size, hidden_size, output_size, cell="lstm", clip_value=5.0):
        self.I, self.H, self.O = input_size, hidden_size, output_size
        self.cell = cell
        self.clip_value = clip_value

        self.W, self.b = {}, {}

        if self.cell == "lstm":
            self.layer = self.LSTM_unit
            self.W["w_f"] = np.random.randn(self.H, self.H + self.I) * 0.01
            self.W["w_i"] = np.random.randn(self.H, self.H + self.I) * 0.01
            self.W["w_o"] = np.random.randn(self.H, self.H + self.I) * 0.01
            self.W["w_c"] = np.random.randn(self.H, self.H + self.I) * 0.01
            self.b["b_f"] = np.zeros((self.H, 1))
            self.b["b_i"] = np.zeros((self.H, 1))
            self.b["b_o"] = np.zeros((self.H, 1))
            self.b["b_c"] = np.zeros((self.H, 1))
        else:
            self.layer = self.Vanilla_unit
            self.W["W_hh"] = np.random.randn(self.H, self.H) * 0.01
            self.W["W_ih"] = np.random.randn(self.H, self.I) * 0.01
            self.b["b_h"]  = np.zeros((self.H, 1))

        # readout
        self.W["W_hy"] = np.random.randn(self.O, self.H) * 0.01
        self.b["b_y"]  = np.zeros((self.O, 1))

        self.h0 = np.zeros((self.H,1))
        self.c0 = np.zeros((self.H,1)) if self.cell == "lstm" else None

    # ---------- helpers (class-private) ----------
    @staticmethod
    def _sigmoid(x): 
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x): 
        return np.tanh(x)

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=0, keepdims=True)
        ez = np.exp(z)
        return ez / (np.sum(ez, axis=0, keepdims=True) + 1e-12)

    @staticmethod
    def _clip_by_global_norm(grad_dicts, max_norm=5.0):
        arrays = [g for d in grad_dicts for g in d.values()]
        norm = np.sqrt(sum(float(np.sum(a*a)) for a in arrays)) + 1e-12
        if norm > max_norm:
            scale = max_norm / norm
            for d in grad_dicts:
                for k in d:
                    d[k] *= scale

    # ---------- core units ----------
    def Vanilla_unit(self, x_t, h_prev, c_prev=None):
        a_t = self.W["W_ih"] @ x_t + self.W["W_hh"] @ h_prev + self.b["b_h"]
        h_t = self._tanh(a_t)
        return h_t, a_t, None, None

    def LSTM_unit(self, x_t, h_prev, c_prev):
        concat = np.vstack((h_prev, x_t))
        f = self._sigmoid(self.W["w_f"] @ concat + self.b["b_f"])
        i = self._sigmoid(self.W["w_i"] @ concat + self.b["b_i"])
        o = self._sigmoid(self.W["w_o"] @ concat + self.b["b_o"])
        g = self._tanh   (self.W["w_c"] @ concat + self.b["b_c"])
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, None, c, (concat, f, i, o, g, c_prev, c)

    # ---------- forward ----------
    def forward(self, x_seq):
        T = x_seq.shape[0]
        self.cache_core, self.cache_h = [], []
        h_prev = self.h0.copy()
        c_prev = (self.c0.copy() if self.cell=="lstm" else None)

        h_seq = np.zeros((T, self.H, 1))
        logits = np.zeros((T, self.O, 1))

        for t in range(T):
            x_t = x_seq[t]
            h_t, a_t, c_t, cache = self.layer(x_t, h_prev, c_prev)
            if self.cell == "lstm":
                self.cache_core.append(("lstm", (x_t, h_prev, c_prev, h_t, c_t, cache)))
                h_prev, c_prev = h_t, c_t
            else:
                self.cache_core.append(("vanilla", (x_t, h_prev, a_t, h_t)))
                h_prev = h_t

            h_seq[t] = h_t
            self.cache_h.append(h_t.copy())
            logits[t] = self.W["W_hy"] @ h_t + self.b["b_y"]

        return logits, h_seq

    # ---------- losses ----------
    @staticmethod
    def ce_loss_masked(logits, y, mask=None):
        T = logits.shape[0]
        loss, denom = 0.0, 0
        for t in range(T):
            if mask is None or mask[t]:
                # use the class's softmax (static), but keep signature minimal
                p = RNN_Numpy._softmax(logits[t])
                loss -= float(np.sum(y[t] * np.log(p + 1e-12)))
                denom += 1
        return loss / max(1, denom)

    @staticmethod
    def mse_loss(y_pred, y_true):
        return float(np.mean((y_pred - y_true)**2))

    # ---------- backward through time ----------
    def backward_core(self, dH):
        T = dH.shape[1]
        if self.cell == "vanilla":
            dW_hh = np.zeros_like(self.W["W_hh"])
            dW_ih = np.zeros_like(self.W["W_ih"])
            db_h  = np.zeros_like(self.b["b_h"])
            dx    = np.zeros((self.I, T))
            dh_next = np.zeros((self.H,1))
            for t in reversed(range(T)):
                _, (x_t, h_prev, a_t, h_t) = self.cache_core[t]
                dh = dH[:, [t]] + dh_next
                da = dh * (1 - np.tanh(a_t)**2)
                dW_ih += da @ x_t.T
                dW_hh += da @ h_prev.T
                db_h  += da
                dh_next = self.W["W_hh"].T @ da
                dx[:, [t]] = self.W["W_ih"].T @ da
            return {"W_hh": dW_hh, "W_ih": dW_ih, "b_h": db_h}, dh_next, dx
        else:
            w_f,w_i,w_o,w_c = self.W["w_f"], self.W["w_i"], self.W["w_o"], self.W["w_c"]
            dw_f,dw_i,dw_o,dw_c = np.zeros_like(w_f), np.zeros_like(w_i), np.zeros_like(w_o), np.zeros_like(w_c)
            db_f,db_i,db_o,db_c = np.zeros_like(self.b["b_f"]), np.zeros_like(self.b["b_i"]), np.zeros_like(self.b["b_o"]), np.zeros_like(self.b["b_c"])

            dx = np.zeros((self.I, T))
            dh_next = np.zeros((self.H,1))
            dc_next = np.zeros((self.H,1))

            for t in reversed(range(T)):
                _, (x_t, h_prev, c_prev, h_t, c_t, cache) = self.cache_core[t]
                concat, f, i, o, g, c_prev_cached, c_t_cached = cache

                dh = dH[:, [t]] + dh_next
                tanh_c = np.tanh(c_t)
                do_pre = (dh * tanh_c) * (o * (1 - o))
                dc = dh * o * (1 - tanh_c**2) + dc_next
                df_pre = (dc * c_prev) * (f * (1 - f))
                di_pre = (dc * g)      * (i * (1 - i))
                dg_pre = (dc * i)      * (1 - g**2)

                dw_f += df_pre @ concat.T
                dw_i += di_pre @ concat.T
                dw_o += do_pre @ concat.T
                dw_c += dg_pre @ concat.T
                db_f += df_pre
                db_i += di_pre
                db_o += do_pre
                db_c += dg_pre

                dconcat = (w_f.T @ df_pre) + (w_i.T @ di_pre) + (w_o.T @ do_pre) + (w_c.T @ dg_pre)
                dh_prev = dconcat[:self.H, :]
                dx[:, [t]] = dconcat[self.H:, :]

                dh_next = dh_prev
                dc_next = dc * f

            return {"w_f": dw_f, "w_i": dw_i, "w_o": dw_o, "w_c": dw_c,
                    "b_f": db_f, "b_i": db_i, "b_o": db_o, "b_c": db_c}, dh_next, dc_next, dx

    # ---------- train loop ----------
    def train(self, X, Y, epochs=100, lr=0.01):
        """
        X: list of input sequences (each (T,I,1))
        Y: list of targets (classification: (T,O,1) with labels on selected steps, e.g. last; regression: (O,1))
        """

        # ---- live plot setup ----
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Training Loss (live)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        line, = ax.plot([], [], marker="o")  # start empty; no explicit color
        losses = []

        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for x_seq, y_target in zip(X, Y):
                logits, h_seq = self.forward(x_seq)

                # ---- classification branch (masked to where targets exist) ----
                if y_target.shape[0] == logits.shape[0]:
                    mask = np.array([np.any(y_target[t] != 0.0) for t in range(y_target.shape[0])], dtype=bool)
                    loss = RNN_Numpy.ce_loss_masked(logits, y_target, mask=mask)

                    dW_hy = np.zeros_like(self.W["W_hy"])
                    db_y  = np.zeros_like(self.b["b_y"])
                    dH    = np.zeros((self.H, x_seq.shape[0]))

                    for t in range(x_seq.shape[0]):
                        if mask[t]:
                            p  = RNN_Numpy._softmax(logits[t])
                            dY = p - y_target[t]  # (O,1)
                            dW_hy += dY @ self.cache_h[t].T
                            db_y  += dY
                            dH[:, [t]] += self.W["W_hy"].T @ dY
                            
                # ---- regression branch (use last step) ----
                else:
                    y_pred = logits[-1]
                    loss   = RNN_Numpy.mse_loss(y_pred, y_target)
                    dy     = 2 * (y_pred - y_target) / y_target.size
                    dW_hy  = dy @ self.cache_h[-1].T
                    db_y   = dy.copy()
                    dH     = np.zeros((self.H, x_seq.shape[0]))
                    dH[:, [-1]] = self.W["W_hy"].T @ dy

                # backward through core
                if self.cell == "lstm":
                    core_grads, _, _, _ = self.backward_core(dH)
                else:
                    core_grads, _, _ = self.backward_core(dH)

                readout = {"W_hy": dW_hy, "b_y": db_y}
                self._clip_by_global_norm([core_grads, readout], self.clip_value)

                # SGD update
                for k, g in core_grads.items():
                    if k in self.W: self.W[k] -= lr * g
                    if k in self.b: self.b[k] -= lr * g

                self.W["W_hy"] -= lr * dW_hy
                self.b["b_y"]  -= lr * db_y

                total_loss += loss

            avg_loss = total_loss/len(X)
            losses.append(avg_loss)
            
            print(f"Epoch {ep}/{epochs} loss={avg_loss}")

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

    # ---------- inference helpers ----------
    def _warmup_context(self, x_seq):
        """
        Run through provided tokens to set (h,c) before sampling.
        Returns the final (h_prev, c_prev).
        """
        T = x_seq.shape[0]
        h_prev = self.h0.copy()
        c_prev = self.c0.copy() if self.cell == "lstm" else None
        for t in range(T):
            if np.any(x_seq[t] != 0):
                if self.cell == "lstm":
                    h_prev, _, c_prev, _ = self.LSTM_unit(x_seq[t], h_prev, c_prev)
                else:
                    h_prev, _, _, _ = self.Vanilla_unit(x_seq[t], h_prev)
        return h_prev, c_prev

    # ---------- predict ----------
    def predict(self, x_seq, steps=1, temperature=1.0):
        """
        If classification: use provided context, then autoregress for `steps`.
        If regression: rolling forecast.
        """
        if self.O > 1:  # classification
            h_prev, c_prev = self._warmup_context(x_seq)
            seq = []
            x_t = x_seq[-1].copy()
            for _ in range(steps):
                if self.cell == "lstm":
                    h_prev, _, c_prev, _ = self.LSTM_unit(x_t, h_prev, c_prev)
                else:
                    h_prev, _, _, _ = self.Vanilla_unit(x_t, h_prev)
                logits = self.W["W_hy"] @ h_prev + self.b["b_y"]
                z = logits / max(1e-6, temperature)
                p = self._softmax(z).ravel()
                idx = np.random.choice(len(p), p=p/np.sum(p))
                seq.append(idx)
                x_t = np.zeros_like(x_t)
                x_t[idx, 0] = 1
            return seq
        else:  # regression
            window = x_seq.squeeze(-1)  # (T, I)
            preds = []
            for _ in range(steps):
                y_pred, _ = self.forward(window.reshape(window.shape[0], self.I, 1))
                y = y_pred[-1].reshape(-1)
                preds.append(y.copy())
                window = np.vstack([window[1:], y.reshape(1, -1)])
            return np.array(preds)




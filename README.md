# RNN_implementation

This project provides a minimal character-level RNN for text prediction, with two interchangeable backends:

- **NumPy** (`RNN_Numpy`): from-scratch implementation with **Vanilla RNN** and **LSTM** cores, manual BPTT, gradient clipping, and a simple readout layer.
- **PyTorch** (`RNN_Pytorch`): uses `nn.RNN` / `nn.GRU` / `nn.LSTM` plus a linear readout layer, with live loss plotting and temperature-based sampling.

The driver trains on a built-in toy corpus and then generates text autoregressively.

---

## Features

- **Backends**: `numpy` or `pytorch` (select with `--model`)
- **Cells**:
  - NumPy: `vanilla`, `lstm`
  - PyTorch: `vanilla` (`nn.RNN`), `gru`, `lstm`
- **Character-level modeling** with one-hot inputs
- **Final-step supervision** (predicts the next character from the last timestep)
- **Temperature sampling** for diverse generation
- **Gradient clipping** (global norm for NumPy; `clip_grad_norm_` for PyTorch)
- **Live training-loss plotting** with Matplotlib

---

## File Layout

├─ rnn_driver_simple.py # CLI entrypoint: builds dataset, trains, and generates text
├─ RNN_Numpy.py # NumPy backend (Vanilla/LSTM, manual BPTT)
└─ RNN_Pytorch.py # PyTorch backend (RNN/GRU/LSTM + Linear head)


If you keep everything in a single file, the classes are already included at the bottom of `rnn_driver_simple.py`.

---

## Installation

```bash
# Python 3.9+
pip install numpy matplotlib

# Optional (required for --model pytorch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or install CUDA build from https://pytorch.org/get-started/locally/ if you have a GPU
```

---

## Usage

Train with PyTorch LSTM (default):

```bash
python rnn_driver_simple.py
```

Explicitly select NumPy LSTM:

```bash
python rnn_driver_simple.py --model numpy --cell lstm --hidden 64 --epochs 50 --seq_len 12 --l_rate 0.01 --temperature 0.8 --gen_steps 200
```

Provide a custom prompt for generation:

```bash
python rnn_driver_simple.py --prompt "In the lab at midnight,"
```

---

## Example Output

```bash
=== Generation ===
Seed: 'hello, model—'
Out : llo, model—data dribbled into HDF5, neatly tuned the Keithley again...
```



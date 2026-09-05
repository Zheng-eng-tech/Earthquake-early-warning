# Adaptive-Gated Low-Latency Earthquake Early Warning Based on a Single Station

## MATLAB–Python Environment

The MATLAB main program uses **Python 3.10.11 (64-bit, Windows)** through MATLAB's `InProcess` execution mode.

The following package versions were recorded from the configured Python environment. They are provided as an environment reference.

| Package                | Version     |
| ---------------------- | ----------- |
| NumPy                  | 2.2.6       |
| SciPy                  | 1.15.3      |
| PyTorch                | 2.4.1+cu121 |
| scikit-learn           | 1.7.2       |
| ObsPy                  | 1.5.0       |
| Matplotlib             | 3.10.9      |
| XGBoost                | 3.2.0       |
| python-speech-features | 0.6         |
| PyGeodesy              | 26.6.12     |

Before running the MATLAB main program, configure MATLAB to use the appropriate local Python executable:

```matlab
pyenv('Version', 'C:\path\to\Python310\python.exe', ...
      'ExecutionMode', 'InProcess');
```

Replace the example path with the location of your local Python installation.

## TinyPhase-GRU Model

The newly trained TinyPhase-GRU model is included in:

```text
Python/tinyphase_gru_10s_p100_900/best.py
```

## External E3WS Model Files

The reference E3WS source-parameter estimation model files are not included in this repository because of their file sizes. Before running the complete workflow, download the required model files from the upstream E3WS repository and place them in the paths expected by the code:

https://github.com/PabloELara/E3WS

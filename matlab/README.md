MATLAB code for the earthquake early warning system.
# MATLAB–Python Environment

The original MATLAB main program uses **Python 3.10.11 (64-bit, Windows)** with MATLAB's `InProcess` execution mode.

The following package versions were recorded from the configured Python environment. They are provided as an environment reference, rather than a guarantee of compatibility with every MATLAB release.

| Package | Version |
| --- | --- |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| PyTorch | 2.4.1+cu121 |
| scikit-learn | 1.7.2 |
| ObsPy | 1.5.0 |
| Matplotlib | 3.10.9 |
| XGBoost | 3.2.0 |
| python-speech-features | 0.6 |
| PyGeodesy | 26.6.12 |

Configure MATLAB to use the appropriate Python executable on your machine before running the main program:

```matlab
pyenv('Version', 'C:\path\to\Python310\python.exe', ...
      'ExecutionMode', 'InProcess');
```

Replace the example path with your local Python installation path.

## Tinyphase-GRU Model

The training code and the newly trained **Tinyphase-GRU** model are included in the `Python` folder.

 

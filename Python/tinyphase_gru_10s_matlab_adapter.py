from pathlib import Path
import sys

import numpy as np
import torch
from scipy import signal


_MODEL = None
_DEVICE = None
_SOS = None
_FS = 100.0
_SAMPLES = 1000
_EPS = 1.0e-6


def load_model(checkpoint_path, use_cuda=True):
    """Load and warm up the frozen 10-second TinyPhaseGRU."""
    global _MODEL, _DEVICE, _SOS

    checkpoint_path = Path(str(checkpoint_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))

    project_src = checkpoint_path.parents[2] / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from tinyphase_gru import TinyPhaseGRU

    _DEVICE = torch.device(
        "cuda" if bool(use_cuda) and torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        str(checkpoint_path), map_location=_DEVICE, weights_only=True
    )
    preprocessing = checkpoint["preprocessing"]
    if int(preprocessing["window_samples"]) != _SAMPLES:
        raise RuntimeError("Checkpoint window length is not 1000 samples")
    if not np.isclose(float(preprocessing["sampling_rate_hz"]), _FS):
        raise RuntimeError("Checkpoint sampling rate is not 100 Hz")

    _MODEL = TinyPhaseGRU(**checkpoint["model_config"])
    _MODEL.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _MODEL.to(_DEVICE).eval()
    _SOS = signal.butter(
        4, [1.0, 7.0], btype="bandpass", fs=_FS, output="sos"
    )

    with torch.inference_mode():
        _MODEL(torch.zeros((1, _SAMPLES, 3), dtype=torch.float32,
                           device=_DEVICE))
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()

    epoch = checkpoint.get("training_state", {}).get("epoch", "unknown")
    return "10-s TinyPhaseGRU loaded: device={}, epoch={}".format(
        _DEVICE, epoch
    )


def pick(e_data, n_data, z_data, sampling_rate):
    """Return 1000 temporal-softmax probabilities for the latest samples."""
    if _MODEL is None:
        raise RuntimeError("Call load_model() before pick()")
    if not np.isclose(float(sampling_rate), _FS):
        raise RuntimeError("TinyPhaseGRU requires 100-Hz input")

    columns = [np.asarray(value, dtype=np.float64).reshape(-1)
               for value in (e_data, n_data, z_data)]
    if len({column.size for column in columns}) != 1:
        raise RuntimeError("E/N/Z input lengths differ")
    if columns[0].size < _SAMPLES:
        raise RuntimeError("TinyPhaseGRU requires at least 1000 samples")

    waveform = np.column_stack([column[-_SAMPLES:] for column in columns])
    if not np.isfinite(waveform).all():
        raise RuntimeError("TinyPhaseGRU input contains NaN or Inf")

    waveform = signal.detrend(waveform, axis=0, type="constant")
    waveform = signal.detrend(waveform, axis=0, type="linear")
    waveform = signal.sosfiltfilt(_SOS, waveform, axis=0)
    means = waveform.mean(axis=0, keepdims=True)
    stds = waveform.std(axis=0, keepdims=True)
    if np.any(stds < _EPS):
        raise RuntimeError("TinyPhaseGRU received a near-constant component")
    waveform = ((waveform - means) / stds).astype(np.float32)

    tensor = torch.from_numpy(waveform).unsqueeze(0).to(_DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(_MODEL(tensor), dim=1)[0]
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return probabilities.detach().cpu().numpy()

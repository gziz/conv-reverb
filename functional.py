from typing import Optional

import torch
import torchaudio
from torch import Tensor


def _get_rir_boundaries(waveform: Tensor, sample_rate: int) -> Tensor:
    """
    Get the impulse start and end boundaries from a RIR waveform
    """
    from torchaudio.transforms import Spectrogram, AmplitudeToDB
    spec_transform = Spectrogram()
    bins_ratio = spec_transform.hop_length
    s2db = AmplitudeToDB()

    freq = s2db(spec_transform(waveform))
    freq_sum = freq.sum(axis=1).flatten()
    freq_mean_noise = freq_sum[:int(0.1*sample_rate)].mean()

    peak_idx = freq_sum.argmax()
    before_peak = freq_sum[:peak_idx] <= freq_mean_noise
    after_peak = freq_sum[peak_idx:] <= freq_mean_noise

    start_idx = torch.nonzero(before_peak)[-1]
    end_idx = peak_idx + torch.nonzero(after_peak)[0]

    return start_idx * bins_ratio, end_idx * bins_ratio


def process_rir(rir: Optional[Tensor] = None, sample_rate: int = None) -> Tensor:
    r"""Process RIR file, download sample file if no RIR is provided

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        rir (Tensor, None): Room Impulse Response waveform,
        with shape `(1, time)`.
    Returns:
        Tensor: The processed RIR waveform
    """
    if rir is None:
        path = torchaudio.utils.download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
        )
        rir, sample_rate = torchaudio.load(path)

    if not sample_rate:
        raise ValueError("If RIR is provided, sample_rate must be specified")
    
    start_idx, end_idx = _get_rir_boundaries(rir, sample_rate)

    rir = rir[:, start_idx : end_idx]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir
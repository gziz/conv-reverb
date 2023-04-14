from typing import Optional

import torch
import torchaudio.functional as F
from torch import Tensor

class ConvolutionalReverb(torch.nn.Module):
    r"""Apply convolutional reverberation to a waveform using FFT
    and a Room Impulse Response (RIR) tensor.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        rir (Tensor, None): Room Impulse Response waveform.
        apply_processing_rir (bool, optional): Whether the RIR waveform should be
            processed (Default: ``True``).

    Example
        >>> waveform, sample_rate = torchaudio.load("speech.wav")
        >>> transform = torchaudio.transforms.ConvolutionalReverb()
        >>> augmented = transform(waveform)
    """

    def __init__(
            self,
            rir: Optional[Tensor] = None,
            sample_rate: int = None,
            apply_processing_rir: bool = True
    ) -> None:
        super().__init__()

        self.rir = rir
        if rir is None or apply_processing_rir:
            self.rir = F.process_rir(rir, sample_rate)

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of shape `(..., time)`.
        Returns:
            Tensor: The reverberated audio of shape `(..., time)`.
        """
        return F.fftconvolve(waveform, self.rir)

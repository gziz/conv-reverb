from typing import Optional

import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor


def get_processed_rir(rir: Optional[Tensor] = None) -> Tensor:
    if rir is None:
        path = torchaudio.utils.download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
        )
        rir, sample_rate = torchaudio.load(path)

        rir = rir[:, int(sample_rate * 1.01): int(sample_rate * 1.3)]
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, [1])
    return rir


class ConvolutionalReverb(torch.nn.Module):
    r"""Apply convolutional reverberation to a waveform using FFT
    and a Room Impulse Response (RIR) tensor.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        rir (Tensor, None): RIR sample file, with the impulse extracted.

    Example
        >>> waveform, sample_rate = torchaudio.load("speech.wav")
        >>> transform = torchaudio.transforms.ConvolutionalReverb()
        >>> augmented = transform(waveform)
    """

    def __init__(
            self,
            rir: Optional[Tensor] = None,
            process_rir: bool = True
    ) -> None:
        super().__init__()

        self.rir = rir
        if rir is None or process_rir:
            self.rir = get_processed_rir(rir)

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of shape `(..., time)`.
        Returns:
            Tensor: The reverberated audio of shape `(..., time)`.
        """
        return F.fftconvolve(waveform, self.rir)

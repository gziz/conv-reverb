import torch
import torchaudio
from torch import Tensor

class ConvReverb(torch.nn.Module):
    """
    Apply convolutional reverberation to a waveform using FFT 
    and a Room Impulse Response (RIR) tensor.
    """

    def __init__(self, rir: Tensor = None) -> None:
        super().__init__()

        self.rir = rir
        if self.rir is None:
            self.rir = self.download_rir_sample()

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of shape `(..., time)`.
        Returns:
            Tensor: The reverberated audio of shape `(..., time)`.
        """
        return torchaudio.functional.fftconvolve(waveform, self.rir)
    
    def download_rir_sample(self) -> Tensor:
        rir_path = torchaudio.utils.download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
        )
        rir_raw, sample_rate = torchaudio.load(rir_path)
        rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, [1])
        return rir

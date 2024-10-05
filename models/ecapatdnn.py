'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

References from https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
'''

import math, torch
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from typing import Any, Optional
from torchaudio.transforms import MelScale

def spec_max_random_normalization(spec: torch.Tensor, min_value=0.5, max_value=1) -> torch.Tensor:
    random_value = (max_value - min_value) * torch.rand((1)).item() + min_value
    if spec.max() == 0:
        return spec
    return spec / spec.max() * random_value

def pad_center(
    data: np.ndarray, size: int, axis: int = -1, **kwargs: Any
) -> np.ndarray:
    """Pad an array to a target length along a target axis.
    """
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            f"Target size ({size:d}) must be at least input size ({n:d})"
        )

    return np.pad(data, lengths, **kwargs)

class OnnxSTFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(OnnxSTFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class Wav2MelSpec(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 80,
        f_max: int = 7600,
        n_mels: int = 80,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        
        self.stft = OnnxSTFT(
            filter_length=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )
        
    def forward(self, wav):
        """
        Args:
            wav (torch.Tensor): (batch_size, audio_length)
        Returns:
            torch.Tensor: (batch_size, n_mels, time_index)
        """
        assert wav.dim() == 2, f"wav must be 2D tensor, but got {wav.dim()}"
        spec, _ = self.stft.transform(wav)
        mel = self.mel_scale(spec)
        mel = torch.log(torch.clamp_min(mel, min=1e-5))
        return mel

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 



    
class ECAPA_TDNN(nn.Module):

    def __init__(
        self, 
        frequency_bins_num=80,
        channel_size=1024,
        hidden_size=192
    ):
        """
        Args:
            channel_size (int): channel size. Defaults to 1000.
            hidden_size (int): output hidden size. Defaults to 64.
        """

        super(ECAPA_TDNN, self).__init__()
        self.conv1  = nn.Conv1d(frequency_bins_num, channel_size, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(channel_size)
        self.layer1 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*channel_size, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)


    def forward(self, x) -> torch.Tensor:
        """音声から特徴抽出 (time_indexは、可変でOK)
        Args:
            x (torch.Tensor): メルスペクトロうグラム (batch_size, n_mels, time_index)

        Returns:
            torch.Tensor: 特徴ベクトル (batch_size, hidden_size)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu,sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        return x

CHANNEL_SIZE = 1024
SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
F_MIN = 20
F_MAX = 7600
N_MELS = 80

class _SpeakerEmbeddingJa():
    def __init__(
        self,
        hidden_size: int = 192,
        
    ):
        self.model = ECAPA_TDNN(
            frequency_bins_num=N_MELS,
            channel_size=CHANNEL_SIZE,
            hidden_size=hidden_size
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.sample_rate = SAMPLE_RATE
        
        self.preprocess = Wav2MelSpec(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS
        )
        self.preprocess.eval()
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.inference_mode():
            mel = self.preprocess(x)
            return self.model(mel)

def SpeakerEmbeddingJa(model_path="ecapatdnn_ja_l512_st2_ep19.ckpt", hidden_size: int = 512) -> _SpeakerEmbeddingJa:
    model = _SpeakerEmbeddingJa(hidden_size=hidden_size)
    model.model.load_state_dict(torch.load(model_path, weights_only=True))
    model.model.eval()
    model.preprocess.eval()
    for param in model.model.parameters():
        param.requires_grad = False
    return model

if __name__ == "__main__":
    import torchaudio
    import torch.nn.functional as F
    model = SpeakerEmbeddingJa(
        model_path="data/ecapatdnn_ja_l512_st2_ep19.ckpt",
        hidden_size=512
    )
    wav, sr = torchaudio.load("data/sample.wav")
    wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    
    emb = model(wav)
    emb = F.normalize(torch.FloatTensor(emb), p=2, dim=1).detach().cpu()
    
    score = torch.matmul(emb, emb.T)
    print(score) # tensor([[1.]])
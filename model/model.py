import torch
from torch import nn
import torchaudio
import math


class SPNet2D(nn.Module):
    def __init__(self, n_targets=1, embedding_dim=512, n_mels=64, dropout=0.3, use_attention=False, frontend='dB', fs=16000):
        super(SPNet2D, self).__init__()
        self.use_attention = use_attention
        self.fs = fs
        self.batch_size = 16
        self.n_targets = n_targets
        self.embedding_dim = embedding_dim
        assert frontend in ['dB', 'persa+', 'persa', 'pcen']
        if frontend == 'dB':
            self.frontend = torchaudio.transforms.AmplitudeToDB()

        self.mel_specgram = torchaudio.transforms.MelSpectrogram(self.fs, n_fft=400, hop_length=200,
                                                                 n_mels=n_mels, normalized=False).cuda() # (channel, n_mels, time)

        self.conv1 = torch.nn.Conv2d(1, 512, kernel_size=(5, 5), padding=5//2)
        self.norm1 = torch.nn.BatchNorm2d(512)
        self.drop1 = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv2d(512, 64, kernel_size=(3, 3), padding=3//2)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.drop2 = torch.nn.Dropout(dropout)
        self.pool2 = torch.nn.MaxPool2d((2, 2))

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=3//2)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.drop3 = torch.nn.Dropout(dropout)
        self.pool3 = torch.nn.MaxPool2d((2, 2))

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=3 // 2)
        self.norm4 = torch.nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout(dropout)
        self.pool4 = torch.nn.MaxPool2d((2, 2))

        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=3 // 2)
        self.norm5 = torch.nn.BatchNorm2d(512)
        self.drop5 = torch.nn.Dropout(dropout)
        self.pool5 = torch.nn.MaxPool2d((2, 2))

        self.lin6 = nn.Linear(512, self.embedding_dim)
        self.drop6 = nn.Dropout(dropout)
        self.lin7 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop7 = nn.Dropout(dropout)
        self.linout = nn.Linear(self.embedding_dim, self.n_targets)

    def forward(self, x, x_len=None):
        if x.dim() == 1: x = x.unsqueeze(0)
        if x_len is None:
            # x_shape = jitable_shape(x)
            # x_len = [x_shape[-1]] * x_shape[0]
            x_len = [x.shape[-1]] * x.shape[0]

        x_len_padded = x.shape[-1]
        x = self.mel_specgram(x)
        x = self.frontend(x)
        x = x.unsqueeze(1)

        x = self.drop1(self.norm1(torch.relu(self.conv1(x))))
        x = self.drop2(self.pool2(self.norm2(torch.relu(self.conv2(x)))))
        x = self.drop3(self.pool3(self.norm3(torch.relu(self.conv3(x)))))
        x = self.drop4(self.pool4(self.norm4(torch.relu(self.conv4(x)))))
        x = self.drop5(self.pool5(self.norm5(torch.relu(self.conv5(x)))))

        # (batch, channels, freq, time)
        x = torch.mean(x, dim=-2)
        # (batch, channels, time)

        # take mean only the segments that are not padded
        x_per = [max(1, math.ceil(xl * x.shape[-1] / x_len_padded)) for xl in x_len]
        x = torch.cat([torch.mean(x[j, :, :x_per[j]], dim=-1).unsqueeze(0) for j in range(x.shape[0])], dim=0)
        # emb = x
        x = self.lin6(x)
        x = torch.relu(x)
        # This embedding produces the best results
        emb = x
        x = self.drop6(x)
        x = self.lin7(x)
        x = torch.relu(x)
        x = self.drop7(x)
        x = self.linout(x)

        return x, emb

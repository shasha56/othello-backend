# ニューラルネットでQ値を出す
import torch
import torch.nn as nn

# 8×8盤面用ニューラルネットワークのみ

class QNetwork(nn.Module): # nn.ModuleとはPyTorchのニューラルネットであることの宣言
    def __init__(self,size):
        super().__init__() # super()はnn.Moduleには必要

        if size == 8:
            self.fc1 = nn.Linear(64, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 64)
        else:
            self.fc1 = nn.Linear(size ** 2, (size ** 2) * 2) # fcが層のイメージ
            self.fc2 = nn.Linear((size ** 2) * 2, (size ** 2) * 2) # nn.Linear(入力,出力) 変換の過程で重みとバイアスがかかる
            self.fc3 = nn.Linear((size ** 2) * 2, size ** 2)

        self.size = size

    def forward(self, x): # forward() はニューラルネットの中をどう通っていくか

        if self.size == 8:
            x = torch.relu(self.fc1(x)) # ReLUは負の値 → 0,正の値 → そのまま(max(0, x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
        else:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

        return x
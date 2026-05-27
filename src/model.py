import torch
import torch.nn as nn

class URL1DCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, max_length=200):
        super(URL1DCNN, self).__init__()
        # 1. 임베딩 레이어: 숫자를 밀집 벡터로 변환
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. 1D CNN 레이어 (문자열의 로컬 패턴 추출)
        # nn.Conv1d는 (Batch, Channel, Length) 순서로 입력을 받음
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        
        self.pool = nn.AdaptiveMaxPool1d(1) # 길이를 1로 압축
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # 3. 전결합 레이어 (최종 이진 분류)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        # x: (Batch, Length) -> [Batch, 200]
        x = self.embedding(x) # (Batch, Length, Embedding_dim) -> [Batch, 200, 64]
        
        # Conv1d 연산을 위해 차원 축을 변경: (Batch, Embedding_dim, Length)
        x = x.permute(0, 2, 1) # [Batch, 64, 200]
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        x = self.pool(x).squeeze(-1) # [Batch, 128]
        x = self.dropout(x)
        x = self.fc(x) # [Batch, 1]
        
        return x # 시그모이드를 거치지 않은 Logit 값을 반환 (BCEWithLogitsLoss 연산 최적화용)
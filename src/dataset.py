import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class URLDataset(Dataset):
    def __init__(self, csv_path, max_length=200):
        # 1. 데이터 로드
        df = pd.read_csv(csv_path)
        
        # 2. 타겟 이진 분류 변환 (benign은 0, 나머지는 모두 1)
        df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
        
        self.urls = df['url'].values
        self.labels = df['label'].values
        self.max_length = max_length
        
        # 3. 글자 사전(Vocabulary) 빌드 (알파벳, 숫자, 특수문자 등)
        # 기본 문자와 특수문자들을 인덱스 매핑 (0은 패딩용으로 비워둠)
        all_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;="
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(all_chars)}
        self.vocab_size = len(self.char_to_idx) + 1 # 패딩 번호(0) 포함
        
    def __len__(self):
        return len(self.urls)
        
    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        
        # URL 글자를 숫자로 변환 (사전에 없는 이상한 문자는 0처리)
        numerical_url = [self.char_to_idx.get(char, 0) for char in url]
        
        # 고정된 길이(max_length)로 패딩 및 자르기
        if len(numerical_url) < self.max_length:
            # 부족하면 뒤를 0으로 채움 (Post-Padding)
            numerical_url = numerical_url + [0] * (self.max_length - len(numerical_url))
        else:
            # 넘치면 잘라냄
            numerical_url = numerical_url[:self.max_length]
            
        return torch.tensor(numerical_url, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
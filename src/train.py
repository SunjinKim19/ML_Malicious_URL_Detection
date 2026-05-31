import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from dataset import URLDataset
from model import URL1DCNN
from baseline import run_baseline
from visualize import generate_presentation_plots

# 실험의 재현성(Reproducibility)을 위한 랜덤 시드 고정 함수
def set_seed(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 학습 시작 전 시드 고정
    set_seed(42)
    
    # 하이퍼파라미터 및 경로 설정
    CSV_PATH = "data/malicious_phish.csv"
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 0.001
    
    # ----------------------------------------------------
    # 단계 1: 비교용 베이스라인 모델(머신러닝) 먼저 실행
    # ----------------------------------------------------
    baseline_metrics = None
    try:
        baseline_metrics = run_baseline(CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Baseline 실행 중 오류가 발생했습니다: {e}") from e

    # ----------------------------------------------------
    # 단계 2: 파이토치 데이터셋 세팅 및 7:1:2 분할
    # ----------------------------------------------------
    print("\n=== 데이터셋 로드 및 전처리 시작 ===")
    dataset = URLDataset(CSV_PATH, max_length=200)
    
    total_len = len(dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    
    # 데이터셋을 Train(70%), Val(10%), Test(20%)로 정교하게 분할
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 연산 장치(GPU/CPU) 세팅
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # ----------------------------------------------------
    # 단계 3: 딥러닝 모델 및 손실함수, 옵티마이저 초기화
    # ----------------------------------------------------
    model = URL1DCNN(vocab_size=dataset.vocab_size).to(device)
    
    # 데이터 불균형 문제를 해결하기 위해 악성(1) 클래스에 가중치 2.0 부여
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 히스토리 기록용 리스트
    train_losses = []
    val_losses = []
    best_val_f1 = 0.0
    
    # ----------------------------------------------------
    # 단계 4: 훈련(Training) 및 검증(Validation) 루프
    # ----------------------------------------------------
    print("\n=== 딥러닝 모델 (1D-CNN) 학습 및 오버피팅 모니터링 시작 ===")
    for epoch in range(EPOCHS):
        # --- [Train Phase] ---
        model.train()
        running_loss = 0.0
        for i, (urls, labels) in enumerate(train_loader):
            urls, labels = urls.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(urls).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # --- [Validation Phase] ---
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for urls, labels in val_loader:
                urls, labels = urls.to(device), labels.to(device)
                outputs = model(urls).squeeze(-1)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_f1 = f1_score(val_labels, val_preds)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1-Score: {val_f1:.4f}")
        
        # 최고 성능(Val F1 기준)을 달성한 모델 가중치 저장 (Model Checkpoint)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_1dcnn_model.pth")
            print("=> 최우수 모델 가중치 갱신 및 저장 완료!")
                
    # 학습 종료 후 보고서용 간단 Loss Curve 시각화 및 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_curve.png")
    plt.close()
    print("\n[알림] 기본 학습 곡선 그래프가 'plots/loss_curve.png'로 보존되었습니다.")

    # ----------------------------------------------------
    # 단계 5: 최종 테스트(Test Set) 성능 평가 및 발표 자료 생성
    # ----------------------------------------------------
    print("\n=== 최우수 가중치 로드 및 최종 Test 세트 성능 검증 ===")
    model.load_state_dict(torch.load("models/best_1dcnn_model.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for urls, labels in test_loader:
            urls = urls.to(device)
            outputs = model(urls).squeeze(-1)
            probs = torch.sigmoid(outputs) # 예측 확률값 계산
            preds = (probs > 0.5).float()   # 0.5 임계값 기준 이진 분류
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # 최종 결과 지표 산출 
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    print("\n================ 최종 실험 결과 ================")
    print(f"Deep Learning Accuracy : {accuracy:.4f}")
    print(f"Deep Learning F1-Score : {f1:.4f}")
    print(f"Deep Learning ROC-AUC  : {roc_auc:.4f}")
    print("================================================")
    
    # ----------------------------------------------------
    # 단계 6: 발표 전용 고해상도 시각화 이미지 3종 생성 호출
    # ----------------------------------------------------
    dl_metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    generate_presentation_plots(all_labels, all_preds, all_probs, baseline_metrics, dl_metrics)

if __name__ == "__main__":
    main()
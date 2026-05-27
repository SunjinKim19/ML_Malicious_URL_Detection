import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os

def generate_presentation_plots(all_labels, all_preds, all_probs, baseline_metrics, dl_metrics):
    """
    all_labels: 실제 정답 리스트
    all_preds: 딥러닝 예측값 리스트 (0 또는 1)
    all_probs: 딥러닝 예측 확률 리스트 (0 ~ 1 사이 수치)
    baseline_metrics: {'accuracy': 구조, 'f1': 구조, 'roc_auc': 구조} 형태의 베이스라인 결과
    dl_metrics: {'accuracy': 구조, 'f1': 구조, 'roc_auc': 구조} 형태의 딥러닝 결과
    """
    os.makedirs("plots", exist_ok=True)
    
    # 그래프 스타일 세팅 (깔끔한 화이트톤)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available() else 'default')
    
    # ----------------------------------------------------
    # 시각화 1: 베이스라인 vs 딥러닝 모델 성능 비교 바 차트
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    metrics_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
    
    base_vals = [baseline_metrics['accuracy'], baseline_metrics['f1'], baseline_metrics['roc_auc']]
    dl_vals = [dl_metrics['accuracy'], dl_metrics['f1'], dl_metrics['roc_auc']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline (Logistic)', color='#9ca3af')
    rects2 = ax.bar(x + width/2, dl_vals, width, label='Main Model (1D-CNN)', color='#3b82f6')
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper left')
    
    # 바 차트 위에 숫자 표시
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('plots/presentation_model_comparison.png')
    plt.close()

    # ----------------------------------------------------
    # 시각화 2: 혼동 행렬 (Confusion Matrix) 
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Matplotlib으로 깔끔하게 그리는 Confusion Matrix
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)
    
    ax.set_title('Confusion Matrix (1D-CNN)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign (0)', 'Malicious (1)'], fontsize=11)
    ax.set_yticklabels(['Benign (0)', 'Malicious (1)'], fontsize=11)
    
    # 각 칸에 수치 써주기
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va='center', ha='center', size='large', fontweight='bold',
                    color="white" if cm[i, j] > cm.max()/2 else "black")
            
    plt.tight_layout()
    plt.savefig('plots/presentation_confusion_matrix.png')
    plt.close()

    # ----------------------------------------------------
    # 시각화 3: ROC 커브 (ROC Curve)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#10b981', lw=3, label=f'1D-CNN (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#ef4444', lw=2, linestyle='--', label='Random Guess')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plt.savefig('plots/presentation_roc_curve.png')
    plt.close()
    
    print("\n[성공] 발표용 시각화 이미지 3종이 'plots/' 폴더에 저장되었습니다!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

def run_baseline(csv_path):
    print("=== 베이스라인 모델 (로지스틱 회귀) 학습 시작 ===")
    df = pd.read_csv(csv_path)
    df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        df['url'].values, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'].values
    )
    
    # URL 텍스트를 TF-IDF 수치 벡터로 변환
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 모델 학습
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # 예측 및 평가
    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)[:, 1]
    
    print(f"Baseline Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Baseline F1-Score: {f1_score(y_test, preds):.4f}")
    print(f"Baseline ROC-AUC : {roc_auc_score(y_test, probs):.4f}")
    print("===============================================\n")

    return {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs)
    }
# ML_Malicious_URL_Detection

머신러닝 텀 프로젝트로 진행한 **악성 URL 탐지 프로젝트**이다.
URL 문자열을 입력으로 받아 해당 URL이 정상 URL인지 악성 URL인지 분류하는 이진 분류 모델을 구현하였다.

기존의 블랙리스트 기반 방식은 이미 알려진 URL만 탐지할 수 있다는 한계가 있기 때문에, 본 프로젝트에서는 URL 문자열 자체의 패턴을 학습하는 방식으로 접근하였다.

## 1. 프로젝트 주제

본 프로젝트의 목표는 다음과 같다.

> URL 텍스트만을 이용해 정상 URL과 악성 URL을 자동으로 분류하기

악성 URL은 피싱, 악성코드 유포, 계정 탈취 등에 사용될 수 있기 때문에 빠르게 탐지하는 것이 중요하다.
특히 새롭게 생성되는 URL은 기존 블랙리스트에 등록되어 있지 않을 수 있으므로, 문자열 패턴을 기반으로 판단하는 머신러닝 모델이 필요하다고 판단하였다.

## 2. 사용 데이터셋

Kaggle에 공개된 **Malicious URLs Dataset**을 사용하였다.

데이터셋은 용량 및 라이선스 문제를 고려하여 GitHub 저장소에 직접 포함하지 않았다.  
코드를 실행하려면 아래 링크에서 데이터셋을 직접 다운로드한 뒤 이 폴더에 저장해야 한다.

* 데이터 수: 651,191개
* 정상 URL: 428,103개
* 악성 URL: 223,088개
* 정상/악성 비율: 약 65.7% / 34.3%

데이터셋 링크:

```text
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
```

데이터셋은 용량 및 라이선스 문제를 고려해 저장소에 직접 포함하지 않았다.
실행하려면 Kaggle에서 데이터를 다운로드한 뒤 `data/` 폴더에 넣어야 한다.

```text
data/
└── malicious_phish.csv
```

## 3. 프로젝트 구조

```text
ML_Malicious_URL_Detection/
├── data/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

`src/` 폴더에는 전처리, 모델 학습, 평가에 사용한 코드가 들어 있다.

## 4. 전처리 방법

URL은 일반 문장처럼 띄어쓰기가 있는 텍스트가 아니고, 알파벳, 숫자, 특수문자가 섞여 있다.
따라서 단어 단위가 아니라 **문자 단위(character-level)** 로 URL을 나누어 처리하였다.

전처리 과정은 다음과 같다.

1. URL 문자열을 문자 단위로 분리
2. 각 문자를 숫자 인덱스로 변환
3. URL 길이를 최대 200자로 맞춤
4. 짧은 URL은 padding 처리
5. 긴 URL은 200자 이후를 잘라냄
6. Train / Validation / Test 데이터로 분할

## 5. 사용한 모델

이번 프로젝트에서는 두 가지 모델을 비교하였다.

### 5.1 Baseline Model

기준 모델로는 다음 방식을 사용하였다.

* TF-IDF
* Logistic Regression

이 모델은 구현이 간단하고 빠르기 때문에 baseline으로 사용하였다.

### 5.2 Proposed Model

제안 모델로는 **Character-level 1D-CNN**을 사용하였다.

전체 흐름은 다음과 같다.

```text
URL 입력
→ 문자 단위 토큰화
→ Embedding Layer
→ 1D Convolution Layer
→ Max Pooling
→ Fully Connected Layer
→ 정상 / 악성 분류
```

1D-CNN은 URL 안에 있는 짧은 문자열 패턴을 학습할 수 있기 때문에, 악성 URL에서 자주 나타나는 특수문자 조합이나 이상한 문자열 구조를 잡아내는 데 적합하다고 판단하였다.

## 6. 클래스 불균형 처리

데이터셋에서 정상 URL이 악성 URL보다 더 많았기 때문에 클래스 불균형 문제가 있었다.

이를 완화하기 위해 PyTorch의 `BCEWithLogitsLoss`에서 `pos_weight`를 사용하였다.

```text
pos_weight = 2.0
```

악성 URL을 잘못 분류했을 때 더 큰 penalty를 주도록 해서, 모델이 악성 URL을 더 민감하게 탐지하도록 설정하였다.

## 7. 실험 결과

| Model                        | Accuracy | F1 Score | ROC-AUC |
| ---------------------------- | -------: | -------: | ------: |
| Logistic Regression + TF-IDF |   0.9611 |   0.9423 |  0.9908 |
| Character-level 1D-CNN       |   0.9820 |   0.9738 |  0.9975 |

실험 결과 Character-level 1D-CNN 모델이 baseline 모델보다 모든 지표에서 더 좋은 성능을 보였다.
특히 F1 Score가 향상된 점을 통해, 불균형 데이터 상황에서도 악성 URL 탐지 성능이 개선되었음을 확인할 수 있었다.

## 8. 실행 방법

필요한 패키지를 설치한다.

```bash
pip install -r requirements.txt
```

데이터셋을 Kaggle에서 다운로드한 뒤 `data/` 폴더에 넣는다.

그 다음 학습 코드를 실행한다.

```bash
python src/train.py
```

코드가 여러 파일로 나누어져 있다면 아래와 같은 순서로 실행하면 된다.

```bash
python src/preprocess.py
python src/train_baseline.py
python src/train_cnn.py
python src/evaluate.py
```

실제 파일명이 다를 경우, `src/` 폴더 안의 파일명에 맞게 명령어를 수정해야 한다.

## 9. 한계점

이번 프로젝트에서 확인한 한계는 다음과 같다.

* URL 단축 서비스를 사용하는 경우 탐지가 어려울 수 있음
* 정상 URL이지만 인증 토큰이나 긴 파라미터가 포함된 경우 악성으로 오탐할 가능성이 있음
* 비슷한 철자를 사용하는 typosquatting 공격은 더 정교한 모델이 필요할 수 있음

향후에는 Transformer 기반 모델을 적용하거나, URL뿐만 아니라 HTML 코드와 JavaScript까지 함께 분석하는 방식으로 확장해볼 수 있을 것 같다.

## 10. AI 사용 여부

초기 코드 작성과 보고서 문장 정리 과정에서 생성형 AI 도구를 일부 활용하였다.
다만 생성된 코드는 직접 검토하고 수정하였으며, 실행 결과와 최종 분석은 직접 확인하였다.

## 11. 참고 자료

* Kaggle Malicious URLs Dataset
  https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

* Zhang, X., Zhao, J., & LeCun, Y. (2015).
  Character-level convolutional networks for text classification.

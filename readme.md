#  AutoIntMLPModel (AutoInt+)

**AutoIntMLPModel**은 기존 AutoInt (Automatic Feature Interaction Learning) 모델에  
**MLP (Multi-Layer Perceptron)** 구조를 결합한 확장형 추천 시스템 모델입니다.  
기존 AutoInt가 *피처 간의 상호작용(feature interaction)* 을 Attention 메커니즘으로 학습하는 데 초점을 맞췄다면,  
**AutoIntMLP**는 여기에 **비선형 관계를 학습하는 MLP**를 추가하여 표현력을 강화했습니다.

---

## 모델 구성

AutoIntMLP는 크게 세 부분으로 구성됩니다.

###  Feature Embedding
- 각 범주형 feature(예: `user_id`, `movie_id`, `genre_id`)를 고정된 크기의 벡터로 변환합니다.
- `FeaturesEmbedding` 클래스를 사용하여 `(num_features, embedding_size)` 형태로 임베딩합니다.

###  Multi-Head Self-Attention (AutoInt)
- `MultiHeadSelfAttention`을 사용해 **특성 간 상호작용(feature interactions)** 을 학습합니다.
- `att_layer_num`개의 Attention Layer를 쌓아서 더 복잡한 관계를 모델링합니다.
- Residual Connection(`att_res=True`)으로 학습 안정성을 높입니다.

###  MLP (Multi-Layer Perceptron)
- Attention 출력(`att_output`)과 Flatten된 임베딩(`dnn_embed`)을 각각 처리합니다.
- `dnn_hidden_units`, `dnn_dropout`, `dnn_activation` 등의 설정을 통해 비선형 패턴을 학습합니다.
- 두 출력을 더해(`att_output + dnn_output`) sigmoid 활성화를 적용해 **최종 확률**을 산출합니다.

---

## 주요 클래스 관계

| 클래스 | 역할 |
|--------|------|
| `FeaturesEmbedding` | 범주형 feature를 embedding vector로 변환 |
| `MultiHeadSelfAttention` | feature 간 상호작용 학습 |
| `MultiLayerPerceptron` | 임베딩된 feature의 비선형 조합 학습 |
| `AutoIntMLP` | Attention + MLP를 결합한 layer |
| `AutoIntMLPModel` | 전체 구조를 감싸는 Keras 기반 모델 클래스 |

---

## 하이퍼파라미터 설명

| 파라미터 | 설명 | 기본값 |
|-----------|------|---------|
| `field_dims` | 각 feature의 카테고리 개수 리스트 | – |
| `embedding_size` | 임베딩 벡터 크기 | 16 |
| `att_layer_num` | Attention Layer 개수 | 3 |
| `att_head_num` | Multi-Head Attention 헤드 수 | 2 |
| `att_res` | Residual connection 사용 여부 | True |
| `dnn_hidden_units` | MLP 은닉층 구조 | (32, 32) |
| `dnn_activation` | MLP 활성화 함수 | relu |
| `dnn_dropout` | Dropout 비율 | 0.4 |
| `dnn_use_bn` | Batch Normalization 사용 여부 | False |
| `l2_reg_dnn` | DNN L2 정규화 강도 | 0 |
| `l2_reg_embedding` | 임베딩 L2 정규화 강도 | 1e-5 |
| `init_std` | 가중치 초기화 표준편차 | 0.0001 |

---

## 모델 입출력

- **입력 (`inputs`)**  
  정수형 인덱스 벡터 `[batch_size, num_fields]`  
  (예: 사용자 ID, 영화 ID, 장르 ID 등)

- **출력 (`y_pred`)**  
  각 샘플에 대한 예측 확률 `[batch_size, 1]`  
  → 사용자가 특정 아이템을 선호할 확률

---

## 사용 방법
streamlit run show_st.py
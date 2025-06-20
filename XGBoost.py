import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================
# 0. Set‐Transformer 관련 클래스 정의
# ========================================
class FeatureEmbedding(nn.Module):
    """
    각 피처 이름과 값을 받아서:
      1. name_embedding: feature_name → ℝ^{d_name}
      2. value_projection: feature_value → ℝ^{d_val}
    두 벡터를 concat하여 “컬럼 토큰” e_feature ∈ ℝ^D 생성
    """
    def __init__(self, all_feature_names: list, d_name=16, d_val=16):
        super().__init__()
        self.d_name = d_name
        self.d_val = d_val
        self.D = d_name + d_val

        # 1) name embedding table: 고정된 “합집합” feature_names를 사전(dictionary)으로 관리
        self.name2idx = {name: idx for idx, name in enumerate(all_feature_names)}
        self.name_embed = nn.Embedding(len(all_feature_names), d_name)

        # 2) value projection: 1-d → d_val-dim 벡터
        self.value_proj = nn.Linear(1, d_val)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(self, feature_names: list, feature_values: torch.Tensor):
        """
        feature_names: Python list of strings, 길이 = n_features
        feature_values: Tensor, shape = (batch_size, n_features)
                        (여기서는 batch_size=1일 수도 있음)
        returns: Tensor, shape = (batch_size, n_features, D)
        """
        device = feature_values.device
        n_features = len(feature_names)

        # 1) name 임베딩
        name_indices = torch.tensor(
            [self.name2idx[n] for n in feature_names],
            dtype=torch.long,
            device=device
        )
        # name_embed: shape = (n_features, d_name)
        name_emb = self.name_embed(name_indices)  # (n_features, d_name)

        # 2) value projection: feature_values: (batch_size, n_features) → unsqueeze(-1) → (batch_size, n_features, 1)
        val = feature_values.unsqueeze(-1).float()  # (batch_size, n_features, 1)
        val_proj = self.value_proj(val)             # (batch_size, n_features, d_val)

        # 3) name_emb을 batch 차원에 맞춰 확장
        name_emb_batch = name_emb.unsqueeze(0).expand(
            feature_values.size(0), -1, -1
        )  # (batch_size, n_features, d_name)

        # 4) concat → token_embeddings: (batch_size, n_features, D)
        token_emb = torch.cat([name_emb_batch, val_proj], dim=-1)  # (batch_size, n_features, d_name+d_val)

        return token_emb  # (batch_size, n_features, D)


class SetTransformerEncoder(nn.Module):
    """
    간단한 Self‐Attention 기반 모듈:
      - 입력: (batch_size, n_features, D)
      - MultiheadAttention → residual + layer‐norm → feed‐forward → residual + layer‐norm
      - 출력: (batch_size, D) 로 평균 풀링
    """
    def __init__(self, D, n_heads=4, dim_ff=64, dropout=0.1):
        super().__init__()
        self.D = D
        self.n_heads = n_heads

        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=n_heads,
                                          batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(D)
        self.ff = nn.Sequential(
            nn.Linear(D, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, D),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(D)

    def forward(self, x, mask=None):
        """
        x: (batch_size, n_features, D)
        mask: (batch_size, n_features) boolean mask (1: valid, 0: pad)
        returns: (batch_size, D) pooled embedding
        """
        # 1) Self-Attention
        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=(~mask) if mask is not None else None
        )
        x2 = self.ln1(x + attn_out)  # residual + layernorm

        # 2) Feed‐forward
        ff_out = self.ff(x2)
        x3 = self.ln2(x2 + ff_out)   # residual + layernorm

        # 3) Pooling: “각 feature 토큰들의 평균”
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (batch_size, n_features, 1)
            x3 = x3 * mask_f
            sum_x3 = x3.sum(dim=1)               # (batch_size, D)
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # (batch_size, 1)
            pooled = sum_x3 / denom
        else:
            pooled = x3.mean(dim=1)  # (batch_size, D)

        return pooled  # (batch_size, D)


class TabularSetEmbedder(nn.Module):
    """
    Set‐Transformer 기반 Tabular Embedding:
      - all_feature_names: 고정된 “합집합” feature 컬럼 리스트
      - forward: feature_values (batch_size, n_features) → h (batch_size, D)
    """
    def __init__(self, all_feature_names: list, d_name=16, d_val=16,
                 n_heads=4, dim_ff=64, dropout=0.1):
        super().__init__()
        self.embedder = FeatureEmbedding(all_feature_names,
                                         d_name=d_name,
                                         d_val=d_val)
        D = d_name + d_val
        self.encoder = SetTransformerEncoder(D,
                                             n_heads=n_heads,
                                             dim_ff=dim_ff,
                                             dropout=dropout)
        self.out_dim = D
        self.all_feature_names = all_feature_names

    def forward(self, feature_values: torch.Tensor):
        """
        feature_values: Tensor, shape=(batch_size, n_features)
        returns: Tensor, shape=(batch_size, D)
        """
        batch_size, n_features = feature_values.size()
        # mask: 모든 feature가 valid하다고 가정 → 모두 True
        mask = torch.ones((batch_size, n_features),
                          dtype=torch.bool,
                          device=feature_values.device)

        feature_names = self.all_feature_names  # 고정된 순서
        token_emb = self.embedder(feature_names, feature_values)
        h = self.encoder(token_emb, mask=mask)  # (batch_size, D)
        return h


# ===============================
# 1. 라이브러리 및 설정
# ===============================
print("🚀 일반화된 임베딩 기반 XGBoost 순차 학습 시작")
print("="*70)

# ===============================
# 2. 데이터 로드 및 자동 분석 (임베딩 버전)
# ===============================
def load_and_analyze_data_embedding(sentiment_file, target_file):
    """데이터 로드 및 구조 자동 분석 (임베딩 버전)"""

    sentiment_df = pd.read_csv(sentiment_file)
    target_df = pd.read_csv(target_file)

    print(f"📊 입력 데이터 형태: {sentiment_df.shape}")
    print(f"📊 타겟 데이터 형태: {target_df.shape}")

    # 입력 데이터 구조 검증
    required_sentiment_cols = ['final_cluster', 'negative', 'positive', 'time_label']
    missing_sentiment_cols = [
        col for col in required_sentiment_cols
        if col not in sentiment_df.columns
    ]
    if missing_sentiment_cols:
        raise ValueError(f"입력 데이터에 필수 컬럼이 없습니다: {missing_sentiment_cols}")

    # 타겟 데이터에서 후보자 자동 추출
    target_columns = [col for col in target_df.columns if col != 'time_label']
    if 'time_label' not in target_df.columns:
        raise ValueError("타겟 데이터에 'time_label' 컬럼이 없습니다.")

    print(f"\n📋 감지된 후보자: {target_columns}")
    print(f"📋 입력 데이터 time_label: {sorted(sentiment_df['time_label'].unique())}")
    print(f"📋 타겟 데이터 time_label: {sorted(target_df['time_label'].unique())}")

    # 데이터 가용성 분석
    available_target_labels = sorted(target_df['time_label'].unique())
    all_input_labels = sorted(sentiment_df['time_label'].unique())
    missing_target_labels = [
        label for label in all_input_labels
        if label not in available_target_labels
    ]

    print(f"\n📊 데이터 현황:")
    print(f"   학습 가능 time_label: {available_target_labels}")
    print(f"   예측 대상 time_label: {missing_target_labels}")

    return sentiment_df, target_df, target_columns, available_target_labels, missing_target_labels


# ===============================
# 3. 일반화된 클러스터 임베딩 생성 (KcELECTRA)
# ===============================
def create_kcelectra_embeddings(df, embedding_dim=768):
    """KcELECTRA를 활용한 클러스터 임베딩 생성"""

    print(f"\n🔧 KcELECTRA 기반 클러스터 임베딩 생성")

    model_name = "beomi/KcELECTRA-base-v2022"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    unique_clusters = df['final_cluster'].unique()
    print(f"   📊 총 고유 클러스터 수: {len(unique_clusters)}")

    cluster_texts = []
    for cluster in unique_clusters:
        if pd.isna(cluster):
            cluster_texts.append("알 수 없음")
        else:
            processed_text = str(cluster).replace('_', ' 관련 ')
            cluster_texts.append(processed_text)

    embeddings = []
    with torch.no_grad():
        for text in cluster_texts:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=128,
                truncation=True,
                padding=True
            ).to(device)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])

    embeddings = np.array(embeddings)

    if embedding_dim < embeddings.shape[1]:
        pca = PCA(n_components=embedding_dim)
        embeddings = pca.fit_transform(embeddings)
        print(f"   📈 PCA 설명 분산 비율: {pca.explained_variance_ratio_.sum():.3f}")

    embedding_dict = {
        cluster: embeddings[i]
        for i, cluster in enumerate(unique_clusters)
    }

    print(f"   ✅ KcELECTRA 임베딩 생성 완료: {embeddings.shape}")
    return embedding_dict, tokenizer, model


# ===============================
# 4. 일반화된 임베딩 기반 감정 데이터 집계
# ===============================
def extract_candidate_from_cluster_general(cluster_name, candidate_list):
    """일반화된 후보자 추출"""
    if pd.isna(cluster_name):
        return 'unknown'
    for candidate in candidate_list:
        if candidate in str(cluster_name):
            return candidate
    return 'other'


def aggregate_sentiment_with_embeddings_general(df, embedding_dict, candidate_list):
    """일반화된 임베딩 기반 감정 데이터 집계"""

    print(f"\n🔧 임베딩 기반 감정 데이터 집계 (후보자: {candidate_list})")

    aggregated_data = []
    embedding_dim = len(next(iter(embedding_dict.values())))

    for time_label in df['time_label'].unique():
        time_data = df[df['time_label'] == time_label]

        for candidate in candidate_list:
            candidate_clusters = []
            for _, row in time_data.iterrows():
                cluster_candidate = extract_candidate_from_cluster_general(
                    row['final_cluster'], candidate_list
                )
                if cluster_candidate == candidate:
                    candidate_clusters.append(row)

            if candidate_clusters:
                total_positive = sum([row['positive'] for row in candidate_clusters])
                total_negative = sum([row['negative'] for row in candidate_clusters])
                cluster_count = len(candidate_clusters)

                avg_positive = total_positive / cluster_count
                avg_negative = total_negative / cluster_count
                sentiment_score = avg_positive - avg_negative

                embeddings = []
                for row in candidate_clusters:
                    if row['final_cluster'] in embedding_dict:
                        embeddings.append(embedding_dict[row['final_cluster']])

                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                else:
                    avg_embedding = np.zeros(embedding_dim)

                result = {
                    'time_label': time_label,
                    'candidate': candidate,
                    'avg_positive': avg_positive,
                    'avg_negative': avg_negative,
                    'sentiment_score': sentiment_score,
                    'cluster_count': cluster_count
                }

                for i in range(embedding_dim):
                    result[f'embedding_{i}'] = avg_embedding[i]

                aggregated_data.append(result)

            else:
                result = {
                    'time_label': time_label,
                    'candidate': candidate,
                    'avg_positive': 0.5,
                    'avg_negative': 0.5,
                    'sentiment_score': 0.0,
                    'cluster_count': 0
                }

                for i in range(embedding_dim):
                    result[f'embedding_{i}'] = 0.0

                aggregated_data.append(result)

    result_df = pd.DataFrame(aggregated_data)
    print(f"   ✅ 집계 완료: {result_df.shape}")
    return result_df

# ===============================
# 5. 일반화된 임베딩 기반 피처 엔지니어링
# ===============================
def create_embedding_features_general(agg_df, candidate_list):
    """일반화된 임베딩 기반 피처 생성"""

    print(f"\n🔧 임베딩 기반 피처 엔지니어링 (후보자 {len(candidate_list)}명)")

    features_data = []
    embedding_dim = len([col for col in agg_df.columns if col.startswith('embedding_')])

    for time_label in agg_df['time_label'].unique():
        time_data = agg_df[agg_df['time_label'] == time_label]

        features = {'time_label': time_label}

        for candidate in candidate_list:
            candidate_data = time_data[time_data['candidate'] == candidate].iloc[0]

            features[f'{candidate}_positive'] = candidate_data['avg_positive']
            features[f'{candidate}_negative'] = candidate_data['avg_negative']
            features[f'{candidate}_sentiment'] = candidate_data['sentiment_score']
            features[f'{candidate}_cluster_count'] = candidate_data['cluster_count']

            for i in range(embedding_dim):
                features[f'{candidate}_embedding_{i}'] = candidate_data[f'embedding_{i}']

        sentiments = [features[f'{c}_sentiment'] for c in candidate_list]
        total_sentiment = sum(sentiments)
        if total_sentiment != 0:
            for i, candidate in enumerate(candidate_list):
                features[f'{candidate}_relative_sentiment'] = sentiments[i] / total_sentiment
        else:
            for candidate in candidate_list:
                features[f'{candidate}_relative_sentiment'] = 1.0 / len(candidate_list)

        embeddings = {}
        for candidate in candidate_list:
            embedding = np.array([
                features[f'{candidate}_embedding_{i}']
                for i in range(embedding_dim)
            ])
            embeddings[candidate] = embedding

        for i, cand1 in enumerate(candidate_list):
            for j, cand2 in enumerate(candidate_list):
                if i < j:
                    sim = cosine_similarity(
                        [embeddings[cand1]], [embeddings[cand2]]
                    )[0][0]
                    features[f'similarity_{cand1}_{cand2}'] = sim

        for candidate in candidate_list:
            embedding_norm = np.linalg.norm(embeddings[candidate])
            features[f'{candidate}_embedding_strength'] = embedding_norm

        all_embeddings = np.array(list(embeddings.values()))
        features['embedding_mean'] = np.mean(all_embeddings)
        features['embedding_std'] = np.std(all_embeddings)
        features['embedding_max'] = np.max(all_embeddings)
        features['embedding_min'] = np.min(all_embeddings)

        features_data.append(features)

    result_df = pd.DataFrame(features_data)
    print(f"   ✅ 피처 생성 완료: {result_df.shape}")
    return result_df


# ===============================
# 6. 일반화된 XGBoost 클래스
# ===============================
class GeneralSequentialMultiOutputXGBoost:
    def __init__(self, target_columns, **params):
        self.target_columns = target_columns
        self.models = {}
        self.params = params
        self.is_fitted = False

    def fit(self, X, y):
        """초기 학습"""
        print("   🔧 새로운 모델 초기화 및 학습...")
        for col in self.target_columns:
            if col in y.columns:
                model = xgb.XGBRegressor(**self.params)
                model.fit(X, y[col])
                self.models[col] = model
        self.is_fitted = True

    def partial_fit(self, X, y):
        """기존 모델에 증분 학습"""
        if not self.is_fitted:
            return self.fit(X, y)

        print("   🔄 기존 모델에 증분 학습 수행...")
        for col in self.target_columns:
            if col in y.columns and col in self.models:
                existing_booster = self.models[col].get_booster()
                new_model = xgb.XGBRegressor(**self.params)
                new_model.fit(X, y[col], xgb_model=existing_booster)
                self.models[col] = new_model

    def predict(self, X):
        """예측 수행"""
        predictions = {}
        for col in self.target_columns:
            if col in self.models:
                predictions[col] = self.models[col].predict(X)
            else:
                predictions[col] = np.full(len(X), 1.0 / len(self.target_columns))

        return pd.DataFrame(predictions, index=X.index)

# ===============================
# 7. 일반화된 메인 실행 함수
# ===============================
def run_general_embedding_pipeline(sentiment_file, target_file, embedding_dim=10):
    """일반화된 임베딩 기반 예측 모델 실행"""

    try:
        # 1. 데이터 로드 및 분석
        sentiment_df, target_df, target_columns, available_target_labels, missing_target_labels = \
            load_and_analyze_data_embedding(sentiment_file, target_file)

        # 2. 클러스터 임베딩 생성 (KcELECTRA)
        embedding_dict, tokenizer, model = create_kcelectra_embeddings(
            sentiment_df, embedding_dim=embedding_dim
        )

        # 3. 임베딩 기반 감정 데이터 집계
        agg_sentiment = aggregate_sentiment_with_embeddings_general(
            sentiment_df, embedding_dict, target_columns
        )

        # 4. 임베딩 기반 피처 엔지니어링
        features_df = create_embedding_features_general(agg_sentiment, target_columns)

        # 5. 하이퍼파라미터 튜닝 (간소화된 버전)
        best_params = {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        # 6. Set‐Transformer 임베더 초기화
        #    features_df.columns 중 'time_label' 제외
        all_feature_names = [col for col in features_df.columns if col != 'time_label']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_embedder = TabularSetEmbedder(
            all_feature_names=all_feature_names,
            d_name=16,
            d_val=16,
            n_heads=4,
            dim_ff=64,
            dropout=0.1
        ).to(device)

        # 7. 최종 모델 학습 (XGBoost + Set‐Transformer 임베딩)
        print(f"\n{'='*70}")
        print("🏆 최종 모델 학습 (임베딩 + XGBoost)")
        print(f"{'='*70}")

        final_model = GeneralSequentialMultiOutputXGBoost(target_columns, **best_params)

        for epoch in available_target_labels:
            print(f"\n📚 Epoch {epoch} 학습 중...")

            # 7-1. 데이터 준비
            current_features = features_df[features_df['time_label'] == epoch]
            current_target = target_df[target_df['time_label'] == epoch]
            current_merged = pd.merge(
                current_features, current_target,
                on='time_label', how='inner'
            )

            if len(current_merged) > 0:
                feature_columns = [
                    col for col in current_features.columns
                    if col != 'time_label'
                ]
                X_epoch = current_merged[feature_columns].reset_index(drop=True)
                y_epoch = current_merged[target_columns].reset_index(drop=True)

                print(f"   📊 원본 X_epoch shape: {X_epoch.shape}")

                # 7-2. Set‐Transformer 임베딩 생성
                X_vals = torch.tensor(
                    X_epoch.values, dtype=torch.float32, device=device
                )  # (batch_size, n_features)
                with torch.no_grad():
                    emb_epoch = set_embedder(X_vals).cpu().numpy()  # (batch_size, D)

                emb_cols = [f'set_emb_{i}' for i in range(emb_epoch.shape[1])]
                df_emb = pd.DataFrame(emb_epoch, columns=emb_cols)

                X_epoch_emb = pd.concat(
                    [X_epoch.reset_index(drop=True),
                     df_emb.reset_index(drop=True)],
                    axis=1
                )
                print(f"   🔗 X_epoch_emb shape (with set-emb): {X_epoch_emb.shape}")

                # 7-3. XGBoost 학습
                if not final_model.is_fitted:
                    final_model.fit(X_epoch_emb, y_epoch)
                    final_model.is_fitted = True
                    print("   ✅ XGBoost 새 모델 학습 완료")
                else:
                    final_model.partial_fit(X_epoch_emb, y_epoch)
                    print("   🔄 XGBoost 모델 증분 학습(Partial Fit) 완료")

                # 7-4. Epoch별 성능 평가
                predictions = final_model.predict(X_epoch_emb)
                predictions_normalized = predictions.div(
                    predictions.sum(axis=1), axis=0
                ) * 100
                y_norm = y_epoch.div(y_epoch.sum(axis=1), axis=0) * 100

                print(f"   📊 Epoch {epoch} 예측 결과:")
                for candidate in target_columns:
                    actual = y_norm[candidate].iloc[0]
                    pred = predictions_normalized[candidate].iloc[0]
                    mae = abs(actual - pred)
                    print(f"      {candidate}: 실제={actual:.2f}%, 예측={pred:.2f}%, MAE={mae:.2f}%")

                overall_mae = mean_absolute_error(
                    y_norm.values.flatten(),
                    predictions_normalized.values.flatten()
                )
                print(f"   🎯 Epoch {epoch} 전체 MAE: {overall_mae:.2f}%")

        # 8. 결측된 time_label들에 대한 예측
        prediction_results = {}
        if missing_target_labels:
            print(f"\n{'='*70}")
            print(f"🔮 결측 time_label 예측: {missing_target_labels}")
            print(f"{'='*70}")

            full_features = features_df.copy()
            feat_vals = torch.tensor(
                full_features.drop(columns=['time_label']).values,
                dtype=torch.float32, device=device
            )
            with torch.no_grad():
                emb_full = set_embedder(feat_vals).cpu().numpy()  # (N_all, D)

            df_emb_full = pd.DataFrame(
                emb_full, columns=[f'set_emb_{i}' for i in range(emb_full.shape[1])]
            )
            full_with_emb = pd.concat(
                [full_features.reset_index(drop=True).drop(columns=['time_label']),
                 df_emb_full.reset_index(drop=True)], axis=1
            )

            for time_label in missing_target_labels:
                mask = full_features['time_label'] == time_label
                if mask.sum() == 0:
                    print(f"   ⚠️ time_label {time_label} 데이터 X")
                    continue

                X_pred = full_with_emb[mask.values].reset_index(drop=True)
                preds = final_model.predict(X_pred)
                preds_norm = preds.div(preds.sum(axis=1), axis=0) * 100

                print(f"\n   🎯 time_label {time_label} 예측 지지율:")
                for candidate in target_columns:
                    val = preds_norm[candidate].iloc[0]
                    print(f"      {candidate}: {val:.2f}%")
                prediction_results[time_label] = preds_norm.iloc[0]

        # 9. 결과 요약
        print(f"\n{'='*70}")
        print("📊 최종 결과 요약")
        print(f"{'='*70}")
        print(f"🏆 사용된 파라미터: {best_params}")
        print(f"📚 학습에 사용된 time_label: {available_target_labels}")
        print(f"🔮 예측된 time_label: {missing_target_labels}")
        print(f"🎯 임베딩 차원: {embedding_dim} + SetTransformer({set_embedder.out_dim})")
        if missing_target_labels:
            print(f"\n🗳️ 예측 결과:")
            for time_label, predictions in prediction_results.items():
                print(f"   time_label {time_label}:")
                for candidate in target_columns:
                    val = predictions[candidate]
                    print(f"      {candidate}: {val:.2f}%")

        print(f"\n✅ 일반화된 임베딩 기반 예측 모델 완료!")
        print("="*70)

        return final_model, prediction_results, embedding_dict

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # sentiment_file, target_file 경로를 본인 환경에 맞게 수정
    sentiment_path = "cluster_sentiment_summary_final_21대.csv"
    target_path = "weighted_average_21대.csv"

    # embedding_dim 설정
    embedding_dim = 10

    # 함수를 호출하여 최종 모델과 예측 결과를 얻음
    final_model, prediction_results, embedding_dict = run_general_embedding_pipeline(
        sentiment_path,
        target_path,
        embedding_dim=embedding_dim
    )

    if final_model is None:
        print("❌ 모델 학습 실패. 에러 메시지 확인")
    else:
        print("\n✅ 파이프라인이 정상적으로 종료")
        print("prediction_results 예시:", prediction_results)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. 라이브러리 및 설정
# ===============================
print("🚀 일반화된 XGBoost 순차 학습 시작")
print("="*70)

# ===============================
# 2. 데이터 로드 및 자동 분석
# ===============================
def load_and_analyze_data(sentiment_file, target_file):
    """데이터 로드 및 구조 자동 분석"""

    # 데이터 로드
    sentiment_df = pd.read_csv(sentiment_file)
    target_df = pd.read_csv(target_file)

    print(f"📊 입력 데이터 형태: {sentiment_df.shape}")
    print(f"📊 타겟 데이터 형태: {target_df.shape}")

    # 입력 데이터 구조 분석
    required_sentiment_cols = ['final_cluster', 'negative', 'positive', 'time_label']
    missing_sentiment_cols = [col for col in required_sentiment_cols if col not in sentiment_df.columns]
    if missing_sentiment_cols:
        raise ValueError(f"입력 데이터에 필수 컬럼이 없습니다: {missing_sentiment_cols}")

    # 타겟 데이터 구조 분석 (time_label을 제외한 나머지가 후보자명)
    target_columns = [col for col in target_df.columns if col != 'time_label']
    if 'time_label' not in target_df.columns:
        raise ValueError("타겟 데이터에 'time_label' 컬럼이 없습니다.")

    print(f"\n📋 감지된 후보자: {target_columns}")
    print(f"📋 입력 데이터 time_label: {sorted(sentiment_df['time_label'].unique())}")
    print(f"📋 타겟 데이터 time_label: {sorted(target_df['time_label'].unique())}")

    # 데이터 가용성 분석
    available_target_labels = sorted(target_df['time_label'].unique())
    all_input_labels = sorted(sentiment_df['time_label'].unique())
    missing_target_labels = [label for label in all_input_labels if label not in available_target_labels]

    print(f"\n📊 데이터 현황:")
    print(f"   학습 가능 time_label: {available_target_labels}")
    print(f"   예측 대상 time_label: {missing_target_labels}")

    return sentiment_df, target_df, target_columns, available_target_labels, missing_target_labels

# ===============================
# 3. 후보자 자동 추출 및 감정 데이터 집계
# ===============================
def extract_candidate_from_cluster(cluster_name, candidate_list):
    """클러스터명에서 후보자명 자동 추출"""
    if pd.isna(cluster_name):
        return 'unknown'

    # 후보자명이 클러스터명에 포함된 경우 추출
    for candidate in candidate_list:
        if candidate in str(cluster_name):
            return candidate
    return 'other'

def aggregate_sentiment_by_time(df, candidate_list):
    """time_label별로 후보자 감정 데이터 집계 (일반화 버전)"""

    print(f"\n🔧 감정 데이터 집계 (후보자: {candidate_list})")

    # 후보자 추출
    df['candidate'] = df['final_cluster'].apply(
        lambda x: extract_candidate_from_cluster(x, candidate_list)
    )

    aggregated_data = []

    for time_label in df['time_label'].unique():
        time_data = df[df['time_label'] == time_label]

        for candidate in candidate_list:
            candidate_data = time_data[time_data['candidate'] == candidate]

            if len(candidate_data) > 0:
                total_mentions = len(candidate_data)
                avg_positive = candidate_data['positive'].mean()
                avg_negative = candidate_data['negative'].mean()
                sentiment_score = avg_positive - avg_negative

                aggregated_data.append({
                    'time_label': time_label,
                    'candidate': candidate,
                    'avg_positive': avg_positive,
                    'avg_negative': avg_negative,
                    'sentiment_score': sentiment_score,
                    'mention_count': total_mentions
                })
            else:
                # 해당 시점에 데이터가 없는 경우 중립값 사용
                aggregated_data.append({
                    'time_label': time_label,
                    'candidate': candidate,
                    'avg_positive': 0.5,
                    'avg_negative': 0.5,
                    'sentiment_score': 0.0,
                    'mention_count': 0
                })

    result_df = pd.DataFrame(aggregated_data)
    print(f"   ✅ 집계 완료: {result_df.shape}")

    return result_df

# ===============================
# 4. 일반화된 피처 엔지니어링
# ===============================
def create_features_general(agg_df, candidate_list):
    """일반화된 피처 생성 (후보자 수에 관계없이 동작)"""

    print(f"\n🔧 피처 엔지니어링 (후보자 {len(candidate_list)}명)")

    features_data = []

    for time_label in agg_df['time_label'].unique():
        time_data = agg_df[agg_df['time_label'] == time_label]

        features = {'time_label': time_label}

        # 각 후보자별 기본 피처
        for candidate in candidate_list:
            candidate_data = time_data[time_data['candidate'] == candidate].iloc[0]

            features[f'{candidate}_positive'] = candidate_data['avg_positive']
            features[f'{candidate}_negative'] = candidate_data['avg_negative']
            features[f'{candidate}_sentiment'] = candidate_data['sentiment_score']
            features[f'{candidate}_mentions'] = candidate_data['mention_count']

        # 상대적 감정 점수 계산
        sentiments = [features[f'{c}_sentiment'] for c in candidate_list]
        total_sentiment = sum(sentiments)

        if total_sentiment != 0:
            for i, candidate in enumerate(candidate_list):
                features[f'{candidate}_relative_sentiment'] = sentiments[i] / total_sentiment
        else:
            for candidate in candidate_list:
                features[f'{candidate}_relative_sentiment'] = 1.0 / len(candidate_list)

        # 후보자간 감정 차이 피처 (모든 조합)
        for i, cand1 in enumerate(candidate_list):
            for j, cand2 in enumerate(candidate_list):
                if i < j:  # 중복 방지
                    features[f'sentiment_gap_{cand1}_{cand2}'] = (
                        features[f'{cand1}_sentiment'] - features[f'{cand2}_sentiment']
                    )

        # 전체 감정 통계
        features['total_positive'] = sum([features[f'{c}_positive'] for c in candidate_list])
        features['total_negative'] = sum([features[f'{c}_negative'] for c in candidate_list])
        features['total_mentions'] = sum([features[f'{c}_mentions'] for c in candidate_list])
        features['sentiment_variance'] = np.var(sentiments)

        features_data.append(features)

    result_df = pd.DataFrame(features_data)
    print(f"   ✅ 피처 생성 완료: {result_df.shape}")

    return result_df

# ===============================
# 5. 일반화된 XGBoost 클래스
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
                # 모델이 없는 경우 평균값으로 예측
                predictions[col] = np.full(len(X), 1.0 / len(self.target_columns))

        return pd.DataFrame(predictions, index=X.index)

# ===============================
# 6. 일반화된 데이터 준비 및 평가 함수
# ===============================
def prepare_epoch_data_general(features_df, target_df, epoch, target_columns):
    """일반화된 epoch 데이터 준비"""
    current_features = features_df[features_df['time_label'] == epoch]
    current_target = target_df[target_df['time_label'] == epoch]

    if len(current_features) == 0 or len(current_target) == 0:
        return None

    current_merged = pd.merge(current_features, current_target, on='time_label', how='inner')

    if len(current_merged) == 0:
        return None

    feature_columns = [col for col in current_features.columns if col != 'time_label']

    X_epoch = current_merged[feature_columns]
    y_epoch = current_merged[target_columns]

    return X_epoch, y_epoch

def evaluate_epoch_performance_general(model, X, y, epoch, target_columns):
    """일반화된 epoch별 성능 평가"""
    predictions = model.predict(X)

    # 예측값과 실제값을 0-100 범위로 정규화
    predictions_normalized = predictions.div(predictions.sum(axis=1), axis=0) * 100
    y_normalized = y.div(y.sum(axis=1), axis=0) * 100

    print(f"   📊 Epoch {epoch} 예측 결과:")
    for candidate in target_columns:
        if candidate in y_normalized.columns and candidate in predictions_normalized.columns:
            actual = y_normalized[candidate].iloc[0]
            predicted = predictions_normalized[candidate].iloc[0]
            mae = abs(actual - predicted)
            print(f"      {candidate}: 실제={actual:.2f}%, 예측={predicted:.2f}%, MAE={mae:.2f}%")

    # 전체 MAE 계산
    overall_mae = mean_absolute_error(y_normalized.values.flatten(),
                                     predictions_normalized.values.flatten())

    print(f"   🎯 Epoch {epoch} 전체 MAE: {overall_mae:.2f}%")

    return {
        'epoch': epoch,
        'predictions': predictions_normalized,
        'actual': y_normalized,
        'mae': overall_mae
    }

# ===============================
# 7. 일반화된 하이퍼파라미터 튜닝
# ===============================
def hyperparameter_tuning_general(features_df, target_df, target_columns, available_labels):
    """일반화된 하이퍼파라미터 튜닝"""

    small_param_grid = {
        'n_estimators': [30, 50],
        'learning_rate': [0.1, 0.15],
        'max_depth': [3, 4],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    best_score = float('inf')
    best_params = None

    if len(available_labels) < 2:
        print("⚠️ 교차 검증을 위한 충분한 데이터가 없습니다. 기본 파라미터를 사용합니다.")
        return {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    total_combinations = len(list(ParameterGrid(small_param_grid)))

    print(f"\n{'='*70}")
    print("🔧 하이퍼파라미터 튜닝 시작")
    print(f"{'='*70}")
    print(f"사용 가능한 time_label: {available_labels}")
    print(f"총 {total_combinations}개 조합 테스트")

    for i, params in enumerate(ParameterGrid(small_param_grid)):
        print(f"\n🧪 조합 {i+1}/{total_combinations}: {params}")

        try:
            fold_scores = []

            for test_label in available_labels:
                train_labels = [label for label in available_labels if label != test_label]

                if not train_labels:
                    continue

                model = GeneralSequentialMultiOutputXGBoost(target_columns, **params)

                # 학습
                for epoch in train_labels:
                    epoch_data = prepare_epoch_data_general(features_df, target_df, epoch, target_columns)
                    if epoch_data is not None:
                        X_epoch, y_epoch = epoch_data
                        if epoch == train_labels[0]:
                            model.fit(X_epoch, y_epoch)
                        else:
                            model.partial_fit(X_epoch, y_epoch)

                # 테스트
                test_data = prepare_epoch_data_general(features_df, target_df, test_label, target_columns)
                if test_data is not None:
                    X_test, y_test = test_data
                    predictions = model.predict(X_test)
                    predictions_normalized = predictions.div(predictions.sum(axis=1), axis=0) * 100
                    y_test_normalized = y_test.div(y_test.sum(axis=1), axis=0) * 100

                    mae = mean_absolute_error(y_test_normalized.values.flatten(),
                                            predictions_normalized.values.flatten())
                    fold_scores.append(mae)

            if fold_scores:
                avg_score = np.mean(fold_scores)
                print(f"   📊 평균 MAE: {avg_score:.3f}%")

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params.copy()
                    print(f"   ✨ 새로운 최고 성능! MAE: {avg_score:.3f}%")

        except Exception as e:
            print(f"   ❌ 오류 발생: {str(e)}")
            continue

    return best_params

# ===============================
# 8. 일반화된 예측 함수
# ===============================
def predict_missing_time_labels_general(model, features_df, missing_labels, target_columns):
    """일반화된 결측 time_label 예측"""

    print(f"\n{'='*70}")
    print(f"🔮 결측 time_label 예측: {missing_labels}")
    print(f"{'='*70}")

    predictions_results = {}

    for time_label in missing_labels:
        predict_features = features_df[features_df['time_label'] == time_label]

        if len(predict_features) == 0:
            print(f"   ⚠️ time_label {time_label} 데이터가 없습니다.")
            continue

        feature_columns = [col for col in predict_features.columns if col != 'time_label']
        X_predict = predict_features[feature_columns]

        print(f"   📊 time_label {time_label} 예측 데이터 크기: {X_predict.shape}")

        # 예측 수행
        predictions = model.predict(X_predict)
        predictions_normalized = predictions.div(predictions.sum(axis=1), axis=0) * 100

        print(f"   🎯 time_label {time_label} 예측 지지율:")
        for candidate in target_columns:
            if candidate in predictions_normalized.columns:
                predicted_rate = predictions_normalized[candidate].iloc[0]
                print(f"      {candidate}: {predicted_rate:.2f}%")

        predictions_results[time_label] = predictions_normalized.iloc[0]

    return predictions_results

# ===============================
# 9. 메인 실행 함수 (일반화)
# ===============================
def run_general_pipeline(sentiment_file, target_file):
    """일반화된 파이프라인 실행"""

    try:
        # 1. 데이터 로드 및 분석
        sentiment_df, target_df, target_columns, available_target_labels, missing_target_labels = load_and_analyze_data(
            sentiment_file, target_file
        )

        # 2. 감정 데이터 집계
        agg_sentiment = aggregate_sentiment_by_time(sentiment_df, target_columns)

        # 3. 피처 엔지니어링
        features_df = create_features_general(agg_sentiment, target_columns)

        # 4. 하이퍼파라미터 튜닝
        best_params = hyperparameter_tuning_general(
            features_df, target_df, target_columns, available_target_labels
        )

        if best_params is None:
            print("하이퍼파라미터 튜닝 실패. 기본 파라미터 사용.")
            best_params = {
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 4,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

        # 5. 최종 모델 학습
        print(f"\n{'='*70}")
        print("🏆 최종 모델 학습")
        print(f"{'='*70}")

        final_model = GeneralSequentialMultiOutputXGBoost(target_columns, **best_params)

        for epoch in available_target_labels:
            print(f"\n📚 Epoch {epoch} 학습 중...")
            epoch_data = prepare_epoch_data_general(features_df, target_df, epoch, target_columns)
            if epoch_data is not None:
                X_epoch, y_epoch = epoch_data
                print(f"   📊 데이터 크기: {X_epoch.shape}")

                if epoch == available_target_labels[0]:
                    final_model.fit(X_epoch, y_epoch)
                else:
                    final_model.partial_fit(X_epoch, y_epoch)

                evaluate_epoch_performance_general(final_model, X_epoch, y_epoch, epoch, target_columns)

        # 6. 결측된 time_label들에 대한 예측
        prediction_results = {}
        if missing_target_labels:
            prediction_results = predict_missing_time_labels_general(
                final_model, features_df, missing_target_labels, target_columns
            )

        # 7. 결과 요약
        print(f"\n{'='*70}")
        print("최종 결과 요약")
        print(f"{'='*70}")
        print(f"최적 파라미터: {best_params}")
        print(f"학습에 사용된 time_label: {available_target_labels}")
        print(f"예측된 time_label: {missing_target_labels}")

        if missing_target_labels:
            print(f"\n🗳️ 예측 결과:")
            for time_label, predictions in prediction_results.items():
                print(f"   time_label {time_label}:")
                for candidate in target_columns:
                    if candidate in predictions:
                        print(f"      {candidate}: {predictions[candidate]:.2f}%")

        print(f"\n일반화된 예측 모델 완료!")
        print("="*70)

        return final_model, prediction_results

    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# ===============================
# 10. 사용 예시
# ===============================
if __name__ == "__main__":
    '''
    # 사용 예시 1: 20대 대선 데이터
    print("20대 대선 데이터로 테스트")
    model_20, results_20 = run_general_pipeline(
        "cluster_sentiment_summary_final_20대.csv",
        "weighted_average_20대.csv"
    )
    '''

    # 사용 예시 2: 21대 대선 데이터
    print("🗳️ 21대 대선 데이터로 테스트")
    model_21, results_21 = run_general_pipeline(
        "cluster_sentiment_summary_final_21대.csv",
        "weighted_average_21대.csv"
    )
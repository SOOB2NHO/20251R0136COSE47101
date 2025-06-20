import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import os

# GPU 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

class PoliticalDomainHyperParameters:
    """정치 도메인 특화 하이퍼파라미터"""

    # 모델 설정
    MODEL_NAME = 'monologg/koelectra-base-v3-discriminator'
    MAX_LENGTH = 128
    BATCH_SIZE = 32

    # 계층적 토픽 구조
    POLITICAL_HIERARCHY = {
        'level1': ['yoon', 'lee', 'neutral'],
        'level2': ['policy', 'scandal', 'campaign', 'general'],
        'level3_topics_per_person': 3
    }

    # ETM 파라미터
    NUM_TOPICS_TOTAL = 12
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 400
    DROPOUT = 0.5

    # 어휘 설정
    MIN_DF = 3
    MAX_DF = 0.8
    MAX_FEATURES = 1000

class PoliticalDocumentClassifier:
    """정치인별 문서 분류기"""

    def __init__(self):
        self.person_keywords = {
            'yoon': ['윤석열', '윤', '대통령', '현정부', '정부', '여당', '국민의힘'],
            'lee': ['이재명', '이', '민주당', '야당', '더불어민주당'],
            'neutral': []
        }

        self.issue_keywords = {
            'policy': ['정책', '공약', '법안', '제도', '개혁', '예산', '경제', '복지'],
            'scandal': ['수사', '검찰', '재판', '혐의', '비리', '논란', '의혹'],
            'campaign': ['선거', '캠페인', '유세', '공천', '후보', '지지율'],
            'general': ['발언', '회견', '성명', '입장', '반응', '비판']
        }

    def classify_documents(self, texts):
        """문서를 인물별, 이슈별로 분류"""
        person_labels = []
        issue_labels = []

        for text in tqdm(texts, desc="문서 분류 중"):
            text_lower = text.lower()

            # 인물 분류
            person_scores = {}
            for person, keywords in self.person_keywords.items():
                if person == 'neutral':
                    continue
                score = sum(text_lower.count(keyword) for keyword in keywords)
                person_scores[person] = score

            if max(person_scores.values()) > 0:
                dominant_person = max(person_scores, key=person_scores.get)
            else:
                dominant_person = 'neutral'

            person_labels.append(dominant_person)

            # 이슈 분류
            issue_scores = {}
            for issue, keywords in self.issue_keywords.items():
                score = sum(text_lower.count(keyword) for keyword in keywords)
                issue_scores[issue] = score

            if max(issue_scores.values()) > 0:
                dominant_issue = max(issue_scores, key=issue_scores.get)
            else:
                dominant_issue = 'general'

            issue_labels.append(dominant_issue)

        return person_labels, issue_labels

    def get_classification_summary(self, person_labels, issue_labels):
        """분류 결과 요약"""
        person_counts = {}
        issue_counts = {}
        
        for person in person_labels:
            person_counts[person] = person_counts.get(person, 0) + 1
            
        for issue in issue_labels:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
        return person_counts, issue_counts

class KoELECTRAEmbedding:
    """KoELECTRA 임베딩 모델"""

    def __init__(self, model_name=PoliticalDomainHyperParameters.MODEL_NAME):
        print(f"KoELECTRA 모델 로드 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        print("KoELECTRA 모델 로드 완료")

    def encode_documents(self, texts, batch_size=PoliticalDomainHyperParameters.BATCH_SIZE):
        """문서 임베딩 생성"""
        embeddings = []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="문서 임베딩 생성"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=PoliticalDomainHyperParameters.MAX_LENGTH,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_words(self, words, batch_size=PoliticalDomainHyperParameters.BATCH_SIZE):
        """단어 임베딩 생성"""
        embeddings = []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(words), batch_size), desc="단어 임베딩 생성"):
                batch = words[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

class ETMPreprocessor:
    """ETM 전처리 클래스 (구조 설정, 사전 설정, 임베딩)"""

    def __init__(self, text_column='comment_text', num_topics=12):
        self.text_column = text_column
        self.num_topics = num_topics

        # 모델 구성요소
        self.embedding_model = None
        self.classifier = None
        self.vectorizer = None
        self.vocab = None

        # 처리 결과 저장
        self.preprocessing_results = {}

    def load_and_preprocess_data(self, csv_files):
        """여러 CSV 파일을 로드하고 전처리"""
        all_texts = []
        file_info = []

        for csv_path in csv_files:
            try:
                print(f"📁 데이터 로드 중: {csv_path}")
                df = pd.read_csv(csv_path, encoding='utf-8')

                texts = df[self.text_column].dropna().drop_duplicates().tolist()
                texts = [text.strip() for text in texts if text.strip()]

                print(f"✅ {len(texts):,}개 유효 텍스트 추출")

                for text in texts:
                    file_info.append({
                        'file_path': csv_path,
                        'text': text
                    })

                all_texts.extend(texts)

            except Exception as e:
                print(f"❌ 파일 로드 실패 {csv_path}: {e}")
                continue

        print(f"🔄 총 {len(all_texts):,}개 문서 로드 완료")
        return all_texts, file_info

    def setup_hierarchical_structure(self):
        """계층적 토픽 구조 설정"""
        print("🏗️ 계층적 토픽 구조 설정 중...")

        hierarchy = PoliticalDomainHyperParameters.POLITICAL_HIERARCHY
        
        print("📊 정치 도메인 계층 구조:")
        print(f"   Level 1 (인물): {hierarchy['level1']}")
        print(f"   Level 2 (이슈): {hierarchy['level2']}")
        print(f"   총 토픽 수: {len(hierarchy['level1']) * len(hierarchy['level2'])}")

        # 토픽 매핑 생성
        topic_mapping = {}
        topic_id = 0
        
        for person in hierarchy['level1']:
            for issue in hierarchy['level2']:
                topic_mapping[topic_id] = {
                    'person': person,
                    'issue': issue,
                    'label': f"{person}_{issue}"
                }
                topic_id += 1

        print("✅ 계층적 구조 설정 완료")
        return hierarchy, topic_mapping

    def setup_political_dictionary(self):
        """정치인별 단어 사전 설정"""
        print("📚 정치인별 단어 사전 설정 중...")

        self.classifier = PoliticalDocumentClassifier()
        
        print("👥 인물별 키워드:")
        for person, keywords in self.classifier.person_keywords.items():
            if keywords:  # neutral은 빈 리스트이므로 제외
                print(f"   {person}: {keywords}")

        print("\n📋 이슈별 키워드:")
        for issue, keywords in self.classifier.issue_keywords.items():
            print(f"   {issue}: {keywords}")

        print("✅ 정치인별 단어 사전 설정 완료")
        return self.classifier

    def classify_documents(self, texts):
        """문서 분류 수행"""
        print("🏷️ 문서 분류 수행 중...")
        
        if self.classifier is None:
            self.setup_political_dictionary()

        person_labels, issue_labels = self.classifier.classify_documents(texts)
        person_counts, issue_counts = self.classifier.get_classification_summary(person_labels, issue_labels)

        print(f"📊 인물별 분포: {person_counts}")
        print(f"📊 이슈별 분포: {issue_counts}")

        return person_labels, issue_labels

    def create_bow_matrix(self, texts):
        """BoW 행렬 생성"""
        print("📊 BoW 행렬 생성 중...")

        self.vectorizer = CountVectorizer(
            min_df=PoliticalDomainHyperParameters.MIN_DF,
            max_df=PoliticalDomainHyperParameters.MAX_DF,
            max_features=PoliticalDomainHyperParameters.MAX_FEATURES,
            token_pattern=r'\b[가-힣]{2,}\b'
        )

        bow_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.vocab = self.vectorizer.get_feature_names_out()

        print(f"📊 어휘 크기: {len(self.vocab)}")
        print(f"📊 BoW 행렬 크기: {bow_matrix.shape}")

        return bow_matrix

    def generate_embeddings(self, texts, person_labels, issue_labels):
        """임베딩 생성"""
        print("🔧 임베딩 생성 중...")

        # KoELECTRA 임베딩 모델 로드
        self.embedding_model = KoELECTRAEmbedding()

        # BoW 행렬 생성
        bow_matrix = self.create_bow_matrix(texts)

        # 단어 임베딩 생성
        print("📝 단어 임베딩 생성 중...")
        word_embeddings = self.embedding_model.encode_words(self.vocab.tolist())

        # 차원 조정
        if word_embeddings.shape[1] != PoliticalDomainHyperParameters.EMBEDDING_DIM:
            target_dim = min(PoliticalDomainHyperParameters.EMBEDDING_DIM, word_embeddings.shape[1])
            if target_dim > 0:
                pca = PCA(n_components=target_dim)
                word_embeddings = pca.fit_transform(word_embeddings)
                word_embeddings = normalize(word_embeddings, norm='l2')

        # 토픽 임베딩 초기화
        print("🏛️ 토픽 임베딩 초기화 중...")
        topic_embeddings = self._initialize_political_topic_embeddings(
            texts, person_labels, issue_labels
        )

        print("✅ 임베딩 생성 완료")
        return bow_matrix, word_embeddings, topic_embeddings

    def _initialize_political_topic_embeddings(self, texts, person_labels, issue_labels):
        """정치인별 토픽 임베딩 초기화"""
        print("🏛️ 정치인별 토픽 임베딩 초기화 중...")

        doc_embeddings = self.embedding_model.encode_documents(texts)

        # 정치인별 평균 임베딩
        person_embeddings = {}
        hierarchy = PoliticalDomainHyperParameters.POLITICAL_HIERARCHY

        for person in hierarchy['level1']:
            if person == 'neutral':
                neutral_mask = np.array(person_labels) == 'neutral'
                if np.any(neutral_mask):
                    person_embeddings[person] = np.mean(doc_embeddings[neutral_mask], axis=0)
                else:
                    person_embeddings[person] = np.random.randn(doc_embeddings.shape[1])
            else:
                person_mask = np.array(person_labels) == person
                if np.any(person_mask):
                    person_embeddings[person] = np.mean(doc_embeddings[person_mask], axis=0)
                else:
                    person_embeddings[person] = np.random.randn(doc_embeddings.shape[1])

        # 이슈별 평균 임베딩
        issue_embeddings = {}
        for issue in hierarchy['level2']:
            issue_mask = np.array(issue_labels) == issue
            if np.any(issue_mask):
                issue_embeddings[issue] = np.mean(doc_embeddings[issue_mask], axis=0)
            else:
                issue_embeddings[issue] = np.random.randn(doc_embeddings.shape[1])

        # 계층적 토픽 임베딩 생성
        topic_embeddings = []
        for person in hierarchy['level1']:
            for issue in hierarchy['level2']:
                topic_emb = (person_embeddings[person] * 0.6 +
                           issue_embeddings[issue] * 0.4)
                topic_embeddings.append(topic_emb)

        return np.array(topic_embeddings)

    def process_all(self, texts):
        """모든 전처리 과정을 한 번에 실행"""
        print("🚀 === ETM 전처리 과정 시작 ===")

        # 1. 계층적 구조 설정
        hierarchy, topic_mapping = self.setup_hierarchical_structure()

        # 2. 정치인별 사전 설정
        classifier = self.setup_political_dictionary()

        # 3. 문서 분류
        person_labels, issue_labels = self.classify_documents(texts)

        # 4. 임베딩 생성
        bow_matrix, word_embeddings, topic_embeddings = self.generate_embeddings(
            texts, person_labels, issue_labels
        )

        # 결과 저장
        self.preprocessing_results = {
            'hierarchy': hierarchy,
            'topic_mapping': topic_mapping,
            'classifier': classifier,
            'person_labels': person_labels,
            'issue_labels': issue_labels,
            'bow_matrix': bow_matrix,
            'word_embeddings': word_embeddings,
            'topic_embeddings': topic_embeddings,
            'vocab': self.vocab,
            'vectorizer': self.vectorizer
        }

        print("✅ === ETM 전처리 과정 완료 ===")
        return self.preprocessing_results

def run_etm_preprocessing_pipeline():
    """ETM 전처리 파이프라인 실행"""
    print("🚀 === ETM 전처리 파이프라인 시작 ===")

    # CSV 파일 리스트 정의
    csv_files = [
        "comments_0204_0206.csv",
        "comments_0212_0214.csv",
        "comments_0222_0224.csv",
        "comments_0226_0228.csv",
        "comments_0303_0305.csv"
    ]

    # 존재하는 파일만 필터링
    existing_files = [f for f in csv_files if os.path.exists(f)]
    print(f"📁 처리 가능한 파일: {len(existing_files)}개")

    if not existing_files:
        print("❌ 처리할 CSV 파일이 없습니다.")
        return

    # 전처리 객체 생성
    preprocessor = ETMPreprocessor(text_column='comment_text', num_topics=12)

    # 1. 데이터 로드
    all_texts, file_info = preprocessor.load_and_preprocess_data(existing_files)

    if not all_texts:
        print("❌ 유효한 텍스트 데이터가 없습니다.")
        return

    # 2. 전체 전처리 과정 실행
    results = preprocessor.process_all(all_texts)

    print("\n🎉 === ETM 전처리 파이프라인 완료 ===")
    print(f"📊 결과 요약:")
    print(f"   문서 수: {results['bow_matrix'].shape[0]:,}")
    print(f"   어휘 수: {results['bow_matrix'].shape[1]:,}")
    print(f"   토픽 수: {results['topic_embeddings'].shape[0]}")
    print(f"   임베딩 차원: {results['word_embeddings'].shape[1]}")

    return preprocessor, results

# 실행 코드
if __name__ == "__main__":
    preprocessor, results = run_etm_preprocessing_pipeline()


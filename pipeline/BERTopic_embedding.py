import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import torch

# GPU 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

class ElectionKeywordDictionary:
    """2025 대선 주요 키워드 사전 클래스"""
    
    def __init__(self):
        self.political_keywords_2025 = self._build_comprehensive_keyword_dict()
        self.keyword_categories = self._categorize_keywords()
    
    def _build_comprehensive_keyword_dict(self):
        """2025 대선 이슈 기반 종합 키워드 사전 구축"""
        print("2025 대선 주요 키워드 사전 구축 중...")
        
        keywords = {
            # 주요 후보자 및 정치인
            "candidates": [
                "이재명", "김문수", "이준석", "권영국", "한덕수", "윤석열", 
                "홍준표", "안철수", "유승민", "원희룡", "김동연"
            ],
            
            # 정당 및 정치세력
            "parties": [
                "더불어민주당", "국민의힘", "개혁신당", "민주노동당", "무소속", 
                "비례정당", "탄핵", "여당", "야당", "제3당", "정권교체"
            ],
            
            # 경제 및 복지 정책
            "economy_welfare": [
                "부동산", "전세", "청년취업", "기본소득", "노동개혁", "최저임금", 
                "재정적자", "육아휴직", "청년정책", "일자리", "경제성장", "물가안정",
                "세금", "부채", "연금", "의료보험", "주택공급", "임대료"
            ],
            
            # 디지털 및 AI 정책
            "digital_ai": [
                "AI", "인공지능", "디지털전환", "플랫폼노동", "데이터주권", 
                "사이버보안", "디지털뉴딜", "스마트시티", "메타버스", "블록체인"
            ],
            
            # 외교 및 안보
            "foreign_security": [
                "북한", "한미동맹", "한중관계", "전작권", "핵무장", "사드", "안보",
                "외교", "통일", "국방", "군사", "평화", "협력", "갈등"
            ],
            
            # 사법 및 개혁
            "judicial_reform": [
                "검찰개혁", "사법개혁", "공수처", "검수완박", "특검", "수사권",
                "기소권", "법원", "헌법재판소", "권력분립", "민주주의"
            ],
            
            # 선거 및 정치제도
            "election_system": [
                "사전투표", "부정선거", "여론조사", "TV토론", "후보단일화", 
                "선거제도", "비례대표", "지역구", "투표율", "공천", "경선"
            ],
            
            # 사회 이슈
            "social_issues": [
                "코로나", "방역", "교육", "환경", "기후변화", "젠더", "인권",
                "다문화", "고령화", "저출산", "지방소멸", "균형발전"
            ]
        }
        
        # 모든 키워드를 하나의 리스트로 통합
        all_keywords = []
        for category, words in keywords.items():
            all_keywords.extend(words)
        
        print(f"총 {len(all_keywords)}개 키워드 사전 구축 완료")
        print(f"카테고리별 키워드 수:")
        for category, words in keywords.items():
            print(f"   {category}: {len(words)}개")
        
        return all_keywords
    
    def _categorize_keywords(self):
        """키워드 카테고리별 분류"""
        return {
            "candidates": ["이재명", "김문수", "이준석", "권영국", "한덕수", "윤석열"],
            "parties": ["더불어민주당", "국민의힘", "개혁신당", "민주노동당"],
            "economy": ["부동산", "전세", "청년취업", "기본소득", "노동개혁"],
            "digital": ["AI", "인공지능", "디지털전환", "플랫폼노동"],
            "foreign": ["북한", "한미동맹", "한중관계", "전작권", "핵무장"],
            "judicial": ["검찰개혁", "사법개혁", "공수처", "검수완박", "특검"],
            "election": ["사전투표", "부정선거", "여론조사", "TV토론"],
            "social": ["코로나", "방역", "교육", "환경", "기후변화"]
        }
    
    def get_keywords_by_category(self, category):
        """카테고리별 키워드 반환"""
        return self.keyword_categories.get(category, [])
    
    def get_all_keywords(self):
        """전체 키워드 리스트 반환"""
        return self.political_keywords_2025
    
    def search_keywords_in_text(self, text):
        """텍스트에서 키워드 검색"""
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.political_keywords_2025:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def analyze_keyword_frequency(self, texts):
        """텍스트 리스트에서 키워드 빈도 분석"""
        keyword_freq = {}
        
        for text in tqdm(texts, desc="키워드 빈도 분석"):
            found_keywords = self.search_keywords_in_text(text)
            for keyword in found_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # 빈도순 정렬
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords

class BERTopicEmbeddingGenerator:
    """BERTopic용 임베딩 생성기"""
    
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        BERTopic 임베딩 생성기 초기화
        
        Parameters:
        model_name: 한국어 SBERT 모델명
        """
        self.model_name = model_name
        self.embedding_model = None
        self.document_embeddings = None
        self.keyword_embeddings = None
        
        print(f"임베딩 모델 로드 중: {model_name}")
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # GPU 사용 가능시 GPU로 이동
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to(device)
            
            print("임베딩 모델 로드 완료")
            
        except Exception as e:
            print(f"임베딩 모델 로드 실패: {e}")
            # 대체 모델 사용
            print("대체 모델 로드 시도...")
            self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            print("대체 임베딩 모델 로드 완료")
    
    def generate_document_embeddings(self, texts, batch_size=32, show_progress=True):
        """문서 임베딩 생성"""
        print(f"문서 임베딩 생성 중... (총 {len(texts):,}개 문서)")
        
        try:
            self.document_embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            print(f"문서 임베딩 생성 완료: {self.document_embeddings.shape}")
            return self.document_embeddings
            
        except Exception as e:
            print(f"문서 임베딩 생성 실패: {e}")
            return None
    
    def generate_keyword_embeddings(self, keywords, batch_size=32):
        """키워드 임베딩 생성"""
        print(f"키워드 임베딩 생성 중... (총 {len(keywords)}개 키워드)")
        
        try:
            self.keyword_embeddings = self.embedding_model.encode(
                keywords,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            print(f"키워드 임베딩 생성 완료: {self.keyword_embeddings.shape}")
            return self.keyword_embeddings
            
        except Exception as e:
            print(f"키워드 임베딩 생성 실패: {e}")
            return None
    
    def calculate_similarity_matrix(self, embeddings1, embeddings2=None):
        """임베딩 간 유사도 행렬 계산"""
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        # 코사인 유사도 계산
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        return similarity_matrix
    
    def find_most_similar_keywords(self, document_idx, top_k=10):
        """특정 문서와 가장 유사한 키워드 찾기"""
        if self.document_embeddings is None or self.keyword_embeddings is None:
            print("임베딩이 생성되지 않았습니다.")
            return []
        
        # 문서와 키워드 간 유사도 계산
        doc_embedding = self.document_embeddings[document_idx].reshape(1, -1)
        similarities = np.dot(doc_embedding, self.keyword_embeddings.T).flatten()
        
        # 상위 k개 키워드 인덱스
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def get_embedding_statistics(self):
        """임베딩 통계 정보 반환"""
        stats = {}
        
        if self.document_embeddings is not None:
            stats['document_embeddings'] = {
                'shape': self.document_embeddings.shape,
                'mean': np.mean(self.document_embeddings),
                'std': np.std(self.document_embeddings),
                'min': np.min(self.document_embeddings),
                'max': np.max(self.document_embeddings)
            }
        
        if self.keyword_embeddings is not None:
            stats['keyword_embeddings'] = {
                'shape': self.keyword_embeddings.shape,
                'mean': np.mean(self.keyword_embeddings),
                'std': np.std(self.keyword_embeddings),
                'min': np.min(self.keyword_embeddings),
                'max': np.max(self.keyword_embeddings)
            }
        
        return stats

class BERTopicPreprocessor:
    """BERTopic 전처리 클래스 (키워드 사전 + 임베딩)"""
    
    def __init__(self, text_column='comment_text'):
        self.text_column = text_column
        self.keyword_dict = ElectionKeywordDictionary()
        self.embedding_generator = BERTopicEmbeddingGenerator()
        self.vectorizer = None
        
        # 처리 결과 저장
        self.preprocessing_results = {}
    
    def load_and_preprocess_data(self, csv_files):
        """여러 CSV 파일을 로드하고 전처리"""
        all_texts = []
        file_info = []
        
        for csv_path in csv_files:
            if not os.path.exists(csv_path):
                print(f"파일 없음: {csv_path}")
                continue
            
            try:
                print(f"데이터 로드 중: {csv_path}")
                
                # 날짜 추출
                match = re.search(r"\d{4}_comment_\d{4}_\d{4}", csv_path)
                date_str = match.group(0) if match else "unknown"
                
                # CSV 로드 및 텍스트 추출
                df = pd.read_csv(csv_path, encoding='utf-8')
                texts = df[self.text_column].dropna().drop_duplicates().tolist()
                texts = [text.strip() for text in texts if text.strip()]
                
                print(f"{len(texts):,}개 유효 텍스트 추출")
                
                # 파일 정보 저장
                for text in texts:
                    file_info.append({
                        'file_path': csv_path,
                        'date_range': date_str,
                        'text': text
                    })
                
                all_texts.extend(texts)
                
            except Exception as e:
                print(f"파일 로드 실패 {csv_path}: {e}")
                continue
        
        print(f"총 {len(all_texts):,}개 문서 로드 완료")
        print(f"처리된 파일 수: {len(set([info['file_path'] for info in file_info]))}")
        
        return all_texts, file_info
    
    def setup_keyword_dictionary(self):
        """대선 키워드 사전 설정"""
        print("=== 대선 주요 키워드 사전 설정 ===")
        
        all_keywords = self.keyword_dict.get_all_keywords()
        categories = self.keyword_dict.keyword_categories
        
        print(f"전체 키워드 수: {len(all_keywords)}")
        print(f"카테고리 수: {len(categories)}")
        
        # 카테고리별 키워드 출력
        for category, keywords in categories.items():
            print(f"   {category}: {keywords[:5]}..." if len(keywords) > 5 else f"   {category}: {keywords}")
        
        return all_keywords, categories
    
    def analyze_keyword_distribution(self, texts):
        """텍스트에서 키워드 분포 분석"""
        print("키워드 분포 분석 중...")
        
        # 키워드 빈도 분석
        keyword_freq = self.keyword_dict.analyze_keyword_frequency(texts)
        
        print(f"발견된 키워드 수: {len(keyword_freq)}")
        print(f"상위 10개 키워드:")
        
        for keyword, freq in keyword_freq[:10]:
            print(f"   {keyword}: {freq:,}회")
        
        return keyword_freq
    
    def create_keyword_vectorizer(self, keywords):
        """키워드 기반 벡터라이저 생성"""
        print("키워드 기반 벡터라이저 생성 중...")
        
        self.vectorizer = CountVectorizer(
            vocabulary=keywords,
            lowercase=False,
            token_pattern=r'\b\w+\b'
        )
        
        print(f"벡터라이저 생성 완료 (어휘 크기: {len(keywords)})")
        return self.vectorizer
    
    def generate_embeddings(self, texts, keywords):
        """문서 및 키워드 임베딩 생성"""
        print("=== 임베딩 생성 시작 ===")
        
        # 문서 임베딩 생성
        document_embeddings = self.embedding_generator.generate_document_embeddings(
            texts, batch_size=32, show_progress=True
        )
        
        # 키워드 임베딩 생성
        keyword_embeddings = self.embedding_generator.generate_keyword_embeddings(
            keywords, batch_size=32
        )
        
        # 임베딩 통계 정보
        stats = self.embedding_generator.get_embedding_statistics()
        print(f"임베딩 통계:")
        print(f"   문서 임베딩: {stats.get('document_embeddings', {}).get('shape', 'N/A')}")
        print(f"   키워드 임베딩: {stats.get('keyword_embeddings', {}).get('shape', 'N/A')}")
        
        return document_embeddings, keyword_embeddings, stats
    
    def process_all(self, texts):
        """모든 전처리 과정을 한 번에 실행"""
        print("=== BERTopic 전처리 과정 시작 ===")
        
        # 1. 키워드 사전 설정
        all_keywords, categories = self.setup_keyword_dictionary()
        
        # 2. 키워드 분포 분석
        keyword_freq = self.analyze_keyword_distribution(texts)
        
        # 3. 벡터라이저 생성
        vectorizer = self.create_keyword_vectorizer(all_keywords)
        
        # 4. 임베딩 생성
        document_embeddings, keyword_embeddings, embedding_stats = self.generate_embeddings(
            texts, all_keywords
        )
        
        # 결과 저장
        self.preprocessing_results = {
            'all_keywords': all_keywords,
            'keyword_categories': categories,
            'keyword_frequency': keyword_freq,
            'vectorizer': vectorizer,
            'document_embeddings': document_embeddings,
            'keyword_embeddings': keyword_embeddings,
            'embedding_stats': embedding_stats,
            'num_documents': len(texts),
            'num_keywords': len(all_keywords)
        }
        
        print("=== BERTopic 전처리 과정 완료 ===")
        return self.preprocessing_results

def run_bertopic_preprocessing_pipeline():
    """BERTopic 전처리 파이프라인 실행"""
    print("=== BERTopic 전처리 파이프라인 시작 ===")
    
    # CSV 파일 리스트 정의
    csv_files = [
        "2025_comment_0519_0521.csv",
        "2025_comment_0524_0526.csv", 
        "2025_comment_0528_0530.csv"
    ]
    
    # 존재하는 파일만 필터링
    existing_files = [f for f in csv_files if os.path.exists(f)]
    print(f"처리 가능한 파일: {len(existing_files)}개")
    
    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        return
    
    # 전처리 객체 생성
    preprocessor = BERTopicPreprocessor(text_column='comment_text')
    
    # 1. 데이터 로드
    all_texts, file_info = preprocessor.load_and_preprocess_data(existing_files)
    
    if not all_texts:
        print("유효한 텍스트 데이터가 없습니다.")
        return
    
    # 2. 전체 전처리 과정 실행
    results = preprocessor.process_all(all_texts)
    
    print("\n=== BERTopic 전처리 파이프라인 완료 ===")
    print(f"결과 요약:")
    print(f"   문서 수: {results['num_documents']:,}")
    print(f"   키워드 수: {results['num_keywords']}")
    print(f"   문서 임베딩 차원: {results['document_embeddings'].shape if results['document_embeddings'] is not None else 'N/A'}")
    print(f"   키워드 임베딩 차원: {results['keyword_embeddings'].shape if results['keyword_embeddings'] is not None else 'N/A'}")
    print(f"   발견된 키워드: {len(results['keyword_frequency'])}개")
    
    return preprocessor, results

# 실행 코드
if __name__ == "__main__":
    # 필요한 라이브러리 확인
    try:
        import sentence_transformers
        print("sentence-transformers 라이브러리 확인")
    except ImportError:
        print("sentence-transformers 설치 필요")
        print("pip install sentence-transformers")
        exit()
    
    # 파이프라인 실행
    preprocessor, results = run_bertopic_preprocessing_pipeline()

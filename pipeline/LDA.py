import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models

class LDAPreprocessor:
    def __init__(self, text_column='comment_text'):
        """
        LDA 전처리 클래스 (텍스트 벡터화, Gensim 변환, 코퍼스 생성)
        Parameters:
        text_column: 텍스트가 포함된 컬럼명
        """
        self.text_column = text_column
        self.vectorizer = None
        self.doc_term_matrix = None
        self.feature_names = None
        self.gensim_dictionary = None
        self.gensim_corpus = None
        self.tokenized_texts = None

    def load_and_preprocess_data(self, csv_files):
        """
        여러 CSV 파일을 로드하고 전처리
        Parameters:
        csv_files: CSV 파일 경로 리스트
        """
        all_texts = []
        file_info = []

        for csv_path in csv_files:
            try:
                print(f"데이터 로드 중: {csv_path}")
                df = pd.read_csv(csv_path, encoding='utf-8')

                # 텍스트 데이터 추출 및 전처리
                texts = df[self.text_column].dropna().drop_duplicates().tolist()

                # 빈 문자열 제거
                texts = [text.strip() for text in texts if text.strip()]

                print(f"{len(texts):,}개 유효 텍스트 추출")

                # 파일 정보 저장
                for text in texts:
                    file_info.append({
                        'file_path': csv_path,
                        'text': text
                    })

                all_texts.extend(texts)

            except Exception as e:
                print(f"파일 로드 실패 {csv_path}: {e}")
                continue

        print(f"총 {len(all_texts):,}개 문서 로드 완료")
        return all_texts, file_info

    def vectorize_text(self, texts, min_df=2, max_df=0.95, max_features=1000):
        """
        텍스트 벡터화 (Scikit-learn CountVectorizer 사용)
        Parameters:
        texts: 텍스트 리스트
        min_df: 최소 문서 빈도
        max_df: 최대 문서 빈도 비율
        max_features: 최대 특성 수
        """
        print("텍스트 벡터화 중...")

        # Scikit-learn CountVectorizer
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            token_pattern=r'\b[가-힣]{2,}\b',  # 한국어 2글자 이상
            lowercase=False
        )

        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"벡터화 완료: {self.doc_term_matrix.shape[0]:,}개 문서, {self.doc_term_matrix.shape[1]:,}개 단어")
        
        return self.doc_term_matrix, self.feature_names

    def convert_to_gensim_format(self, texts, min_df=2, max_df=0.95, max_features=1000):
        """
        Gensim 형식으로 변환
        Parameters:
        texts: 텍스트 리스트
        min_df: 최소 문서 빈도
        max_df: 최대 문서 빈도 비율
        max_features: 최대 특성 수
        """
        print("Gensim 형식 변환 중...")

        # 토큰화된 텍스트 생성
        self.tokenized_texts = []
        for text in texts:
            # 간단한 토큰화 (실제로는 더 정교한 전처리 필요)
            tokens = [word for word in text.split() if len(word) >= 2]
            self.tokenized_texts.append(tokens)

        return self.tokenized_texts

    def create_gensim_corpus(self, min_df=2, max_df=0.95, max_features=1000):
        """
        Gensim 사전 및 코퍼스 생성 
        Parameters:
        min_df: 최소 문서 빈도
        max_df: 최대 문서 빈도 비율
        max_features: 최대 특성 수
        """
        if self.tokenized_texts is None:
            raise ValueError("먼저 convert_to_gensim_format()을 실행해주세요.")
            
        print("Gensim 사전 및 코퍼스 생성 중...")

        # Gensim 사전 및 코퍼스 생성
        self.gensim_dictionary = corpora.Dictionary(self.tokenized_texts)
        self.gensim_dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features)
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in self.tokenized_texts]

        print(f"Gensim 사전 크기: {len(self.gensim_dictionary)}")
        print(f"코퍼스 크기: {len(self.gensim_corpus)}")

        return self.gensim_dictionary, self.gensim_corpus

    def process_all(self, texts, min_df=2, max_df=0.95, max_features=1000):
        """
        모든 전처리 과정 실행
        Parameters:
        texts: 텍스트 리스트
        min_df: 최소 문서 빈도
        max_df: 최대 문서 빈도 비율
        max_features: 최대 특성 수
        """
        print("=== 전체 전처리 과정 시작 ===")
        
        # 1. 텍스트 벡터화
        self.vectorize_text(texts, min_df, max_df, max_features)
        
        # 2. Gensim 형식 변환
        self.convert_to_gensim_format(texts, min_df, max_df, max_features)
        
        # 3. 코퍼스 생성
        self.create_gensim_corpus(min_df, max_df, max_features)
        
        print("=== 전체 전처리 과정 완료 ===")
        
        return {
            'doc_term_matrix': self.doc_term_matrix,
            'feature_names': self.feature_names,
            'tokenized_texts': self.tokenized_texts,
            'gensim_dictionary': self.gensim_dictionary,
            'gensim_corpus': self.gensim_corpus
        }

def run_preprocessing_pipeline():
    """전처리 파이프라인 실행"""
    print("=== LDA 전처리 파이프라인 시작 ===")

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
    print(f"처리 가능한 파일: {len(existing_files)}개")

    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        return

    # 전처리 객체 생성
    preprocessor = LDAPreprocessor(text_column='comment_text')

    # 1. 데이터 로드
    all_texts, file_info = preprocessor.load_and_preprocess_data(existing_files)

    if not all_texts:
        print("유효한 텍스트 데이터가 없습니다.")
        return

    # 2. 전체 전처리 과정 실행
    results = preprocessor.process_all(
        all_texts,
        min_df=3,
        max_df=0.8,
        max_features=1000
    )

    print("\n=== 전처리 파이프라인 완료 ===")
    print(f"결과 요약:")
    print(f"   문서 수: {results['doc_term_matrix'].shape[0]:,}")
    print(f"   단어 수: {results['doc_term_matrix'].shape[1]:,}")
    print(f"   Gensim 사전 크기: {len(results['gensim_dictionary'])}")

    return preprocessor, results

# 실행 코드
if __name__ == "__main__":
    preprocessor, results = run_preprocessing_pipeline()

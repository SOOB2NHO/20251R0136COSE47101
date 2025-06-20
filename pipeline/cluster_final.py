import os
import re
import pandas as pd
import numpy as np
import torch
from itertools import product
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

# 처리할 CSV 목록
csv_files = [
    "/content/2025_comment_0519_0521.csv",
    "/content/2025_comment_0524_0526.csv",
    "/content/2025_comment_0528_0530.csv"
]

# KcELECTRA SentenceTransformer 모델 로드
hf_model = "beomi/KcELECTRA-base"
word_embedding_model = models.Transformer(hf_model, max_seq_length=128)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda" if torch.cuda.is_available() else "cpu")

# 후보자 리스트
candidates = ["이재명", "김문수", "이준석"]
candidate_embs = model.encode(candidates, convert_to_numpy=True)

# 파일별 처리
for csv_path in csv_files:
    print(f"\n📂 Processing: {csv_path}")
    filename = os.path.basename(csv_path)
    date_match = re.search(r"\d{4}_\d{4}", filename)  # ← 날짜 패턴 수정
    if not date_match:
        print("❌ 날짜 패턴 인식 실패")
        continue
    date_str = date_match.group()

    # 키워드 파일 로드
    txt_path = f"/content/{date_str}.txt"
    if not os.path.exists(txt_path):
        print(f"❌ 키워드 파일 없음: {txt_path}")
        continue
    with open(txt_path, 'r', encoding='utf-8') as f:
        keywords = [line.strip().split(":")[-1].strip() for line in f if ":" in line]

    # 댓글 데이터 로드
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df.dropna(subset=['comment_text']).reset_index(drop=True)
    sentences = df['comment_text'].astype(str).tolist()

    # 댓글 임베딩
    embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # 키워드 임베딩 및 1차 클러스터
    keyword_embeddings = model.encode(keywords, batch_size=4, convert_to_numpy=True)
    sims = cosine_similarity(embeddings, keyword_embeddings)
    assigned = np.argmax(sims, axis=1)
    df['candidate_cluster'] = [keywords[i] for i in assigned]

    # 2차 클러스터링 (사람 이름은 그대로, 나머지는 _정치인명 추가)
    final_clusters = []
    for i, row in df.iterrows():
        topic = row['candidate_cluster']
        if topic in candidates:
            final_clusters.append(topic)
        else:
            emb = embeddings[i].reshape(1, -1)
            cand_sim = cosine_similarity(emb, candidate_embs)
            best_cand = candidates[np.argmax(cand_sim)]
            final_clusters.append(f"{topic}_{best_cand}")
    df['final_cluster'] = final_clusters

    # 결과 저장
    output_path = f"/content/clustered_{date_str}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {output_path}")

import glob

# 분석 대상 파일 목록 (원하는 범위로 패턴 수정 가능)
clustered_files = glob.glob("/content/clustered_*.csv")

for path in clustered_files:
    print(f"\n📊 [분포 분석] 파일: {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')

    # 클러스터 분포 출력
    if 'candidate_cluster' in df.columns:
        cluster_counts = df['final_cluster'].value_counts().sort_values(ascending=False)
        print(cluster_counts)
    else:
        print("⚠️ 'candidate_cluster' 열이 존재하지 않습니다.")
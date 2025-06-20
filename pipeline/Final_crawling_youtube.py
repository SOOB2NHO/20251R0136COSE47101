import pandas as pd
import numpy as np

# ———————————————————————————————————————————————————————————
# 2) 최적화된 봇 필터링 함수 (작성자별 그룹핑, 민감도 ↑)
# ———————————————————————————————————————————————————————————
def optimized_filter_bot_comments(df):
    print("=== 최적화된 봇 댓글 필터링 시작 (높은 민감도) ===")
    df = df.copy()

    # 1) timestamp, 길이, 단어 반복율 전처리
    df['ts'] = pd.to_datetime(df['comment_published'], errors='coerce')
    df.dropna(subset=['ts'], inplace=True)
    df.sort_values(['comment_author_id','ts'], inplace=True)
    df['length'] = df['comment_text'].fillna('').str.len()
    df['repetition_ratio'] = (
        df['comment_text'].fillna('')
          .str.split()
          .map(lambda w: 1 - len(set(w)) / len(w) if len(w)>0 else 0)
    )

    # 2) 시간 간격(diff) 계산
    df['interval'] = df.groupby('comment_author_id')['ts'].diff().dt.total_seconds()

    # 3) 사용자별 통계량 집계
    stats = df.groupby('comment_author_id').agg(
        comment_count = ('comment_text','size'),
        time_var_sec  = ('interval', lambda x: np.var(x.dropna()) if len(x.dropna())>=2 else np.nan),
        length_var    = ('length','var'),
        avg_rep_ratio = ('repetition_ratio','mean')
    )
    stats['time_var_hr'] = stats['time_var_sec'] / 3600

    # 4) AI 생성 의심 확률 벡터화 (휴리스틱)
    def ai_prob_series(texts):
        s = texts.fillna('').astype(str)
        lens       = s.str.split('.').map(lambda ss: [len(x) for x in ss if x])
        var_len    = lens.map(lambda L: np.var(L) if len(L)>1 else 0)
        punct_cnt  = s.str.count(r'[.!?]')
        sent_cnt   = s.str.count(r'\.') + 1
        punct_ratio= punct_cnt / sent_cnt
        emo_ind    = s.str.contains('ㅋㅋ|ㅎㅎ|ㅠㅠ|!!!|\?\?\?')
        return (
            (var_len < 50).astype(float) +
            (punct_ratio > 0.8).astype(float) +
            (~emo_ind & (s.str.len() > 50)).astype(float)
        ) / 3.0

    df['ai_prob'] = ai_prob_series(df['comment_text'])
    ai_stats     = df.groupby('comment_author_id')['ai_prob'].mean().rename('avg_ai_prob')
    stats        = stats.join(ai_stats, how='left').fillna(0)

    # 5) 플래그 조건
    stats['cond_speed']      = ((stats['comment_count'] >= 5) & (stats['time_var_hr'] < 0.2)).astype(int)
    stats['cond_ai']         = (stats['avg_ai_prob'] > 0.5).astype(int)
    stats['suspicious_score']= (stats['length_var'] < 200).astype(float)*0.2 \
                               + (stats['avg_rep_ratio'] > 0.6).astype(float)*0.3
    stats['cond_suspicious'] = (stats['suspicious_score'] > 0.3).astype(int)
    stats['high_rep']        = (stats['avg_rep_ratio'] > 0.6).astype(int)
    stats['low_len_var']     = (stats['length_var'] < 200).astype(int)

    # 6) 총 6개 플래그 중 ≥3개 만족 시 봇으로 판정
    stats['is_bot'] = (
        stats[['cond_speed','cond_ai','cond_suspicious','high_rep','low_len_var']]
        .sum(axis=1) >= 3
    )

    # 7) 필터링
    bot_users = stats.index[stats['is_bot']]
    filtered  = df[~df['comment_author_id'].isin(bot_users)].drop(
        ['ts','length','repetition_ratio','interval','ai_prob'], axis=1
    )

    print(f"원본 댓글 수: {len(df):,} → 필터링 후: {len(filtered):,}")
    return filtered.reset_index(drop=True), stats.reset_index()

if __name__ == '__main__':
    INPUT_CSV = '/Users/hosubin/Downloads/combined_youtube_news_0519_to_0530.csv'
    OUT_COMMS = 'youtube_filtered_comments_combined_youtube_news_0519_to_0530.csv'
    OUT_STATS = 'youtube_bot_analysis_stats_combined_youtube_news_0519_to_0530.csv'

    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    print(f"Loaded {len(df):,} comments from {INPUT_CSV}")

    filtered_df, stats_df = optimized_filter_bot_comments(df)

    filtered_df.to_csv(OUT_COMMS, index=False, encoding='utf-8-sig')
    stats_df.to_csv(OUT_STATS, index=False, encoding='utf-8-sig')
    print("✅ 필터링 및 통계 저장 완료")
    print(f"  • Filtered comments → {OUT_COMMS}")
    print(f"  • Bot stats         → {OUT_STATS}")
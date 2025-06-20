import pandas as pd
import numpy as np
import requests
import json
import html
import time

# ———————————————————————————————————————————————————————————
# 1) 댓글 크롤링 함수 (작성자 ID, 작성 시간, 내용까지 수집)
# ———————————————————————————————————————————————————————————
def crawl_comments(url, template="default_society", pool="cbox5", page_size=20):
    oid = url.split("article/")[1].split("/")[0]
    aid = url.split("article/")[1].split("/")[1].split("?")[0]
    page = 1
    rows = []
    headers = {"User-Agent": "Mozilla/5.0", "Referer": url}

    while True:
        callback = f"jQuery{int(time.time()*1000)}"
        api_url = (
            "https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json"
            f"?ticket=news&templateId={template}&pool={pool}"
            f"&_callback={callback}&lang=ko&country=KR"
            f"&objectId=news{oid}%2C{aid}"
            f"&pageSize={page_size}&indexSize=10&listType=OBJECT&pageType=more"
            f"&page={page}&sort=FAVORITE"
        )
        res = requests.get(api_url, headers=headers)
        text = res.text
        json_str = text[text.find('(')+1 : text.rfind(')')]
        data = json.loads(json_str)

        comment_list = data.get("result", {}).get("commentList", [])
        if not comment_list:
            break

        for c in comment_list:
            rows.append({
                "url": url,
                "comment_author_id": c.get("maskedUserId"),
                "comment_published": c.get("modTime", c.get("createTime")),
                "comment_text": html.unescape(c.get("contents", ""))
            })

        total = data.get("result", {}).get("totalCommentCount", 0)
        if page * page_size >= total:
            break

        page += 1
        time.sleep(0.3)

    return pd.DataFrame(rows)


# ———————————————————————————————————————————————————————————
# 2) 최적화된 봇 필터링 함수 (작성자별 그룹핑, 민감도 ↑)
# ———————————————————————————————————————————————————————————
def optimized_filter_bot_comments(df):
    print("=== 최적화된 봇 댓글 필터링 시작 (높은 민감도) ===")
    df = df.copy()

    # 1) timestamp, 길이, 단어 반복율 전처리
    df['ts'] = pd.to_datetime(df['comment_published'])
    df.sort_values(['comment_author_id','ts'], inplace=True)
    df['length'] = df['comment_text'].str.len()
    df['repetition_ratio'] = (
        df['comment_text']
          .str.split()
          .map(lambda w: 1 - len(set(w)) / len(w) if len(w)>0 else 0)
    )

    # 2) 시간 간격(diff) 계산
    df['interval'] = df.groupby('comment_author_id')['ts'] \
                       .diff() \
                       .dt.total_seconds()

    # 3) 사용자별 통계량 집계
    stats = df.groupby('comment_author_id').agg(
        comment_count=('comment_text','size'),
        time_var_sec=('interval', lambda x: np.var(x.dropna()) if len(x.dropna())>=2 else np.nan),
        length_var=('length', 'var'),
        avg_rep_ratio=('repetition_ratio', 'mean')
    )
    stats['time_var_hr'] = stats['time_var_sec'] / 3600

    # 4) AI 생성 의심 확률 벡터화 (기존 휴리스틱)
    def ai_prob_series(texts: pd.Series) -> pd.Series:
        s = texts.fillna('').astype(str)
        sent_lens = s.str.split('.').map(lambda ss: [len(x) for x in ss if x])
        var_len = sent_lens.map(lambda L: np.var(L) if len(L)>1 else np.nan)
        punct_cnt = s.str.count(r'[.!?]')
        sent_cnt  = s.str.count(r'\.') + 1
        punct_ratio = punct_cnt / sent_cnt
        emo_ind = s.str.contains('ㅋㅋ|ㅎㅎ|ㅠㅠ|!!!|\?\?\?')
        return (
            (var_len < 50).fillna(0).astype(float) +
            (punct_ratio > 0.8).astype(float) +
            (~emo_ind & (s.str.len()>50)).astype(float)
        ) / 3.0

    df['ai_prob'] = ai_prob_series(df['comment_text'])
    ai_stats = df.groupby('comment_author_id')['ai_prob'].mean().rename('avg_ai_prob')

    # 5) 통합 & 봇 판정 (민감도 ↑)
    stats = stats.join(ai_stats, how='left').fillna(0)

    # 1) 빠른 속도: 댓글 ≥5개 & 시간 분산(hr) < 0.2
    stats['cond_speed'] = (
        (stats['comment_count'] >= 5) &
        (stats['time_var_hr'] < 0.2)
    ).astype(int)

    # 2) AI 생성 의심: avg_ai_prob > 0.5
    stats['cond_ai'] = (stats['avg_ai_prob'] > 0.5).astype(int)

    # 3) 휴리스틱 의심: suspicious_score > 0.3
    stats['suspicious_score'] = (
        (stats['length_var'] < 200).astype(float) * 0.2 +
        (stats['avg_rep_ratio'] > 0.6).astype(float) * 0.3
    )
    stats['cond_suspicious'] = (stats['suspicious_score'] > 0.3).astype(int)

    # 4) 단어 반복율 과다: avg_rep_ratio > 0.6
    stats['high_rep'] = (stats['avg_rep_ratio'] > 0.6).astype(int)

    # 5) 댓글 길이 단조: length_var < 200
    stats['low_len_var'] = (stats['length_var'] < 200).astype(int)

    # **총 5개 지표 중 ≥2개** 만족 시 봇
    stats['is_bot'] = (
        stats['cond_speed'] +
        stats['cond_ai'] +
        stats['cond_suspicious'] +
        stats['high_rep'] +
        stats['low_len_var']
    ) >= 2

    bot_users = stats.index[stats['is_bot']].tolist()
    filtered = df[~df['comment_author_id'].isin(bot_users)].drop(
        ['ts','length','repetition_ratio','interval','ai_prob'], axis=1
    )

    print(f"원본: {len(df):,}개 → 필터링 후: {len(filtered):,}개")
    return filtered, stats.reset_index()


# ———————————————————————————————————————————————————————————
# 3) 메인 실행
# ———————————————————————————————————————————————————————————
if __name__ == "__main__":
    # (1) 원본 엑셀에서 URL 불러오기
    excel_path = "/Users/hosubin/Desktop/ubuntu_data/Data Science/21대 대선/기사/all_news_이재명, 이준석, 김문수, 대선_20250519~20250521.xlsx"
    df_urls    = pd.read_excel(excel_path)
    urls       = df_urls['Naverlink'] if 'Naverlink' in df_urls else df_urls['URL']

    # (2) 전체 댓글 크롤링
    all_comments = []
    for idx, url in enumerate(urls.dropna(), 1):
        print(f"[{idx}/{len(urls)}] 댓글 수집: {url}")
        try:
            df_c = crawl_comments(url)
            all_comments.append(df_c)
        except Exception as e:
            print("  오류:", e)
            continue

    comments_df = pd.concat(all_comments, ignore_index=True)

    # (3) 봇 필터링
    filtered_df, stats_df = optimized_filter_bot_comments(comments_df)

    # (4) 결과 저장
    filtered_df.to_csv('filtered_comments_이재명, 이준석, 김문수, 대선_20250519~20250521.csv', index=False, encoding='utf-8-sig')
    stats_df.to_csv('bot_analysis_stats_이재명, 이준석, 김문수, 대선_20250519~20250521.csv', index=False, encoding='utf-8-sig')

    print("✅ 완료! filtered_comments.csv 와 bot_analysis_stats.csv 생성되었습니다.")
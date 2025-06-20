# # main.py - 네이버 API로 뉴스 URL 수집 후 댓글 수집 (Selenium 없이 API 방식)

# import pandas as pd
# import requests
# import re
# import time

# # ✅ 네이버 API 인증 정보
# CLIENT_ID = "DkqIEaI_ltBe70Xdm4W6"  # 본인의 API 키 입력
# CLIENT_SECRET = "BAA_gdcR17"
# HEADERS = {
#     "X-Naver-Client-Id": CLIENT_ID,
#     "X-Naver-Client-Secret": CLIENT_SECRET
# }
# NAVER_NEWS_API = "https://openapi.naver.com/v1/search/news.json"

# # ✅ 기사 URL 수집 함수
# def get_news_urls(query="이재명|김문수|대선", max_articles=2000):
#     collected_data = []
#     for start in range(1, max_articles + 1, 100):
#         display = min(100, max_articles - len(collected_data))
#         params = {
#             "query": query,
#             "display": display,
#             "start": start,
#             "sort": "date"
#         }
#         res = requests.get(NAVER_NEWS_API, headers=HEADERS, params=params)
#         data = res.json()
#         for item in data.get("items", []):
#             title = re.sub(r"<.*?>", "", item["title"])
#             link = item["link"]
#             pub_date = item["pubDate"]
#             if "n.news.naver.com" in link:
#                 collected_data.append([title, link, pub_date])
#         if len(data.get("items", [])) < 100:
#             break
#         time.sleep(0.5)
#     return collected_data

# # ✅ 수집된 기사 저장 및 댓글 수집 시작
# news_output_csv = "filtered_naver_news.csv"
# news_data = get_news_urls(query="21대 대선", max_articles=2000)
# df_news = pd.DataFrame(news_data, columns=["제목", "기사URL", "작성일"])
# df_news.to_csv(news_output_csv, index=False, encoding='utf-8-sig')
# print(f"📰 기사 URL {len(df_news)}개 저장 완료: {news_output_csv}")

# # ✅ 댓글 수집 대상 파일 경로
# csv_path = news_output_csv
# output_path = "naver_comments_result.csv"

# # ✅ CSV 로드 및 URL 컬럼 선택
# df = pd.read_csv(csv_path)
# if 'Naverlink' in df.columns:
#     urls = df['Naverlink']
# elif '기사URL' in df.columns:
#     urls = df['기사URL']
# else:
#     raise ValueError("CSV 파일에 'Naverlink' 또는 '기사URL' 컬럼이 없습니다.")

# # ✅ 댓글 리스트 평탄화 함수
# def flatten(l):
#     return [item for sublist in l for item in (sublist if isinstance(sublist, list) else [sublist])]

# # ✅ 전체 댓글 저장 리스트 초기화
# total_comments = []

# # ✅ 각 기사별 댓글 수집 반복
# for idx, url in enumerate(urls):
#     try:
#         print(f"{idx+1}/{len(urls)}번째 기사 댓글 수집 중: {url}")

#         oid = url.split("article/")[1].split("/")[0]
#         aid = url.split("article/")[1].split("/")[1].split("?")[0]

#         page = 1
#         all_comments = []

#         header = {
#             "User-agent": "Mozilla/5.0",
#             "referer": url
#         }

#         while True:
#             callback_id = f"jQuery1123{int(time.time() * 1000)}"
#             c_url = (
#                 f"https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json"
#                 f"?ticket=news&templateId=default_society&pool=cbox5&_callback={callback_id}"
#                 f"&lang=ko&country=KR&objectId=news{oid}%2C{aid}&pageSize=20&indexSize=10"
#                 f"&listType=OBJECT&pageType=more&page={page}&sort=FAVORITE"
#             )

#             res = requests.get(c_url, headers=header)
#             html = res.text

#             if 'comment":' not in html:
#                 print("🚫 댓글 없음 또는 JSON 구조 이상")
#                 break

#             try:
#                 total_comm = int(html.split('comment":')[1].split(",")[0])
#             except Exception as e:
#                 print("🚫 댓글 수 파싱 오류:", e)
#                 break

#             matches = re.findall(r'"contents":"(.*?)","userIdNo"', html)
#             if not matches:
#                 break

#             all_comments.extend(matches)

#             if page * 20 >= total_comm:
#                 break
#             page += 1
#             time.sleep(0.1)

#         total_comments.append({
#             "url": url,
#             "댓글 수": len(all_comments),
#             "댓글": flatten(all_comments)
#         })

#     except Exception as e:
#         print(f"⛔ 오류 발생: {e}")
#         total_comments.append({
#             "url": url,
#             "댓글 수": 0,
#             "댓글": []
#         })

# # ✅ 결과를 CSV로 저장
# df_result = pd.DataFrame(total_comments)
# df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
# print(f"✅ 저장 완료: {output_path}")

# import requests
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from openpyxl import Workbook, load_workbook
# from datetime import datetime, timedelta
# import time
# import os

# # 키워드 리스트
# keywords = ["이준석, 이재명, 김문수, 대선"]
# start_date_str = "2025.05.19"
# end_date_str = "2025.05.21"
# max_articles_per_keyword = 6000  # 수집 제한

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
# }

# # 셀레니움 설정
# options = Options()
# options.add_argument("--headless")
# options.add_argument("--disable-gpu")
# options.add_argument("window-size=1920x1080")
# driver = webdriver.Chrome(options=options)

# def str_to_date(s):
#     return datetime.strptime(s, "%Y.%m.%d")

# # 기사 본문 파싱 함수 (내용 제외)
# def fetch_article_content(url):
#     def parse_soup(soup):
#         title = (
#             soup.select_one(".media_end_head_headline") or
#             soup.find("h2") or
#             soup.find("div", class_="news_title") or
#             soup.find("div", class_="end_tit")
#         )
#         date = (
#             soup.select_one("span.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME") or
#             soup.find("em", class_="date") or
#             soup.find("span", class_="date") or
#             soup.find("div", class_="date")
#         )
#         if title and date:
#             return title.text.strip(), date.text.strip()
#         return None, None

#     try:
#         res = requests.get(url, headers=headers, timeout=5)
#         soup = BeautifulSoup(res.text, "html.parser")
#         result = parse_soup(soup)
#         if all(result):
#             return result
#     except:
#         pass

#     try:
#         driver.get(url)
#         time.sleep(2)
#         soup = BeautifulSoup(driver.page_source, "html.parser")
#         return parse_soup(soup)
#     except:
#         return None, None

# # ✅ 뉴스 링크 수집 함수
# def collect_news_links(search_url):
#     try:
#         res = requests.get(search_url, headers=headers, timeout=5)
#         soup = BeautifulSoup(res.text, "html.parser")
#         links = list({a['href'] for a in soup.select(
#             "a[href^='https://n.news.naver.com'], a[href^='https://m.entertain.naver.com']") if a.get('href')})
#         if len(links) >= 5:
#             return links
#     except:
#         pass
#     try:
#         driver.get(search_url)
#         time.sleep(2)
#         soup = BeautifulSoup(driver.page_source, "html.parser")
#         links = list({a['href'] for a in soup.select(
#             "a[href^='https://n.news.naver.com'], a[href^='https://m.entertain.naver.com']") if a.get('href')})
#         return links
#     except:
#         return []

# # ✅ 키워드별 전체 크롤링 루프
# for keyword in keywords:
#     print(f"\n🔍 키워드 [{keyword}] 크롤링 시작")
#     file_name = f"all_news_{keyword}_20250519~20250521.xlsx"

#     if os.path.exists(file_name):
#         wb = load_workbook(file_name)
#         ws = wb.active
#     else:
#         wb = Workbook()
#         ws = wb.active
#         ws.title = "뉴스기사"
#         ws.append(['제목', '날짜', 'URL'])

#     ws.column_dimensions['A'].width = 60
#     ws.column_dimensions['B'].width = 30
#     ws.column_dimensions['C'].width = 60

#     visited_links = set()
#     collected = 0
#     current_date = str_to_date(start_date_str)
#     end_date = str_to_date(end_date_str)

#     while current_date <= end_date and collected < max_articles_per_keyword:
#         ds = current_date.strftime("%Y.%m.%d")
#         de = ds
#         print(f"----- 📅 {ds} 크롤링 중 -----")
#         fail_count = 0
#         max_fails = 5

#         for start in range(1, 2000, 10):
#             if collected >= max_articles_per_keyword:
#                 break

#             search_url = (
#                 f"https://search.naver.com/search.naver?where=news&query={keyword}"
#                 f"&sm=tab_opt&sort=2&photo=0&field=0&pd=3&ds={ds}&de={de}&start={start}"
#             )
#             news_links = collect_news_links(search_url)

#             if not news_links:
#                 fail_count += 1
#                 print(f"⚠️ 뉴스 없음 (실패 {fail_count}/{max_fails})")
#                 if fail_count >= max_fails:
#                     print("⛔ 연속 실패 횟수 초과, 종료")
#                     break
#                 continue
#             else:
#                 fail_count = 0

#             for href in news_links:
#                 if href in visited_links or collected >= max_articles_per_keyword:
#                     continue
#                 visited_links.add(href)

#                 title, date = fetch_article_content(href)
#                 if title and date:
#                     collected += 1
#                     print(f"✨ ({collected}) {date} - {title}")
#                     ws.append([title, date, href])
#                 else:
#                     print("⚠️ 본문 누락:", href)

#             time.sleep(1)

#         current_date += timedelta(days=1)

#     wb.save(file_name)
#     print(f"\n✅ 저장 완료: {file_name}")

# driver.quit()

# import pandas as pd
# import numpy as np
# import requests
# import json
# import html
# import time

# # ———————————————————————————————————————————————————————————
# # 1) 댓글 크롤링 함수 (작성자 ID, 작성 시간, 내용까지 수집)
# # ———————————————————————————————————————————————————————————
# def crawl_comments(url, template="default_society", pool="cbox5", page_size=20):
#     oid = url.split("article/")[1].split("/")[0]
#     aid = url.split("article/")[1].split("/")[1].split("?")[0]
#     page = 1
#     rows = []
#     headers = {"User-Agent": "Mozilla/5.0", "Referer": url}

#     while True:
#         callback = f"jQuery{int(time.time()*1000)}"
#         api_url = (
#             "https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json"
#             f"?ticket=news&templateId={template}&pool={pool}"
#             f"&_callback={callback}&lang=ko&country=KR"
#             f"&objectId=news{oid}%2C{aid}"
#             f"&pageSize={page_size}&indexSize=10&listType=OBJECT&pageType=more"
#             f"&page={page}&sort=FAVORITE"
#         )
#         res = requests.get(api_url, headers=headers)
#         text = res.text
#         json_str = text[text.find('(')+1 : text.rfind(')')]
#         data = json.loads(json_str)

#         comment_list = data.get("result", {}).get("commentList", [])
#         if not comment_list:
#             break

#         for c in comment_list:
#             rows.append({
#                 "url": url,
#                 "comment_author_id": c.get("maskedUserId"),
#                 "comment_published": c.get("modTime", c.get("createTime")),
#                 "comment_text": html.unescape(c.get("contents", ""))
#             })

#         total = data.get("result", {}).get("totalCommentCount", 0)
#         if page * page_size >= total:
#             break

#         page += 1
#         time.sleep(0.3)

#     return pd.DataFrame(rows)


# # ———————————————————————————————————————————————————————————
# # 2) 최적화된 봇 필터링 함수 (작성자별 그룹핑, 민감도 ↑)
# # ———————————————————————————————————————————————————————————
# def optimized_filter_bot_comments(df):
#     print("=== 최적화된 봇 댓글 필터링 시작 (높은 민감도) ===")
#     df = df.copy()

#     # 1) timestamp, 길이, 단어 반복율 전처리
#     df['ts'] = pd.to_datetime(df['comment_published'])
#     df.sort_values(['comment_author_id','ts'], inplace=True)
#     df['length'] = df['comment_text'].str.len()
#     df['repetition_ratio'] = (
#         df['comment_text']
#           .str.split()
#           .map(lambda w: 1 - len(set(w)) / len(w) if len(w)>0 else 0)
#     )

#     # 2) 시간 간격(diff) 계산
#     df['interval'] = df.groupby('comment_author_id')['ts'] \
#                        .diff() \
#                        .dt.total_seconds()

#     # 3) 사용자별 통계량 집계
#     stats = df.groupby('comment_author_id').agg(
#         comment_count=('comment_text','size'),
#         time_var_sec=('interval', lambda x: np.var(x.dropna()) if len(x.dropna())>=2 else np.nan),
#         length_var=('length', 'var'),
#         avg_rep_ratio=('repetition_ratio', 'mean')
#     )
#     stats['time_var_hr'] = stats['time_var_sec'] / 3600

#     # 4) AI 생성 의심 확률 벡터화 (기존 휴리스틱)
#     def ai_prob_series(texts: pd.Series) -> pd.Series:
#         s = texts.fillna('').astype(str)
#         sent_lens = s.str.split('.').map(lambda ss: [len(x) for x in ss if x])
#         var_len = sent_lens.map(lambda L: np.var(L) if len(L)>1 else np.nan)
#         punct_cnt = s.str.count(r'[.!?]')
#         sent_cnt  = s.str.count(r'\.') + 1
#         punct_ratio = punct_cnt / sent_cnt
#         emo_ind = s.str.contains('ㅋㅋ|ㅎㅎ|ㅠㅠ|!!!|\?\?\?')
#         return (
#             (var_len < 50).fillna(0).astype(float) +
#             (punct_ratio > 0.8).astype(float) +
#             (~emo_ind & (s.str.len()>50)).astype(float)
#         ) / 3.0

#     df['ai_prob'] = ai_prob_series(df['comment_text'])
#     ai_stats = df.groupby('comment_author_id')['ai_prob'].mean().rename('avg_ai_prob')

#     # 5) 통합 & 봇 판정 (민감도 ↑)
#     stats = stats.join(ai_stats, how='left').fillna(0)

#     # 1) 빠른 속도: 댓글 ≥5개 & 시간 분산(hr) <0.2
#     stats['cond_speed'] = (
#         (stats['comment_count'] >= 5) &
#         (stats['time_var_hr'] < 0.2)
#     ).astype(int)

#     # 2) AI 생성 의심: avg_ai_prob > 0.5
#     stats['cond_ai'] = (stats['avg_ai_prob'] > 0.5).astype(int)

#     # 3) 휴리스틱 의심: suspicious_score > 0.3
#     stats['suspicious_score'] = (
#         (stats['length_var'] < 200).astype(float) * 0.2 +
#         (stats['avg_rep_ratio'] > 0.6).astype(float) * 0.3
#     )
#     stats['cond_suspicious'] = (stats['suspicious_score'] > 0.3).astype(int)

#     # 4) 단어 반복율 과다: avg_rep_ratio > 0.6
#     stats['high_rep'] = (stats['avg_rep_ratio'] > 0.6).astype(int)

#     # 5) 댓글 길이 단조: length_var < 200
#     stats['low_len_var'] = (stats['length_var'] < 200).astype(int)

#     # **총 5개 지표 중 ≥2개** 만족 시 봇
#     stats['is_bot'] = (
#         stats['cond_speed'] +
#         stats['cond_ai'] +
#         stats['cond_suspicious'] +
#         stats['high_rep'] +
#         stats['low_len_var']
#     ) >= 2

#     bot_users = stats.index[stats['is_bot']].tolist()
#     filtered = df[~df['comment_author_id'].isin(bot_users)].drop(
#         ['ts','length','repetition_ratio','interval','ai_prob'], axis=1
#     )

#     print(f"원본: {len(df):,}개 → 필터링 후: {len(filtered):,}개")
#     return filtered, stats.reset_index()


# # ———————————————————————————————————————————————————————————
# # 3) 메인 실행
# # ———————————————————————————————————————————————————————————
# if __name__ == "__main__":
#     # (1) 원본 엑셀에서 URL 불러오기
#     excel_path = "all_news_안철수_20250522~20250524.xlsx"
#     df_urls    = pd.read_excel(excel_path)
#     urls       = df_urls['Naverlink'] if 'Naverlink' in df_urls else df_urls['URL']

#     # (2) 전체 댓글 크롤링
#     all_comments = []
#     for idx, url in enumerate(urls.dropna(), 1):
#         print(f"[{idx}/{len(urls)}] 댓글 수집: {url}")
#         try:
#             df_c = crawl_comments(url)
#             all_comments.append(df_c)
#         except Exception as e:
#             print("  오류:", e)
#             continue

#     comments_df = pd.concat(all_comments, ignore_index=True)

#     # (3) 봇 필터링
#     filtered_df, stats_df = optimized_filter_bot_comments(comments_df)

#     # (4) 결과 저장
#     filtered_df.to_csv('filtered_comments_안철수.csv', index=False, encoding='utf-8-sig')
#     stats_df.to_csv('bot_analysis_stats_안철수.csv', index=False, encoding='utf-8-sig')

#     print("✅ 완료! filtered_comments.csv 와 bot_analysis_stats.csv 생성되었습니다.")

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

# ——————————————————————————
# 설정
# ——————————————————————————
API_KEYS = [
    "AIzaSyBn1v3HjJWhqh_YHyF6sJkV41cOyLabemI", # 1
    "AIzaSyAg6WNmdx7MYigHbrA1TMf6cO89tuuqSX8", # 2
    "AIzaSyBujNpcCzdVU4QBstIIXNR2yc85WxYWdHQ", # 3
    "AIzaSyClz3cRUGIoe7ql0-KAP2b-tTIcONJkOzo", # 4
    "AIzaSyD80A71_82-_WTDVYruqPBTG0fr034ajBk", # 5
    "AIzaSyDFhQRYh_cZ6lAgjRgFQYl9BzaTE58Qh1Q", # 6
    "AIzaSyCmZxMmIIyBNfw6tR_5-xQUdlPMlVHU-_8", # 7
    "AIzaSyBh265b37BHH_0j62feHIcqXH7QYDaV-Ag", # 8
    "AIzaSyCWPFZYM197mcRNCFac5klW-WPhLjbQk74", # 9
    "AIzaSyDMRehGlKMt-Dy2czwz6DbAco18hL_qdR8", # 10
]

CHANNEL_IDS = [
    "UCugbqfMO94F9guLEb6Olb2A",  # 한겨레
    "UCF4Wxdo3inmxP-Y59wXDsFw",  # MBC
    "UCHXvjavEtkPFJCfGlm0wTXw",  # 경향
    "UCnHyx6H7fhKfUoAAaVGYvxQ",  # 동아
    "UCWlV3Lz_55UaX4JsMj-z__Q",  # 조선
    "UCH3mJ-nHxjjny2FJbJaqiDA"   # 중앙
]

KEYWORDS = ["대선", "김문수", "이재명", "이준석"]
START_DATE = "2025-05-19"
END_DATE   = "2025-05-21"
TOTAL_LIMIT = 100_000
PER_CHANNEL_LIMIT = TOTAL_LIMIT // len(CHANNEL_IDS)
POLITICS_CATEGORY = "25"  # News & Politics

# ——————————————————————————
# 전역 변수 초기화
# ——————————————————————————
_exhausted = set()
_services = {}
results = []
counts = {cid: 0 for cid in CHANNEL_IDS}
video_queues = {cid: deque() for cid in CHANNEL_IDS}

# ——————————————————————————
# 헬퍼 함수
# ——————————————————————————
def get_svc(key):
    if key not in _services:
        _services[key] = build("youtube", "v3", developerKey=key)
    return _services[key]

def rotate_call(fn, *args, **kwargs):
    for k in API_KEYS:
        if k in _exhausted: continue
        try:
            return fn(get_svc(k), *args, **kwargs)
        except HttpError as e:
            if e.resp.status == 403 and "quota" in str(e).lower():
                print(f"⚠️ {k} quotaExceeded → 제외")
                _exhausted.add(k)
                continue
            else:
                raise
    raise RuntimeError("모든 키 소진됨")

def search_videos(svc, channelId, q, start, end, pageToken=None, category=None):
    params = {
        "part": "id",
        "channelId": channelId,
        "type": "video",
        "publishedAfter": start + "Z",
        "publishedBefore": end + "Z",
        "maxResults": 50,
        "pageToken": pageToken,
        "fields": "nextPageToken,items(id/videoId)"
    }
    if q:
        params["q"] = q
    if category:
        params["videoCategoryId"] = category
    return svc.search().list(**params).execute()

def fetch_video_info(svc, videoId):
    return svc.videos().list(
        part="snippet,statistics",
        id=videoId,
        fields="items(snippet(title,channelTitle,publishedAt,categoryId,tags),"
               "statistics(viewCount,likeCount,commentCount))"
    ).execute()

def fetch_comments_with_replies(svc, videoId, pageToken=None):
    """
    top-level 댓글과, 
   그 댓글 작성자를 멘션(@작성자ID)한 답글만 함께 가져옵니다.
    """
    resp = svc.commentThreads().list(
        part="snippet,replies",
        videoId=videoId,
        maxResults=100,
        pageToken=pageToken,
        textFormat="plainText",
        fields="nextPageToken,items(snippet/topLevelComment/id,snippet/topLevelComment/snippet(authorChannelId/value,textDisplay,publishedAt,likeCount),replies/comments(id,snippet(authorChannelId/value,textDisplay,publishedAt)))"
    ).execute()

    out = []
    for thread in resp.get("items", []):
        top = thread["snippet"]["topLevelComment"]["snippet"]
        top_id     = thread["snippet"]["topLevelComment"]["id"]
        top_auth   = top.get("authorChannelId", {}).get("value", "")
        top_text   = top.get("textDisplay", "")
        top_publ   = top.get("publishedAt", "")
        top_likes  = top.get("likeCount", 0)

        # 1) 원댓글 저장
        out.append({
            "comment_id":        top_id,
            "comment_author_id": top_auth,
            "comment_text":      top_text,
            "comment_published": top_publ,
            "comment_likes":     top_likes,
            "is_reply":          False,
            "reply_to_id":       None,
            "reply_author_id":   None,
            "reply_text":        None,
            "reply_published":   None
        })

        # 2) 답글 중 ‘@원댓글작성자ID’ 멘션만 저장
        for reply in thread.get("replies", {}).get("comments", []):
            r = reply["snippet"]
            if f"@{top_auth}" in r.get("textDisplay", ""):
                out.append({
                    "comment_id":        top_id,
                    "comment_author_id": top_auth,
                    "comment_text":      top_text,
                    "comment_published": top_publ,
                    "comment_likes":     top_likes,
                    "is_reply":          True,
                    "reply_to_id":       reply["id"],
                    "reply_author_id":   r.get("authorChannelId", {}).get("value", ""),
                    "reply_text":        r.get("textDisplay", ""),
                    "reply_published":   r.get("publishedAt", "")
                })
    return resp.get("nextPageToken"), out

def fetch_channel_uploads(svc, channelId, pageToken=None):
    # 채널 업로드 플레이리스트 ID 조회
    ch_resp = svc.channels().list(
        part="contentDetails",
        id=channelId,
        fields="items/contentDetails/relatedPlaylists/uploads"
    ).execute()
    items = ch_resp.get("items")
    if not items:
        # 채널 정보가 없으면 빈 결과 반환
        return {"items": [], "nextPageToken": None}

    upload_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    # 업로드 플레이리스트에서 영상 ID 조회
    pl_resp = svc.playlistItems().list(
        part="contentDetails",
        playlistId=upload_pl,
        maxResults=50,
        pageToken=pageToken,
        fields="nextPageToken,items/contentDetails/videoId"
    ).execute()
    return pl_resp

# ——————————————————————————
# Phase 1: 영상 ID 큐 생성
# ——————————————————————————
start = datetime.fromisoformat(START_DATE)
end   = datetime.fromisoformat(END_DATE)
for d in range((end - start).days):
    day1 = (start + timedelta(days=d)).isoformat()
    day2 = (start + timedelta(days=d+1)).isoformat()

    for cid in CHANNEL_IDS:
        # 키워드별
        for kw in KEYWORDS:
            token = None
            while True:
                resp = rotate_call(search_videos, cid, kw, day1, day2, token)
                for it in resp.get("items", []):
                    video_queues[cid].append(it["id"]["videoId"])
                token = resp.get("nextPageToken")
                if not token:
                    break
        # 카테고리별
        token = None
        while True:
            resp = rotate_call(search_videos, cid, None, day1, day2, token, category=POLITICS_CATEGORY)
            for it in resp.get("items", []):
                video_queues[cid].append(it["id"]["videoId"])
            token = resp.get("nextPageToken")
            if not token:
                break
    
        if not video_queues[cid]:
            token2 = None
            while True:
                resp2 = rotate_call(fetch_channel_uploads, cid, token2)
                for it2 in resp2.get("items", []):
                    video_queues[cid].append(it2["contentDetails"]["videoId"])
                token2 = resp2.get("nextPageToken")
                if not token2:
                    break

# ——————————————————————————
# Phase 2: 라운드로빈 댓글 수집
# ——————————————————————————
done_all = False
while sum(counts.values()) < TOTAL_LIMIT and not done_all:
    done_all = True
    for cid in CHANNEL_IDS:
        if counts[cid] >= PER_CHANNEL_LIMIT or not video_queues[cid]:
            continue
        done_all = False
        vid = video_queues[cid].popleft()

        # 영상 정보
        info = rotate_call(fetch_video_info, vid)["items"][0]
        # 태그 정보 수집 X ; 댓글 수집 적게 되는 채널
        '''
        tags = info["snippet"].get("tags", [])
        if not any(kw in tag for tag in KEYWORDS for tag in tags):
            continue
        '''
        base = {
            "video_id": vid,
            "channel": info["snippet"]["channelTitle"],
            "video_title": info["snippet"]["title"],
            "video_published": info["snippet"]["publishedAt"],
            "video_views": info["statistics"].get("viewCount", 0),
            "video_likes": info["statistics"].get("likeCount", 0),
            "video_comment_count": info["statistics"].get("commentCount", 0),
            "video_category_id": info["snippet"]["categoryId"]
        }

        # 댓글+멘션 답글 수집
        token = None
        while counts[cid] < PER_CHANNEL_LIMIT:
            try:
                token, items = rotate_call(fetch_comments_with_replies, vid, token)
            except HttpError as e:
                # commentsDisabled인 경우 건너뛰기
                if e.resp.status == 403 and e.error_details and any(d.get('reason')=='commentsDisabled' for d in e.error_details):
                    print(f"⚠️ {vid}: commentsDisabled → 스킵")
                    break
                else:
                    raise
            for c in items:
                results.append({**base, **c})
                counts[cid] += 1
                if counts[cid] >= PER_CHANNEL_LIMIT or sum(counts.values()) >= TOTAL_LIMIT:
                    break
            if not token or counts[cid] >= PER_CHANNEL_LIMIT:
                break

        time.sleep(0.1)

    # end for CHANNEL_IDS
# end while

# ——————————————————————————
# 결과 저장
# ——————————————————————————
pd.DataFrame(results).to_csv("all_news_김문수, 이재명, 이준석, 대선_20250519~20250521.csv", index=False, encoding="utf-8-sig")
print("채널별 counts:", counts)
print("총 댓글:", len(results))
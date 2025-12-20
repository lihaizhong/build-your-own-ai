from runtime import Args
from typings.securities_past.securities_past import Input, Output
import requests
from bs4 import BeautifulSoup
import time

def get_news_list(page=1, logger=None):
    url = f"https://feed.mix.sina.com.cn/api/roll/get?pageid=186&lid=1746&num=10&page={page}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()
    if logger:
        logger.info(f"获取第{page}页新闻，共{len(data.get('result', {}).get('data', []))}条")
    return data.get("result", {}).get("data", [])

def get_news_detail(news_url, logger=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(news_url, headers=headers, timeout=10)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "lxml")
    # 发布时间
    pub_time = ""
    for sel in [
        ("span", {"class": "date"}),
        ("span", {"id": "pub_date"}),
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "weibo: article:create_at"}),
    ]:
        tag = soup.find(*sel)
        if tag:
            pub_time = tag.get("content") if tag.name == "meta" else tag.text.strip()
            if pub_time: break
    # 正文内容
    content = ""
    for sel in [
        ("div", {"id": "artibody"}),
        ("div", {"class": "article"}),
        ("div", {"class": "main-content"}),
    ]:
        div = soup.find(*sel)
        if div:
            content = "\n".join([p.text.strip() for p in div.find_all("p") if p.text.strip()])
            if content: break
    return pub_time, content

def handler(args: Args[Input]) -> Output:
    logger = getattr(args, "logger", None)
    all_news = []
    for page in range(1, 21):  # 抓取前15页
        if logger:
            logger.info(f"抓取第{page}页...")
        news_list = get_news_list(page, logger=logger)
        for news in news_list[:3]:  # 每页只抓取前2条新闻
            title = news.get("title")
            url = news.get("url")
            if logger:
                logger.info(f"  访问: {title} {url}")
            try:
                pub_time, content = get_news_detail(url, logger=logger)
                all_news.append({
                    "title": title,
                    "publish_time": pub_time,
                    "content": content,
                    "url": url
                })
            except Exception as e:
                if logger:
                    logger.error(f"  解析失败: {e}")
    return {"data": all_news}  # 外层对象，key为data

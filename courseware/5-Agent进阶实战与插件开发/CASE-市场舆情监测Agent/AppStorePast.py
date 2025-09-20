from runtime import Args
from typings.AppStorePast.AppStorePast import Input, Output
import requests
from bs4 import BeautifulSoup

def handler(args: Args[Input]) -> Output:
    app_id = '1042567321'
    comments = []
    page = 1

    url = f'https://itunes.apple.com/cn/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/xml'
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.content, 'xml')
    entries = soup.find_all('entry')[1:]  # 跳过第一个entry

    for entry in entries:
        author = entry.find('name').text
        rating = entry.find('im:rating').text
        title = entry.find('title').text
        content = entry.find('content').text
        updated = entry.find('updated').text
        comments.append({
            'author': author,
            'rating': rating,
            'title': title,
            'content': content,
            'updated': updated
        })

    print(f"共获取到{len(comments)}条评论：\n")

    return {"comments": comments}

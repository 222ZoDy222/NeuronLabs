import os
import re
import requests
from urllib.parse import quote, unquote
from time import sleep

QUERY = "пицца"
MAX_IMAGES = 200
DOWNLOAD_DIR = "pizza_images"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Referer": "https://www.bing.com/"
}


def fetch_mediaurls(query, first):
    """Запрашиваем одну страницу Bing Images и вытаскиваем mediaurl"""
    url = (
        f"https://www.bing.com/images/async"
        f"?q={quote(query)}&first={first}&count=150&adlt=off"
    )

    print(f"\n[FETCH] first={first}")
    html = requests.get(url, headers=HEADERS, timeout=15).text

    encoded = re.findall(r"mediaurl=([^&\"']+)", html)
    decoded = [unquote(u) for u in encoded]

    print(f"[PARSE] найдено mediaurl: {len(decoded)}")
    return decoded


def download_image(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if not r.ok or "text/html" in r.headers.get("Content-Type", ""):
            print(f"[SKIP] не картинка: {url}")
            return False

        with open(path, "wb") as f:
            f.write(r.content)

        print(f"[OK] {path}")
        return True
    except Exception as e:
        print(f"[ERR] {url} -> {e}")
        return False


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    seen_urls = set()
    saved = 0
    first = 0

    while saved < MAX_IMAGES:
        urls = fetch_mediaurls(QUERY, first)

        if not urls:
            print("[STOP] больше URL нет")
            break

        for url in urls:
            if saved >= MAX_IMAGES:
                break

            if url in seen_urls:
                continue

            seen_urls.add(url)

            filename = f"pizza_{saved:03d}.jpg"
            path = os.path.join(DOWNLOAD_DIR, filename)

            if download_image(url, path):
                saved += 1
                sleep(0.2)  # мягко, без бана

        first += 150

    print(f"\n[DONE] скачано изображений: {saved}")


if __name__ == "__main__":
    main()

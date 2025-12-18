import os
import requests
from time import sleep

QUERY = "pizza"
MAX_IMAGES = 200
DOWNLOAD_DIR = "pizza_images"

API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "ImageDatasetBuilder/1.0 (educational use)",
}


def get_next_index(folder):
    """Находим следующий свободный индекс файла"""
    nums = []
    for f in os.listdir(folder):
        if f.startswith("pizza_") and f.endswith(".jpg"):
            try:
                nums.append(int(f.split("_")[1].split(".")[0]))
            except ValueError:
                pass
    return max(nums) + 1 if nums else 0


def fetch_page(query, offset=None, limit=50):
    """Получаем одну страницу JSON из Wikimedia API"""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,          # File:
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    if offset is not None:
        params["gsroffset"] = offset

    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def download_image(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if not r.ok or "image" not in r.headers.get("Content-Type", ""):
            print(f"[SKIP] не изображение: {url}")
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

    current_index = get_next_index(DOWNLOAD_DIR)
    saved = 0
    offset = None

    print(f"[START] начинаем с индекса: {current_index}")

    while saved < MAX_IMAGES:
        data = fetch_page(QUERY, offset)

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            print("[STOP] больше данных нет")
            break

        for page in pages.values():
            if saved >= MAX_IMAGES:
                break

            info = page.get("imageinfo")
            if not info:
                continue

            url = info[0].get("url")
            if not url:
                continue

            filename = f"pizza_{current_index:03d}.jpg"
            path = os.path.join(DOWNLOAD_DIR, filename)

            if download_image(url, path):
                current_index += 1
                saved += 1
                sleep(0.2)  # без агрессии

        # пагинация
        offset = data.get("continue", {}).get("gsroffset")
        if offset is None:
            print("[STOP] достигнут конец выдачи")
            break

    print(f"\n[DONE] скачано новых изображений: {saved}")


if __name__ == "__main__":
    main()

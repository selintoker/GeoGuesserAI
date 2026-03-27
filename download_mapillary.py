import os
import csv
import time
import math
import random
import requests
from typing import List, Tuple, Dict, Set

ACCESS_TOKEN = ""

# Approximate Tokyo, Japan bounding box
WEST = 139.55
SOUTH = 35.50
EAST = 139.95
NORTH = 35.85

TARGET_IMAGES = 9999
OUTPUT_DIR = "images"
CSV_PATH = "metadata.csv"
SEEN_IDS_PATH = "seen_ids.txt"

LON_STEP = 0.02
LAT_STEP = 0.02

PER_REQUEST_LIMIT = 100
SLEEP_BETWEEN_IMAGE_DOWNLOADS = 0.05
SLEEP_BETWEEN_API_CALLS = 0.20
REQUEST_TIMEOUT = 60

FIELDS = "id,captured_at,geometry,thumb_1024_url"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_seen_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_seen_id(path: str, image_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(image_id + "\n")


def ensure_csv_header(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "captured_at",
            "longitude",
            "latitude",
            "filename",
            "tile_west",
            "tile_south",
            "tile_east",
            "tile_north",
        ])


def append_csv_row(path: str, row: List[str]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def generate_tiles(west: float, south: float, east: float, north: float,
                   lon_step: float, lat_step: float) -> List[Tuple[float, float, float, float]]:
    tiles = []
    lat = south
    while lat < north:
        next_lat = min(lat + lat_step, north)
        lon = west
        while lon < east:
            next_lon = min(lon + lon_step, east)
            tiles.append((lon, lat, next_lon, next_lat))
            lon = next_lon
        lat = next_lat
    return tiles


def fetch_images_in_bbox(bbox: Tuple[float, float, float, float], limit: int = 100) -> List[Dict]:
    west, south, east, north = bbox
    url = "https://graph.mapillary.com/images"
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "fields": FIELDS,
        "limit": limit,
    }
    headers = {
        "Authorization": f"OAuth {ACCESS_TOKEN}"
    }

    response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    if "data" not in data:
        return []

    return data["data"]


def download_image(image_url: str, filepath: str) -> None:
    response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(response.content)


def main():
    ensure_csv_header(CSV_PATH)
    seen_ids = load_seen_ids(SEEN_IDS_PATH)

    existing_files = [name for name in os.listdir(OUTPUT_DIR) if name.lower().endswith(".jpg")]
    downloaded_count = len(existing_files)

    print(f"Already have {downloaded_count} images on disk")
    print(f"Already have {len(seen_ids)} seen IDs recorded")

    tiles = generate_tiles(WEST, SOUTH, EAST, NORTH, LON_STEP, LAT_STEP)
    random.shuffle(tiles)
    print(f"Generated {len(tiles)} tiles")

    tile_index = 0

    for tile in tiles:
        if downloaded_count >= TARGET_IMAGES:
            break

        tile_index += 1
        west, south, east, north = tile
        print(f"\nTile {tile_index}/{len(tiles)}: {tile}")

        try:
            images = fetch_images_in_bbox(tile, limit=PER_REQUEST_LIMIT)
        except Exception as e:
            print(f"  Failed tile request: {e}")
            time.sleep(1.0)
            continue

        random.shuffle(images)
        print(f"  Returned {len(images)} images")

        for image in images:
            if downloaded_count >= TARGET_IMAGES:
                break

            image_id = str(image.get("id", "")).strip()
            if not image_id:
                continue

            if image_id in seen_ids:
                continue

            image_url = image.get("thumb_1024_url")
            geometry = image.get("geometry", {})
            coords = geometry.get("coordinates", None)

            if not image_url or not coords or len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]
            captured_at = image.get("captured_at", "")

            filename = f"{image_id}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            try:
                download_image(image_url, filepath)
            except Exception as e:
                print(f"  Failed image download for {image_id}: {e}")
                continue

            append_csv_row(CSV_PATH, [
                image_id,
                captured_at,
                lon,
                lat,
                filename,
                west,
                south,
                east,
                north,
            ])

            append_seen_id(SEEN_IDS_PATH, image_id)
            seen_ids.add(image_id)
            downloaded_count += 1

            print(f"  Saved {downloaded_count}/{TARGET_IMAGES}: {filename}")
            time.sleep(SLEEP_BETWEEN_IMAGE_DOWNLOADS)

        time.sleep(SLEEP_BETWEEN_API_CALLS)

    print("\nDone.")
    print(f"Final downloaded image count: {downloaded_count}")


if __name__ == "__main__":
    main()

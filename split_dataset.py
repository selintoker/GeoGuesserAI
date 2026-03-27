import os
import random
import shutil

CITIES = {
    "nyc":      "images_nyc",
    "istanbul": "images_istanbul",
    "tokyo":    "images_tokyo",
}

DATASET_DIR = "dataset"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def get_images(directory: str):
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def split_images(images, train_ratio, val_ratio):
    random.shuffle(images)
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return images[:train_end], images[train_end:val_end], images[val_end:]


def main():
    random.seed(SEED)

    for city, src_dir in CITIES.items():
        if not os.path.isdir(src_dir):
            print(f"  Skipping {city}: directory '{src_dir}' not found")
            continue

        images = get_images(src_dir)
        if not images:
            print(f"  Skipping {city}: no images found in '{src_dir}'")
            continue

        train, val, test = split_images(images, SPLITS["train"], SPLITS["val"])

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            dest_dir = os.path.join(DATASET_DIR, split_name, city)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in split_files:
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

        print(f"{city}: {len(images)} images -> train={len(train)}, val={len(val)}, test={len(test)}")

    print("\nDataset split complete.")
    print(f"Output: {DATASET_DIR}/{{train,val,test}}/{{city}}/")


if __name__ == "__main__":
    main()

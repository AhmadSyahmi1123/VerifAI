import os
import shutil
import random

# ------------- CONFIG -------------
SOURCE_DIR = "datasets/processed"   # your dataset root
TARGET_DIR = "dataset_split"

SPLIT = {
    "train": 0.7,
    "val":   0.15,
    "test":  0.15
}

CLASSES = ["real", "fake"]
SEED = 42
# -----------------------------------

random.seed(SEED)

def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def split_class(class_name):
    input_dir = os.path.join(SOURCE_DIR, class_name)
    video_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    random.shuffle(video_folders)

    n = len(video_folders)
    n_train = int(SPLIT["train"] * n)
    n_val   = int(SPLIT["val"] * n)

    train_videos = video_folders[:n_train]
    val_videos   = video_folders[n_train : n_train + n_val]
    test_videos  = video_folders[n_train + n_val:]

    print(f"\n[{class_name.upper()}]")
    print("Train:", len(train_videos))
    print("Val:  ", len(val_videos))
    print("Test: ", len(test_videos))

    return train_videos, val_videos, test_videos


def copy_split(videos, class_name, split_name):
    for vid in videos:
        src_dir = os.path.join(SOURCE_DIR, class_name, vid)
        dst_dir = os.path.join(TARGET_DIR, split_name, class_name, vid)

        os.makedirs(dst_dir, exist_ok=True)

        for file in os.listdir(src_dir):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            safe_copy(src_file, dst_file)


def main():
    print("Starting dataset split...")

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

    for cls in CLASSES:
        train_v, val_v, test_v = split_class(cls)

        copy_split(train_v, cls, "train")
        copy_split(val_v, cls, "val")
        copy_split(test_v, cls, "test")

    print("\n✅ Dataset split completed!")
    print(f"Saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()

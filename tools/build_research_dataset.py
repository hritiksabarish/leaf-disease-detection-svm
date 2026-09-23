import os
import shutil
import hashlib
import csv
import json
import random
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\sabar\Desktop\leaf-disease-detection")
SOURCE_DIR = Path(r"C:\Users\sabar\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1")
TARGET_DIR = PROJECT_ROOT / "data" / "research_lab"
MANIFEST_PATH = PROJECT_ROOT / "data" / "research_manifest.csv"

MAPPING = {
    "Pepper__bell___Bacterial_spot": {"local": "Pepper_bell___Bacterial_spot", "crop": "Pepper", "condition": "Bacterial spot"},
    "Pepper__bell___healthy": {"local": "Pepper_bell___healthy", "crop": "Pepper", "condition": "Healthy"},
    "Potato___Early_blight": {"local": "Potato___Early_blight", "crop": "Potato", "condition": "Early blight"},
    "Potato___healthy": {"local": "Potato___healthy", "crop": "Potato", "condition": "Healthy"},
    "Potato___Late_blight": {"local": "Potato___Late_blight", "crop": "Potato", "condition": "Late blight"},
    "Tomato_Bacterial_spot": {"local": "Tomato___Bacterial_spot", "crop": "Tomato", "condition": "Bacterial spot"},
    "Tomato_Early_blight": {"local": "Tomato___Early_blight", "crop": "Tomato", "condition": "Early blight"},
    "Tomato_healthy": {"local": "Tomato___healthy", "crop": "Tomato", "condition": "Healthy"},
    "Tomato_Late_blight": {"local": "Tomato___Late_blight", "crop": "Tomato", "condition": "Late blight"},
    "Tomato_Leaf_Mold": {"local": "Tomato___Leaf_Mold", "crop": "Tomato", "condition": "Leaf Mold"}
}

MAX_IMAGES_PER_CLASS = 200
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

def get_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return None

def main():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    global_hashes = set()
    manifest_rows = []
    
    stats = {
        "classes": {},
        "total_images": 0,
        "invalid_images": 0,
        "duplicates_skipped": 0,
        "capped_skipped": 0
    }
    
    # Pre-gather files from source
    source_files_by_class = {k: [] for k in MAPPING.keys()}
    
    for f in SOURCE_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            parent_name = f.parent.name
            if parent_name in source_files_by_class:
                source_files_by_class[parent_name].append(f)
                
    for src_cls, files in source_files_by_class.items():
        local_cls = MAPPING[src_cls]["local"]
        crop = MAPPING[src_cls]["crop"]
        condition = MAPPING[src_cls]["condition"]
        
        cls_dir = TARGET_DIR / local_cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        stats["classes"][local_cls] = 0
        
        # Sort files for deterministic behavior before shuffling
        files = sorted(list(files), key=lambda x: str(x))
        random.shuffle(files)
        
        for f in files:
            if stats["classes"][local_cls] >= MAX_IMAGES_PER_CLASS:
                stats["capped_skipped"] += 1
                continue
                
            # Check validity
            if f.stat().st_size == 0:
                stats["invalid_images"] += 1
                continue
                
            try:
                with Image.open(f) as img:
                    img.verify()
            except Exception:
                stats["invalid_images"] += 1
                continue
                
            # Check Hash
            h = get_hash(f)
            if h in global_hashes:
                stats["duplicates_skipped"] += 1
                continue
                
            # Valid & Unique -> Add
            global_hashes.add(h)
            
            # Destination path
            dst_file = cls_dir / f.name
            # Prevent name collision if any (unlikely since we deduplicate, but just in case)
            if dst_file.exists():
                dst_file = cls_dir / f"{h}_{f.name}"
                
            shutil.copy2(f, dst_file)
            
            stats["classes"][local_cls] += 1
            stats["total_images"] += 1
            
            # Add to manifest
            manifest_rows.append({
                "image_path": str(dst_file.relative_to(PROJECT_ROOT)),
                "source_path": str(f),
                "source_class": src_cls,
                "local_class": local_cls,
                "crop": crop,
                "condition": condition,
                "domain": "lab",
                "source_dataset": "emmarex/plantdisease",
                "md5": h
            })

    # Write Manifest
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["image_path", "source_path", "source_class", "local_class", "crop", "condition", "domain", "source_dataset", "md5"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

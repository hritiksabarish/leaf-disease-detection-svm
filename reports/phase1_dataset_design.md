# Phase 1 Dataset Design

## Current baseline dataset
The current baseline uses 4 classes (`Potato___Early_blight`, `Potato___healthy`, `Tomato___Late_blight`, `Tomato___healthy`) with a total of 772 images. 

## Source dataset inventory
The source dataset (`emmarex/plantdisease` on Kaggle) contains exactly 3 crops across 15 classes. Notably, the Kaggle extraction contains a nested `PlantVillage/PlantVillage/` directory, meaning every file is duplicated in the source cache. The *unique* image counts are:

| Crop | Disease/Condition | Source Folder | Unique Images |
|---|---|---|---|
| Pepper | Bacterial spot | `Pepper__bell___Bacterial_spot` | 997 |
| Pepper | Healthy | `Pepper__bell___healthy` | 1478 |
| Potato | Early blight | `Potato___Early_blight` | 1000 |
| Potato | Healthy | `Potato___healthy` | **152** (Bottleneck) |
| Potato | Late blight | `Potato___Late_blight` | 1000 |
| Tomato | Bacterial spot | `Tomato_Bacterial_spot` | 2127 |
| Tomato | Early blight | `Tomato_Early_blight` | 1000 |
| Tomato | Healthy | `Tomato_healthy` | 1591 |
| Tomato | Late blight | `Tomato_Late_blight` | 1909 |
| Tomato | Leaf Mold | `Tomato_Leaf_Mold` | 952 |
| Tomato | Septoria leaf spot | `Tomato_Septoria_leaf_spot` | 1771 |
| Tomato | Spider mites | `Tomato_Spider_mites_Two_spotted_spider_mite` | 1676 |
| Tomato | Target Spot | `Tomato__Target_Spot` | 1404 |
| Tomato | Mosaic virus | `Tomato__Tomato_mosaic_virus` | 373 |
| Tomato | YellowLeaf Curl Virus | `Tomato__Tomato_YellowLeaf__Curl_Virus` | 3208 |

## Proposed multi-crop classes
We target **3 crops** and **10 classes**:
1. `Pepper_bell___Bacterial_spot`
2. `Pepper_bell___healthy`
3. `Potato___Early_blight`
4. `Potato___healthy`
5. `Potato___Late_blight`
6. `Tomato___Bacterial_spot`
7. `Tomato___Early_blight`
8. `Tomato___healthy`
9. `Tomato___Late_blight`
10. `Tomato___Leaf_Mold`

## Class selection rationale
- **Pepper & Potato**: All available classes (2 and 3 respectively) are selected to maximize multi-crop diversity.
- **Tomato**: 5 classes are selected to keep the dataset size manageable (10 total classes fits the 8–12 rule) while providing a robust set of diseases that have counterparts in other crops (e.g. Bacterial spot, Early/Late blight).
- **All chosen classes**: Contain real, lab-controlled images with distinct healthy/diseased variations, suitable for the handcrafted texture/color segmentation pipeline.

## Proposed per-class cap
**200 images per class.**
*Rationale*: 200 provides a sufficient number of images to train the SVM without slowing down the handcrafted feature extraction. While most classes easily support 200 images, `Potato___healthy` is limited to exactly 152 unique authentic images. 152 vs 200 is a highly acceptable, mild imbalance that avoids artificially truncating the other 9 classes to 150.

## Final directory structure
```text
data/
    baseline/          # (Preserves the existing 4-class baseline)
    research_lab/      # (The new expanded 10-class dataset)
        Pepper_bell___Bacterial_spot/
        Pepper_bell___healthy/
        Potato___Early_blight/
        ...
```

## Class mapping
| Source Kaggle Folder | Local Target Class | Crop | Condition |
|---|---|---|---|
| `Pepper__bell___Bacterial_spot` | `Pepper_bell___Bacterial_spot` | Pepper | Bacterial spot |
| `Pepper__bell___healthy` | `Pepper_bell___healthy` | Pepper | Healthy |
| `Potato___Early_blight` | `Potato___Early_blight` | Potato | Early blight |
| `Potato___healthy` | `Potato___healthy` | Potato | Healthy |
| `Potato___Late_blight` | `Potato___Late_blight` | Potato | Late blight |
| `Tomato_Bacterial_spot` | `Tomato___Bacterial_spot` | Tomato | Bacterial spot |
| `Tomato_Early_blight` | `Tomato___Early_blight` | Tomato | Early blight |
| `Tomato_healthy` | `Tomato___healthy` | Tomato | Healthy |
| `Tomato_Late_blight` | `Tomato___Late_blight` | Tomato | Late blight |
| `Tomato_Leaf_Mold` | `Tomato___Leaf_Mold` | Tomato | Leaf Mold |

## Duplicate handling
During extraction into `data/research_lab/`, a script will maintain a global set of seen MD5 file hashes. If an image hash is already in the set, the file will be skipped. This solves the Kaggle nested folder duplication issue and ensures absolutely zero duplicate images in the dataset.

## Baseline preservation strategy
The existing `data/subset/` will be renamed to `data/baseline/`. The new expanded dataset will be constructed inside a new `data/research_lab/` directory. The scripts will be updated to point to the new directory via argument flags, ensuring the baseline pipeline remains 100% reproducible and isolated.

## Research dataset manifest
A CSV manifest (`data/research_manifest.csv`) will be generated during extraction with columns: `image_path`, `source_class`, `local_class`, `crop`, `condition`, `domain` ("lab"), and `source_dataset` ("emmarex/plantdisease").

## Risks / limitations
- `Potato___healthy` will naturally have ~152 images instead of 200, leading to a minor class imbalance.
- The pipeline's handcrafted features (18 dimensions) may become a bottleneck in distinguishing 10 complex classes, which perfectly motivates the need for the upcoming research steps (feature selection, kernel comparison).

## Recommended next step
Awaiting your approval to:
1. Rename `data/subset/` to `data/baseline/`.
2. Execute the extraction script to build `data/research_lab/` safely using MD5 hashing.
3. Generate the `data/research_manifest.csv`.

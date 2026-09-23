# Research Lab Dataset

## Source
The authentic source images are derived from the Kaggle dataset `emmarex/plantdisease`.

## Crops
1. Pepper
2. Potato
3. Tomato

## Classes
The dataset is composed of 10 fully authentic classes.

## Class distribution
- **Pepper_bell___Bacterial_spot**: 200
- **Pepper_bell___healthy**: 200
- **Potato___Early_blight**: 200
- **Potato___healthy**: 152
- **Potato___Late_blight**: 200
- **Tomato___Bacterial_spot**: 200
- **Tomato___Early_blight**: 200
- **Tomato___healthy**: 200
- **Tomato___Late_blight**: 200
- **Tomato___Leaf_Mold**: 200

## Total images
**1,952 unique images**

## Deduplication
A global MD5 hash check was performed across all candidate images during extraction. 
- **Duplicates skipped:** 235
Every selected image is cryptographically guaranteed to be unique. No exact file-level duplication exists in this dataset.

## Image validation
- **Invalid images detected:** 0
Every image was verified to be a readable, non-zero-byte, non-corrupted `.jpg` / `.png` / `.jpeg` file using PIL.

## Per-class selection
Images were drawn randomly using `random_state=42` from the verified unique source images to reach a cap of 200 images per class. 
- **Skipped due to capping limit:** 22,225 images.

## Source-to-local mapping
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

## Manifest
A CSV manifest has been generated at `data/research_manifest.csv` containing the exact file paths, original source paths, local classes, crops, conditions, domains ("lab"), source datasets, and MD5 hashes for all 1,952 images.

## Baseline preservation
The verified baseline dataset remains fully intact and untouched inside `data/subset/`. This new multi-crop dataset has been isolated in a dedicated directory: `data/research_lab/`. No modifications were made to the core pipeline (`src/classify.py`, etc.).

## Limitations
**Important:** `Potato___healthy` has a natural limit of 152 unique authentic images in the source dataset. While the other 9 selected classes easily met the 200 target, `Potato___healthy` maxed out at 152. This is a legitimate constraint of the dataset and introduces a minor, acceptable class imbalance. No synthetic augmentation or duplication was used to hide this limitation.

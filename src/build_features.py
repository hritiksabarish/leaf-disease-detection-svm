import os
import csv
import time
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress skimage and opencv warnings during bulk processing
warnings.filterwarnings("ignore")

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "research_manifest.csv"
FEATURES_OUTPUT = PROJECT_ROOT / "data" / "research_features.csv"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "feature_extraction_report.md"
STATS_OUTPUT = PROJECT_ROOT / "reports" / "feature_statistics.csv"

def main():
    start_time = time.time()
    
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    # Read manifest
    df_manifest = pd.read_csv(MANIFEST_PATH)
    total_images = len(df_manifest)
    
    success_count = 0
    fail_count = 0
    failures = []
    
    all_rows = []
    feature_names = []

    print(f"Starting feature extraction for {total_images} images...")

    for idx, row in df_manifest.iterrows():
        img_path = PROJECT_ROOT / row['image_path']
        
        try:
            prep = load_and_preprocess_image(str(img_path))
            leaf_mask = isolate_leaf_otsu(prep['blurred_rgb'])
            seg = segment_diseased_region(prep['blurred_rgb'], prep['hsv'], leaf_mask)
            
            feats_dict, f_vec, f_names = extract_all_features(
                prep['blurred_rgb'], prep['hsv'], prep['gray'],
                seg['diseased_mask'], seg['leaf_mask']
            )
            
            if not feature_names:
                feature_names = f_names
                
            # Fallback for missing/NaN values
            for k, v in feats_dict.items():
                if np.isnan(v) or np.isinf(v):
                    feats_dict[k] = 0.0 # Default fallback
            
            # Combine manifest metadata + features
            out_row = {
                'image_path': row['image_path'],
                'label': row['local_class'],
                'crop': row['crop'],
                'condition': row['condition'],
                'domain': row['domain']
            }
            out_row.update(feats_dict)
            all_rows.append(out_row)
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            failures.append((str(img_path), str(e)))
            print(f"Failed {img_path.name}: {e}")

    # Save to CSV
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(FEATURES_OUTPUT, index=False)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate Feature Statistics
    feat_df = df_out[feature_names]
    stats_df = pd.DataFrame({
        'mean': feat_df.mean(),
        'std': feat_df.std(),
        'min': feat_df.min(),
        'max': feat_df.max(),
        'missing': feat_df.isna().sum(),
        'constant': feat_df.nunique() == 1
    })
    stats_df.to_csv(STATS_OUTPUT)
    
    # Write Extraction Report
    with open(REPORT_OUTPUT, 'w') as f:
        f.write("# Feature Extraction Report\n\n")
        f.write(f"- **Total images attempted**: {total_images}\n")
        f.write(f"- **Successfully processed**: {success_count}\n")
        f.write(f"- **Failed**: {fail_count}\n")
        f.write(f"- **Feature count extracted**: {len(feature_names)}\n")
        f.write(f"- **Processing time**: {duration:.2f} seconds\n")
        
        missing_total = feat_df.isna().sum().sum()
        f.write(f"- **Missing values (NaN) encountered**: {missing_total}\n")
        f.write("\n## Failures\n")
        if not failures:
            f.write("None.\n")
        else:
            for path, err in failures:
                f.write(f"- `{path}`: {err}\n")

    print(f"Extraction complete! Saved to {FEATURES_OUTPUT}")
    print(f"Stats saved to {STATS_OUTPUT}")

if __name__ == "__main__":
    main()

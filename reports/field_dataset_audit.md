# Field Dataset Audit

**Dataset source**: Kaggle (`pratik2901/plantdoc-dataset`)
**Dataset name**: PlantDoc
**Available classes**: Typically 27 classes (17 disease, 10 healthy) across 13 crops.
**Available crops**: Includes Tomato, Potato, Pepper, Corn, Apple, etc.
**Image conditions**: Genuine field conditions (natural backgrounds, varying lighting, scales).

## Audit Result
I attempted to download the PlantDoc dataset from Kaggle via `kagglehub` to construct the natural domain-shift evaluation for Phase 5. However, the download failed with a `403 Client Error: You don't have permission to access resource at URL`. This indicates that Kaggle API authentication is missing or restricted on this machine.

Because a genuine external field dataset could not be securely acquired and verified, **Phase 5C - 5G (Natural Field Domain Shift)** will be omitted. 

As per the research protocol, we will instead rely entirely on **Phase 5H (Controlled Perturbations)** using the established 391-image held-out lab test set to rigorously evaluate the robustness of the Phase 4 reference model against synthetic domain shifts (Brightness, Contrast, Noise, and Blur).

# Phase 7A — Treatment Recommendations Knowledge Base

## 1. Purpose
The purpose of Phase 7A is to establish an application-layer knowledge base that maps the raw machine-learning classifier outputs to structured, actionable, and safe agricultural management advice. This bridges the gap between the algorithmic diagnostic prediction and practical crop management for an end user.

## 2. Knowledge-base architecture
The knowledge base (`src/recommendations.py`) acts as a deterministic lookup table distinct from the ML pipeline. It exposes a simple API: `get_recommendation(predicted_class)`. It does not rely on generative models (LLMs) to invent advice.

## 3. Supported crops
1. Pepper (Bell)
2. Potato
3. Tomato

## 4. Supported diseases
1. Bacterial Spot (Pepper)
2. Early Blight (Potato)
3. Late Blight (Potato)
4. Bacterial Spot (Tomato)
5. Early Blight (Tomato)
6. Late Blight (Tomato)
7. Leaf Mold (Tomato)

## 5. Healthy classes
1. Pepper - Healthy
2. Potato - Healthy
3. Tomato - Healthy

*Note*: The knowledge base is explicitly designed to recognize "Healthy" states and advise *against* chemical disease intervention, promoting standard preventive agriculture instead.

## 6. Recommendation fields
Every class resolves to an 8-field schema:
- `crop`: The host plant species.
- `disease`: The specific disease name.
- `status`: "diseased" or "healthy" or "unknown".
- `summary`: A concise description of the pathogen and presentation.
- `management`: Actionable steps (e.g. sanitation, targeted pruning, chemical control).
- `prevention`: Cultural and systemic practices (e.g. crop rotation, resistant varieties, irrigation).
- `caution`: Specific chemical/agricultural safety warnings.
- `source`: An array of references used to construct the advice.

## 7. Sources used
To ensure safety and reliability, treatments were compiled from authoritative University Extension Services, including:
- **Penn State Extension** (Late Blight, Leaf Mold)
- **Cornell University College of Agriculture and Life Sciences** (Potato Early Blight)
- **UMass Amherst Center for Agriculture** (Pepper Bacterial Spot)
- **University of Minnesota Extension** (Tomato Bacterial Spot)
- **University of Wisconsin-Madison Extension** (Tomato Early Blight)

## 8. Label normalization
The ML models emit labels with varying delimiter structures (e.g., `Tomato___Late_blight`, `Pepper__bell___Bacterial_spot`). A regex-driven normalization layer (`normalize_class_name`) safely strips capitalization and collapses non-alphanumeric character sequences into single underscores to consistently route these labels to their canonical knowledge base keys (e.g., `tomato_late_blight`).

## 9. Safety/usage limitations
- **No universal pesticide claims**: The system strictly avoids recommending generic or dosage-specific chemical applications.
- **Jurisdiction warning**: Every chemical suggestion includes the disclaimer: *"Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance."*
- **Toxicity**: Particular warnings are raised for devastating diseases like Late Blight (to destroy plants immediately) and health considerations for spores like Leaf Mold (allergen warning).

## 10. Testing performed
A dedicated unit test suite (`tests/test_recommendations.py`) verified:
- Total coverage of all 10 ML classes.
- Logical separation of `diseased` vs `healthy` statuses.
- Fallback integrity for unknown/unsupported labels (e.g., `Apple___Apple_scab`).
- Formatting, normalization stringency, and structural validity of the nested `source` dictionaries.
*Result*: 100% pass rate.

## 11. Known limitations
The knowledge base relies entirely on the classifier's accuracy. If the SVM misdiagnoses Early Blight as Late Blight, the UI will recommend destroying the crop. The recommendation module does not perform secondary probabilistic or visual checks.

## 12. Explicit limitation of ML validation
**IMPORTANT**: The treatment recommendations generated in this phase are a completely separate knowledge-base layer. They were *not* part of the classifier's measured accuracy/F1 evaluation in prior phases, and they have not been independently validated by an agronomist against the specific images in the dataset. They are provided solely as reference information.

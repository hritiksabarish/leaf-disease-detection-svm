import re

# Treatment Recommendation Knowledge Base
# Source: Authoritative agricultural extension guidelines (Penn State, UMass, Cornell, USDA, etc.)

KNOWLEDGE_BASE = {
    "pepper_bell_bacterial_spot": {
        "crop": "Pepper",
        "disease": "Bacterial Spot",
        "status": "diseased",
        "summary": "A destructive bacterial disease caused by Xanthomonas species. It causes leaf spotting, defoliation, and fruit spotting.",
        "management": [
            "Remove and destroy infected plant material immediately.",
            "Avoid working in the field when foliage is wet to prevent spreading the bacteria.",
            "Apply copper-based bactericides early in the disease cycle, though effectiveness may be limited if weather is highly conducive to disease."
        ],
        "prevention": [
            "Use certified disease-free seeds and transplants.",
            "Practice a 2-3 year crop rotation away from peppers, tomatoes, and other solanaceous crops.",
            "Ensure excellent weed control, especially of nightshade family weeds.",
            "Use drip irrigation instead of overhead sprinklers to keep foliage dry."
        ],
        "caution": "Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "UMass Amherst Center for Agriculture, Food, and the Environment",
                "title": "Pepper, Bacterial Spot",
                "url": "https://ag.umass.edu/vegetable/fact-sheets/pepper-bacterial-spot"
            }
        ]
    },
    "pepper_bell_healthy": {
        "crop": "Pepper",
        "disease": "None",
        "status": "healthy",
        "summary": "The pepper plant shows no visible signs of leaf disease.",
        "management": [
            "No chemical disease treatment is recommended for healthy plants."
        ],
        "prevention": [
            "Maintain optimal spacing for good airflow.",
            "Water at the base of the plant using drip irrigation.",
            "Monitor weekly for early signs of pests or disease."
        ],
        "caution": "Continue standard preventive agricultural practices.",
        "source": []
    },
    "potato_early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "status": "diseased",
        "summary": "A fungal disease caused by Alternaria solani. It appears as dark lesions with concentric rings (target board appearance) on older leaves.",
        "management": [
            "Apply labeled protectant fungicides (e.g., chlorothalonil or mancozeb) as part of a regular spray program before the disease becomes severe.",
            "Maintain optimal plant nutrition, as early blight is often a disease of aging or stressed plants."
        ],
        "prevention": [
            "Plant resistant or tolerant potato varieties.",
            "Destroy potato vines after harvest and deeply plow remaining debris.",
            "Rotate crops with non-solanaceous plants for 2-3 years.",
            "Allow tubers to mature fully before harvest to minimize tuber infection."
        ],
        "caution": "Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "Cornell University College of Agriculture and Life Sciences",
                "title": "Early blight of potato",
                "url": "https://cals.cornell.edu/nys-ipm/agriculture/vegetables/potato/early-blight-potato"
            }
        ]
    },
    "potato_healthy": {
        "crop": "Potato",
        "disease": "None",
        "status": "healthy",
        "summary": "The potato plant shows no visible signs of leaf disease.",
        "management": [
            "No chemical disease treatment is recommended for healthy plants."
        ],
        "prevention": [
            "Ensure proper hilling to protect developing tubers.",
            "Maintain consistent soil moisture without waterlogging.",
            "Monitor regularly during periods of high humidity and moderate temperatures."
        ],
        "caution": "Continue standard preventive agricultural practices.",
        "source": []
    },
    "potato_late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "status": "diseased",
        "summary": "A devastating water mold disease caused by Phytophthora infestans. It causes rapidly spreading dark water-soaked lesions and can destroy a crop in days.",
        "management": [
            "If late blight is confirmed, immediately destroy infected plants to prevent spore dispersal.",
            "Apply specific systemic and protectant fungicides approved for late blight immediately, following local extension alerts.",
            "Kill vines 2-3 weeks before harvest to prevent spores from washing down to tubers."
        ],
        "prevention": [
            "Plant only certified, disease-free seed potatoes.",
            "Eliminate volunteer potatoes and cull piles where the pathogen overwinters.",
            "Monitor local late blight forecasting systems (e.g., USAblight).",
            "Avoid overhead irrigation during cool, wet weather."
        ],
        "caution": "Late blight is highly contagious and devastating. Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "Penn State Extension",
                "title": "Late Blight on Potatoes and Tomatoes",
                "url": "https://extension.psu.edu/late-blight-on-potatoes-and-tomatoes"
            }
        ]
    },
    "tomato_bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "status": "diseased",
        "summary": "A bacterial disease caused by Xanthomonas species. Causes small, dark, greasy spots on leaves and scabby lesions on fruit.",
        "management": [
            "Remove infected plants or heavily infected lower leaves if the infestation is localized.",
            "Avoid working with plants when they are wet.",
            "Apply copper-based bactericides, often mixed with mancozeb, to slow disease spread, though efficacy is limited."
        ],
        "prevention": [
            "Use hot-water treated seeds and certified disease-free transplants.",
            "Practice a 3-year crop rotation.",
            "Avoid overhead watering; use drip irrigation.",
            "Sanitize all tools and greenhouse structures."
        ],
        "caution": "Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "University of Minnesota Extension",
                "title": "Bacterial spot of tomato and pepper",
                "url": "https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper"
            }
        ]
    },
    "tomato_early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "status": "diseased",
        "summary": "A fungal disease caused by Alternaria linariae (formerly A. tomaticola). Manifests as target-spot lesions on lower leaves, causing them to yellow and drop.",
        "management": [
            "Prune and remove infected lower leaves to improve airflow and reduce inoculum.",
            "Apply protectant fungicides (e.g., chlorothalonil, copper) at the first sign of disease on lower leaves.",
            "Mulch around the base of the plant to prevent soil-borne spores from splashing onto foliage."
        ],
        "prevention": [
            "Stake or cage tomatoes to keep foliage off the ground.",
            "Ensure wide plant spacing for sunlight penetration and rapid drying.",
            "Rotate crops and remove all tomato debris at the end of the season."
        ],
        "caution": "Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "University of Wisconsin-Madison Extension",
                "title": "Early Blight",
                "url": "https://hort.extension.wisc.edu/articles/early-blight/"
            }
        ]
    },
    "tomato_healthy": {
        "crop": "Tomato",
        "disease": "None",
        "status": "healthy",
        "summary": "The tomato plant shows no visible signs of leaf disease.",
        "management": [
            "No chemical disease treatment is recommended for healthy plants."
        ],
        "prevention": [
            "Stake or cage plants to maintain good airflow.",
            "Prune lower suckers to keep foliage off the soil.",
            "Water at the base of the plant in the morning."
        ],
        "caution": "Continue standard preventive agricultural practices.",
        "source": []
    },
    "tomato_late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "status": "diseased",
        "summary": "A devastating water mold disease caused by Phytophthora infestans. Appears as large, dark, water-soaked patches on leaves and stems, often with white fungal growth underneath.",
        "management": [
            "Immediately bag and dispose of infected plants in the trash; do not compost them.",
            "Apply specific targeted fungicides immediately if late blight is forecasted or detected in the area.",
            "Harvest any mature, uninfected fruit immediately."
        ],
        "prevention": [
            "Plant late-blight resistant tomato varieties (e.g., 'Defiant', 'Mountain Magic').",
            "Monitor local blight tracking systems.",
            "Do not allow volunteer tomatoes or potatoes to grow near the crop."
        ],
        "caution": "Late blight is highly contagious and devastating. Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "Penn State Extension",
                "title": "Late Blight on Potatoes and Tomatoes",
                "url": "https://extension.psu.edu/late-blight-on-potatoes-and-tomatoes"
            }
        ]
    },
    "tomato_leaf_mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "status": "diseased",
        "summary": "Caused by Passalora fulva (formerly Fulvia fulva). Thrives in high humidity (>85%), common in greenhouses. Presents as pale green/yellow spots on the upper leaf surface and olive-green mold on the underside.",
        "management": [
            "Dramatically improve air circulation using fans or venting (if in a greenhouse or high tunnel).",
            "Prune lower leaves to improve airflow through the canopy.",
            "Apply labeled protectant fungicides if environmental controls are insufficient."
        ],
        "prevention": [
            "Manage humidity below 85%.",
            "Avoid wetting the foliage; use drip irrigation.",
            "Sanitize greenhouse structures thoroughly between seasons.",
            "Plant resistant varieties, though the pathogen frequently overcomes resistance."
        ],
        "caution": "Tomato leaf mold spores are known human allergens; use a mask when handling heavily infected plants. Use only products registered/labeled for this crop and disease in your jurisdiction, and follow the product label and local agricultural guidance.",
        "source": [
            {
                "organization": "Penn State Extension",
                "title": "Tomato Disease: Leaf Mold",
                "url": "https://extension.psu.edu/tomato-disease-leaf-mold"
            }
        ]
    }
}


def normalize_class_name(raw_label: str) -> str:
    """
    Normalizes a classifier label to match the knowledge base keys.
    Examples:
    'Tomato___Late_blight' -> 'tomato_late_blight'
    'Pepper__bell___Bacterial_spot' -> 'pepper_bell_bacterial_spot'
    'Potato___healthy' -> 'potato_healthy'
    """
    # Lowercase
    label = raw_label.lower()
    
    # Replace any sequence of underscores or non-alphanumeric chars with a single underscore
    label = re.sub(r'[^a-z0-9]+', '_', label)
    
    # Remove leading/trailing underscores
    label = label.strip('_')
    
    return label


def get_recommendation(predicted_class: str) -> dict:
    """
    Retrieves the treatment recommendation for the given classifier label.
    
    Args:
        predicted_class (str): The raw string label emitted by the classifier.
        
    Returns:
        dict: The structured recommendation from the knowledge base, or an 
              error/unknown structure if the class is not found.
    """
    normalized_key = normalize_class_name(predicted_class)
    
    if normalized_key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[normalized_key]
    else:
        # Fallback for unknown classes
        return {
            "crop": "Unknown",
            "disease": "Unknown",
            "status": "unknown",
            "summary": f"No knowledge base entry found for '{predicted_class}' (normalized: {normalized_key}).",
            "management": [],
            "prevention": [],
            "caution": "Consult local agricultural extension for unclassified anomalies.",
            "source": []
        }

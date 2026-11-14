"""Single source of truth for class metadata shared by training and serving."""

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_INFO = {
    "akiec": {
        "short": "Actinic keratosis",
        "full": "Actinic keratosis / Bowen's disease",
        "description": "Sun-damaged, pre-cancerous skin change.",
        "urgency": "priority",
        "urgency_label": "Priority referral",
        "advice": "Pre-cancerous change. A clinician should assess it, usually within weeks.",
    },
    "bcc": {
        "short": "Basal cell carcinoma",
        "full": "Basal cell carcinoma",
        "description": "The most common form of skin cancer; slow growing but needs treatment.",
        "urgency": "priority",
        "urgency_label": "Priority referral",
        "advice": "Suspected skin cancer. Arrange a dermatology review.",
    },
    "bkl": {
        "short": "Benign keratosis",
        "full": "Benign keratosis (seborrhoeic keratosis / solar lentigo)",
        "description": "A common, harmless rough or dark skin growth.",
        "urgency": "routine",
        "urgency_label": "Routine review",
        "advice": "Usually harmless, but any changing lesion should still be checked.",
    },
    "df": {
        "short": "Dermatofibroma",
        "full": "Dermatofibroma",
        "description": "A small, firm, benign fibrous nodule.",
        "urgency": "routine",
        "urgency_label": "Routine review",
        "advice": "Typically benign. Monitor for changes.",
    },
    "mel": {
        "short": "Melanoma",
        "full": "Melanoma",
        "description": "The most dangerous form of skin cancer; early detection is decisive.",
        "urgency": "urgent",
        "urgency_label": "Urgent referral",
        "advice": "Suspected melanoma. Seek a dermatology assessment as soon as possible.",
    },
    "nv": {
        "short": "Melanocytic nevus",
        "full": "Melanocytic nevus (mole)",
        "description": "A common benign mole.",
        "urgency": "routine",
        "urgency_label": "Routine review",
        "advice": "Most moles are harmless. Watch for asymmetry, border change or colour change.",
    },
    "vasc": {
        "short": "Vascular lesion",
        "full": "Vascular lesion",
        "description": "A benign blood-vessel growth such as an angioma.",
        "urgency": "routine",
        "urgency_label": "Routine review",
        "advice": "Usually benign. Have it checked if it bleeds or grows.",
    },
}

URGENCY_ORDER = {"urgent": 3, "priority": 2, "routine": 1}

ALPHA = 0.1
TARGET_COVERAGE = 1.0 - ALPHA

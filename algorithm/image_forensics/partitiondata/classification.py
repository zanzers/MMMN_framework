

def classify_ela_features(ela_features: dict, threshhold: dict = None) -> dict:


    default_thresholds = {
        "mean_error": [30, 50],
        "std_error": [25, 40],
        "tampered_ratio": [0.005, 0.01],
        "total_error": [5e7, 1e8]
    }

    
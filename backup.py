
# that generate the same value to determine if the image are tamperd 

# def ela_score_report(report: dict) -> None:
#     print("\n" + "="*30)
#     print("   📊 ELA Score Report")
#     print("="*30)
    
#     mean = report.get("mean_error", 0)
#     std = report.get("std_error", 0)
#     max_val = report.get("max_error", 0)
#     energy = report.get("total_error_energy", 0)
#     ratio = report.get("tampered_area_ratio", 0)

#     print(f"{'Mean Error:':<22}{mean:.2f}")
#     print(f"{'Std Deviation:':<22}{std:.2f}")
#     print(f"{'Max Error:':<22}{max_val:.2f}")
#     print(f"{'Total Error Energy:':<22}{energy:.0f}")
#     print(f"{'Tampered Area Ratio:':<22}{ratio:.4f}")
    
#     # --- Scoring logic ---
#     score = 0

#     # Threshold-based scoring (weights can be adjusted)
#     if mean > 50: score += 2
#     elif mean > 30: score += 1

#     if std > 40: score += 2
#     elif std > 25: score += 1

#     if ratio > 0.01: score += 2
#     elif ratio > 0.005: score += 1

#     if energy > 50000000: score += 1
#     if energy > 100000000: score += 2

#     # Interpretation
#     if score <= 2:
#         level = "✅ Low - Likely Original"
#     elif score <= 5:
#         level = "🟡 Medium - Possibly Filtered/Compressed"
#     else:
#         level = "🔴 High - Likely Tampered"

#     print("-" * 30)
#     print(f"{'Risk Score:':<22}{score} / 7")
#     print(f"{'Assessment:':<22}{level}")
#     print("=" * 30 + "\n")

import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from algorithm.image_forensics.global_algo import create_saveImg

def resample_detect(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Laplacian enhancement to highlight resampling artifacts
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(laplacian)

    # FFT to transform to frequency domain
    fft_img = fftshift(fft2(abs_lap))
    mag = np.abs(fft_img)
    mag = np.log1p(mag)

    # Normalize the spectrum for display
    spectrum = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)













# In your feature_extractor.py or similar

def extractResample_features(spectrum: np.ndarray) -> dict:
    # Rough periodicity estimation
    central_band = spectrum[
        spectrum.shape[0]//2 - 30 : spectrum.shape[0]//2 + 30,
        spectrum.shape[1]//2 - 30 : spectrum.shape[1]//2 + 30
    ]
    periodicity = float(np.std(central_band))

    # Save the spectrum visualization
    create_saveImg([("resample_spectrum.jpg", spectrum)], remove_after=True)

    features = {
        "features_name": "RESAMPLING DETECTION",
        "periodicity_std": round(periodicity, 4),
        "resampled": periodicity > 10.0  # ← Adjustable threshold based on empirical tests
    }

    return features



# classification.py

def partition_ela_features(ela_features: dict, thresholds: dict = None) -> dict:
    """
    Classifies ELA feature values into a tampered or original label based on thresholds.
    
    Args:
        ela_features (dict): Dictionary containing ELA metrics.
        thresholds (dict): Optional custom thresholds.

    Returns:
        dict: Partition result with computed score and label.
    """
    # Default thresholds
    default_thresholds = {
        "mean_error": [30, 50],  # [low, high]
        "std_error": [25, 40],
        "tampered_ratio": [0.005, 0.01],
        "total_error": [5e7, 1e8]
    }

    if thresholds is None:
        thresholds = default_thresholds

    score = 0

    score += 2 if ela_features["mean_error"] > thresholds["mean_error"][1] else 1 if ela_features["mean_error"] > thresholds["mean_error"][0] else 0
    score += 2 if ela_features["std_error"] > thresholds["std_error"][1] else 1 if ela_features["std_error"] > thresholds["std_error"][0] else 0
    score += 2 if ela_features["tampered_ratio"] > thresholds["tampered_ratio"][1] else 1 if ela_features["tampered_ratio"] > thresholds["tampered_ratio"][0] else 0
    score += 2 if ela_features["total_error"] > thresholds["total_error"][1] else 1 if ela_features["total_error"] > thresholds["total_error"][0] else 0

    label = 1 if score >= 5 else 0  # You can calibrate this score cutoff as needed

    return {
        "features_name": ela_features["features_name"],
        "score": score,
        "label": label
    }





import cv2
import numpy as np
from algorithm.image_forensics.global_algo import *

def extractELA_features(ela_img: np.ndarray) -> dict:

    gray = grayScale(ela_img)
    mean_error, std_error, max_error, total_error = msmt_calculation(gray)
    tamper_map = image_threshold(gray, 200, 255, cv2.THRESH_BINARY)
    tamper_ratio = image_ratio(float, np.sum, tamper_map, 255, gray.shape[0], gray.shape[1])

    create_saveImg([
        ('gray_output.jpg', gray),
        ('bw_output.jpg', tamper_map)
    ], remove_after=True)

    ela_features = {
        "features_name": "ERROR LEVEL ANALYSIS",
        "mean_error": mean_error,
        "std_error": std_error,
        "max_error": max_error,
        "total_error": total_error,
        "tampered_ratio": tamper_ratio,
        "gray_path": 'gray_output',
        "tamper_map": 'bw_output'
    }

    save_features("Ela", ela_features)

    return ela_features



def extractNOISE_features(norm: np.ndarray, prnu: np.ndarray, wavelet: np.ndarray) -> tuple[dict, dict]: 


    mean_norm, std_norm, max_norm, total_norm = msmt_calculation(norm)
    mean_prnu, std_prnu, max_prnu, total_prnu = msmt_calculation(prnu)


    norm_thres = image_threshold(norm, 200, 255)
    prnu_thres = image_threshold(prnu, 200, 255)
    ratio_norm = image_ratio(float, np.sum, norm_thres, 255, norm.shape[0], norm.shape[1])
    ratio_prnu = image_ratio(float, np.sum, prnu_thres, 255, prnu.shape[0], prnu.shape[1])    


    create_saveImg([
        ("norm_output.jpg", norm),
        ("prnu_output.jpg", prnu),
        ("normThresh_output.jpg", norm_thres),
        ("prnuThresh_output.jpg", prnu_thres),
    ], remove_after=True)


    noise_features = {
        "features_name": "NOISE ANALYSIS",
        "mean_norm_error": mean_norm,
        "std_norm_error": std_norm,
        "max_norm_error": max_norm,
        "total_norm_error": total_norm,
        "local_norm_ratio": ratio_norm,

        "mean_prnu_error": mean_prnu,
        "std_prnu_error": std_prnu,
        "max_prnu_error": max_prnu,
        "total_prnu_error": total_prnu,
        "local_prnu_ratio": ratio_prnu,

        "wavelet_noise": wavelet

    }

    save_features("Noise", noise_features)

    return noise_features


def extractCOPYMOVE_features(mask: np.ndarray) -> dict:
    
    tamper_ratio = np.sum(mask) / 255 / (mask.shape[0] * mask.shape[1])
    total_tampered= int(np.sum(mask) / 255)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_area = 0
    bbox_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox_area = w * h
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        create_saveImg([
            ("copy_move_output.jpg", bbox_img),
        ], remove_after=True)

    
    copyMove_Features = {
        "features_name": "COPY-MOVE DETECTION",
        "tamper_ratio": round(tamper_ratio),
        "tampered_area": total_tampered,
        "bounding_box": bbox_area,
        "tampered": tamper_ratio
    }
    save_features("Copy-Move", copyMove_Features)
    return copyMove_Features

def extractResample_features(spectrum: np.ndarray) -> dict:

    central_band = spectrum[
        spectrum.shape[0]//2 - 30 : spectrum.shape[0]//2 + 30,
        spectrum.shape[1]//2 - 30 : spectrum.shape[1]//2 + 30
    ]

    periodicity = float(np.std(central_band))
    create_saveImg([
        ("resample_output.jpg", spectrum)
    ])

    resample_Features = {
        "features_name": "RESAMPLING DETECTION",
        "periodicity_std": round(periodicity, 4),
        "resampled": periodicity
        
    }


    save_features("Resample", resample_Features)
    return resample_Features
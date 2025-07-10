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

    return noise_features






from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import cv2
import numpy as np
import os
from typing import Union
import shutil
from scipy.fftpack import fft2, fftshift
import json


# Handle all repeated funtions!

def norm(img:np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    normalized = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return normalized

def norm_variance(varience: int, dtype: type = None) -> np.ndarray:
    norm_var = cv2.normalize(varience, None, 0, 255, cv2.NORM_MINMAX)
    return norm_var.astype(dtype) if dtype is not None else norm_var

def grayScale(img: np.ndarray, dtype: type = None ) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(dtype) if dtype is not None else gray

def image_quality(img: np.ndarray, path: np.ndarray, compress: int) -> np.ndarray:
    return cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), compress])

def image_threshold(img: np.ndarray, thresh_val: int, max_val: int, thresh_type: type= None) -> np.ndarray:
    
    if thresh_type is None:
        thresh_type = cv2.THRESH_BINARY
    
    _, thresholded = cv2.threshold(img, thresh_val, max_val, thresh_type)   
    return thresholded

def image_ratio(cast_type: type, sum_func: callable, img: np.ndarray, value: float, shape_height: int, shape_width: int) -> float:

    ratio = sum_func(img) / value / (shape_height * shape_width)
    return cast_type(ratio)

def create_saveImg(temp_imgs: Union[tuple[str, any], list[tuple[str, any]]], remove_after: bool = False) -> None:
    if isinstance(temp_imgs, tuple):
        temp_imgs = [temp_imgs]

    save_dir = os.path.join(
        "algorithm", 
        "image_forensics",
        "temp_result")
    os.makedirs(save_dir, exist_ok=True)

    for filename, img_data in temp_imgs:
        if not isinstance(filename, str):
            print(f"[CAUTION] INVALID FILENAME: {filename}")

        cv2.imwrite(filename, img_data)

        target_path = os.path.join(save_dir, os.path.basename(filename))
        shutil.copy(filename, target_path)

        if remove_after:
            os.remove(filename)
 
def check_input(extension: str) -> str:
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    
    ext = os.path.splitext(extension)[1].lower()

    if ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    else:
        return "unsupported"
    
def msmt_calculation(img: np.ndarray) -> dict[str, int]:

    mean_error = float(np.mean(img))
    std_error = float(np.std(img))
    max_error = float(np.max(img))
    total_error = float(np.sum(img))


    return mean_error, std_error, max_error, total_error

def euclidenMask_pca(components_val: int, blocks: np.ndarray ,threshold: int) -> np.ndarray:

    pca = PCA(n_components=components_val)
    reduced = pca.fit_transform(blocks)

    dist = euclidean_distances(reduced)
    np.fill_diagonal(dist, np.inf)
    macthes = np.argwhere(dist < threshold)


    return macthes

def laptican_magitude(img: np.ndarray) -> np.ndarray:

    laptican = cv2.Laplacian(img, cv2.CV_64F)
    abs_lap = np.abs(laptican)

    fft_img = fftshift(fft2(abs_lap))
    magnitude = np.abs(fft_img)
    magnitude = np.log1p(magnitude)

    return magnitude




def save_features(label: str, feature_data: dict, reset: bool = False):
        save_dir = os.path.join(
            "algorithm",
            "image_forensics",
            "partitiondata"
        )
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "features_data.json")    

        existing_data = {} if reset or not os.path.exists(json_path) else json.load(open(json_path, "r"))
        existing_data[label] = feature_data
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=4)
        
        print(f"'{label}' features saved to {json_path}")






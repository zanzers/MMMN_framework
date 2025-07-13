from algorithm.image_forensics.global_algo import *
import cv2
import numpy as np
import pywt

class Image_Forensics:
    def __init__(self, img_input: np.ndarray) -> None:
        self.img_input = img_input
        self.nor_img = norm(img_input)


    def ela_anaylze(self, compression: int , multp: int) -> np.ndarray:

        image_quality(self.nor_img, 'ela_result.jpg', compression)
        compressed = cv2.imread('ela_result.jpg')
        diff = cv2.absdiff(self.nor_img, compressed)
        ela_img = np.clip(diff * multp, 0, 255).astype(np.uint8)    
        return ela_img
    


    def noise_analyze(self, local_mean: int, varience: int) -> np.ndarray: 
        
        local_gray = grayScale(self.img_input, np.float32)
        kernel = np.ones((local_mean, local_mean), np.float32) / (local_mean * local_mean)

        # local variance
        mean = cv2.filter2D(local_gray, -1, kernel)    
        mean_sq = cv2.filter2D(local_gray**2, -1, kernel)    
        var = mean_sq - mean**2
        norm_var = norm_variance(var, np.uint8)

        # PRNU Photo Response Non-Uniformity
        prnu_gray = grayScale(self.nor_img, None)
        denoised = cv2.GaussianBlur(prnu_gray, (3, 3), 0)
        residual = prnu_gray - denoised
        prnu_map = norm_variance(residual, np.uint8)

        # wavelet noise
        wavelet_gray = grayScale(self.img_input, np.float32)
        coeffs2 = pywt.dwt2(wavelet_gray, 'db1')
        LL,(LH, HL, HH) = coeffs2

        std_LH = np.std(LH)
        std_HL = np.std(HL)
        std_HH = np.std(HH)
        noise_estimation = (std_LH + std_HL + std_HH) / 3


        return norm_var, prnu_map, noise_estimation
    
    
    def compyMove(self, block_size: int, step: int, threshold: int) -> np.ndarray:
        
        norm_gray = grayScale(self.img_input)
        h, w = norm_gray.shape
        blocks = []
        position = []

        for y in range(0, h - block_size, step):
            for x in range(0, w - block_size, step):
                block = norm_gray[y:y + block_size, x:x + block_size]
                blocks.append(block.flatten())
                position.append((y, x ))
                
        block = np.array(blocks)
        position = np.array(position)

        macthes = euclidenMask_pca(10, blocks, 1000)
     
        mask = np.zeros_like(norm_gray, dtype=np.uint8)
        for i, j in macthes:
           y1, x1 = position[i] 
           y2, x2 = position[j]
           cv2.rectangle(mask, (x1, y1), (x1 + block_size, y1 + block_size), 255, -1)
           cv2.rectangle(mask, (x2, y2), (x2 + block_size, y2 + block_size), 255, -1)
        
        return mask
    
    def resample_detect(self) -> np.ndarray:
        
        resample_gray = grayScale(self.img_input)
        mag = laptican_magitude(resample_gray)
        spectrum = norm_variance(mag, np.uint8)

        return spectrum

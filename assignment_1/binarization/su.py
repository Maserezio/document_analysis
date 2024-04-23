import numpy as np
import cv2
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class Su_Binarizator:
    def __init__(self, window_size, N_min):
        self.window_size = window_size
        self.N_min = N_min    

    def get_contrast_values(self, input_image):
        cont_values = np.empty_like(input_image, dtype=np.float32)
        
        for i in tqdm(range(input_image.shape[0]), desc="Calculating Contrast Values"):
            for j in range(input_image.shape[1]):
                start_row = max(0, i - self.window_size // 2)
                end_row = min(len(input_image), i + self.window_size // 2 + 1)
                start_col = max(0, j - self.window_size // 2)
                end_col = min(len(input_image[0]), j + self.window_size // 2 + 1)
                
                window = input_image[start_row:end_row, start_col:end_col]
                
                cont_values[i, j] = (np.max(window) - np.min(window)) / (np.max(window) + np.min(window)) if np.max(window) + np.min(window) != 0 else 0
        
        return cont_values

    def su_thresholding(self, otsu_image, input_image):
        bin_image = np.empty_like(input_image, dtype=np.uint8)
        
        for i in tqdm(range(len(input_image)), desc="Applying SU Thresholding"):
            for j in range(len(input_image[0])):
                start_row = max(0, i - self.window_size // 2)
                end_row = min(len(input_image), i + self.window_size // 2 + 1)
                start_col = max(0, j - self.window_size // 2)
                end_col = min(len(input_image[0]), j + self.window_size // 2 + 1)

                E_window = otsu_image[start_row:end_row, start_col:end_col]
                I_window = input_image[start_row:end_row, start_col:end_col]

                E = E_window.flatten()
                I = I_window.flatten()

                Ne = np.sum(E == 255)

                E_mean = np.sum(I * (1 - E)) / Ne if Ne >= 1 else 0
                E_std = np.sqrt(np.sum(((I - E_mean) * (1 - E)) ** 2) / 2)

                if Ne >= self.N_min and input_image[i, j] <= E_mean + E_std / 2:
                    bin_image[i, j] = 0
                else:
                    bin_image[i, j] = 255
        
        return bin_image

    def fit(self, input_images):
        bin_images = []
        for image in tqdm(input_images, desc="Processing Images", leave=True):
            cont_values = self.get_contrast_values(image)
            cont_image = (cont_values * 255).astype(np.uint8)
            _, otsu_image = cv2.threshold(cont_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bin_image = self.su_thresholding(otsu_image, image)
            bin_images.append(bin_image)

        return bin_images

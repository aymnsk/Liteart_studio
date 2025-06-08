import cv2
import numpy as np
from PIL import Image

def apply_cartoon(img: Image.Image) -> Image.Image:
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

def apply_pencil_sketch(img: Image.Image) -> Image.Image:
    img = np.array(img)
    gray, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return Image.fromarray(gray)

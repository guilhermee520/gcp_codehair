import cv2
from image_related.image_constructor import get_mask, apply_mask
from image_related.image_predictor import extractDominantColor
from matplotlib import pyplot as plt
import os


def image_inference(filepath):
    image = cv2.imread(filepath)
    image = cv2.resize(image, (240, 240), interpolation = cv2.INTER_AREA)
    mask = get_mask(image)
    builded_image = apply_mask(image, mask)
    colorInformation = extractDominantColor(builded_image, hasThresholding=True)
    dominantColors = colorInformation[0].get('color') + colorInformation[1].get('color') + colorInformation[2].get('color') + colorInformation[3].get('color') + colorInformation[4].get('color')
    
    return dominantColors

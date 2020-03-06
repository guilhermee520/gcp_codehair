import cv2
import numpy as np
import keras
import cv2
 
height = 224
width = 224
extractor = keras.models.load_model('models/hairnet_matting.hdf5')

def get_mask(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)

    pred = extractor.predict(im)
    mask = pred.reshape((224, 224))
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    return mask


def apply_mask(image, mask):
    heigth_img = len(mask)
    width_img = len(mask[0])
    segmented_image = np.array([[image[h][w] if mask[h][w] != np.float(0) 
        else np.array([np.float32(0) 
            for x in range(3)])
                for w in range(width_img)]  
                    for h in range(heigth_img)])

    return segmented_image



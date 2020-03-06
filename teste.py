from sklearn.externals import joblib
import cv2
from image_related.image_constructor import get_mask, apply_mask
from image_related.image_predictor import extractDominantColor, plotColorBar
from matplotlib import pyplot as plt
from image_related.inference import image_inference
import random

imagem_number = random.randint(000000,999999)

PATH = "test/test_images/cabelo_claro/3_1_1_20170117192731931.jpg"

model = joblib.load('models/classificador_cor_cabelo.weights')
print(PATH)
image = cv2.imread(PATH)
mask = get_mask(image)
print("Get mask")
builded_image = apply_mask(image, mask)
plt.axis("off")
plt.imshow(builded_image)
print("apply_mask")
colorInformation = extractDominantColor(builded_image, hasThresholding=True)
dominantColors = colorInformation[0].get('color') + colorInformation[1].get('color') + colorInformation[2].get('color') + colorInformation[3].get('color') + colorInformation[4].get('color')


infered_data = image_inference(PATH)
print("model predicting")
result = model.predict_proba([infered_data])[0]
hair_color = "Cabelo Claro" if result[0] >= 0.50 else "Cabelo Escuro"
print(hair_color)

colour_bar = plotColorBar(colorInformation)
new_image = "image_temp"+str(imagem_number)+".jpg"
plt.axis("off")
plt.imshow(colour_bar)

plt.imsave("static/"+new_image,colour_bar)
plt.title(hair_color)
plt.show()


hair_color = "Cabelo Claro" if result[0] >= 0.50 else "Cabelo Escuro"
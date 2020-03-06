from sklearn.externals import joblib
import cv2
from image_related.image_constructor import get_mask, apply_mask
from image_related.image_predictor import extractDominantColor, plotColorBar
from matplotlib import pyplot as plt
from image_related.inference import image_inference

def inference(PATH,model,filename):



	#model = joblib.load('models/classificador_cor_cabelo.weights')
	print(PATH)
	image = cv2.imread(PATH)
	mask = get_mask(image)
	print("Get mask")
	builded_image = apply_mask(image, mask)
	cv2.imwrite('static/inference_'+filename+'.png',builded_image)
	print("apply_mask")
	colorInformation = extractDominantColor(builded_image, hasThresholding=True)
	dominantColors = colorInformation[0].get('color') + colorInformation[1].get('color') + colorInformation[2].get('color') + colorInformation[3].get('color') + colorInformation[4].get('color')


	infered_data = image_inference(PATH)
	print("model predicting")
	result = model.predict_proba([infered_data])[0]
	hair_color = "Cabelo Claro" if result[0] >= 0.50 else "Cabelo Escuro"
	print(hair_color)

	colour_bar = plotColorBar(colorInformation)
	paleta_image = "paleta_"+filename+".png"
	inference_image = 'inference_'+filename+'.png'
	

	plt.axis("off")
	plt.imsave("static/"+paleta_image,colour_bar)



	hair_color = "Cabelo Claro" if result[0] >= 0.50 else "Cabelo Escuro"
	return [hair_color,paleta_image,inference_image]
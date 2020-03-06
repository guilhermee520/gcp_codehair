from flask import Flask
import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
from image_related.inference import image_inference

#carrega modelo de inferência
model = joblib.load('models/classificador_cor_cabelo.weights')

#configurações inicias do app
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#pagina inicial, faz download das imagens
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file', filename=filename))

    return render_template("index.html")

#valida a extensão da imagem
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#prepara a imagem para ser exibida na web
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

#pagina de resultados, exibe a imagem enviada pelo usuário
@app.route('/show/results/previsao')
def upload_file(filename):
	infered_data = image_inference(UPLOAD_FOLDER+filename)
	result = model.predict_proba([infered_data])[0]
	hair_color = "Cabelo Claro" if result[0] >= 0.50 else "Cabelo Escuro"
    #print(hair_color)
	return render_template('results.html')#, filename=filename, hair_color = hair_color)

if __name__ == "__main__":
    app.run(threaded=False)

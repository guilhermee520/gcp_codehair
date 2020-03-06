from flask import Flask
import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
from image_inference import inference

#carrega modelo de inferência
print("Model load")
model = joblib.load('models/classificador_cor_cabelo.weights')
print("Model loaded")

#configurações inicias do app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/temp_files'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png','.jfif'}
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
@app.route('/show/results/<filename>')
def upload_file(filename):
    PATH = UPLOAD_FOLDER +'/'+filename
    uploaded_path = 'temp_files'+'/'+filename
    print("loading inference")
    image_hair = inference(PATH,model,filename)
    #a funcao main devolve uma lista com 2 valores , o [0] é a cor do cabelo, e a [1] devolve o caminho da pasta [2] devolve a inferencia da imagem
    return render_template('results.html', filename=filename,paleta=image_hair[1],image_inference=image_hair[2],image = uploaded_path,cabelo=image_hair[0])


if __name__ == "__main__":
        app.run(host='127.0.0.1', port=8080, debug=True)


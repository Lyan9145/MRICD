from flask import *
from PIL import Image
import os
import model

temp_dir = '/temp'
upload_image_path = os.path.join(temp_dir, 'upload_image.png')
pred_image_path = os.path.join(temp_dir, 'pred_image.png')

app = Flask(__name__)
model = model.ModelUNet()

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    img = request.files.get('image')
    if not img:
        return jsonify({'error': 'No image uploaded'})
    if not (os.path.exists(temp_dir)):
        os.makedirs(temp_dir)
    img.save(upload_image_path)
    return jsonify({'success': 'Image uploaded, processing...'})
    
    





if __name__ == '__main__':
    model.load_ckpt()
    app.run(host='localhost', port=5674)


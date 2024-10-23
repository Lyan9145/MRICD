from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
from unet_model import ModelUNet

app = Flask(__name__)
model = ModelUNet()

app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    PORT=5674,
    HOST='localhost',
    THREADS=4
)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
async def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload():
    required_files = ['flair', 't1', 't1ce', 't2']
    images = {}
    try:
        for file_key in required_files:
            if file_key not in request.files:
                return jsonify({'error': f'Missing file: {file_key}'}), 400
            file = request.files[file_key]
            if file.filename == '':
                return jsonify({'error': f'No selected file for {file_key}'}), 400
            if file and allowed_file(file.filename):
                img = Image.open(file.stream)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((240, 240))
                img_io = io.BytesIO()
                img.save(img_io, 'JPEG', quality=100)
                img_io.seek(0)
                images[file_key] = img_io
            else:
                return jsonify({'error': f'Invalid file type for {file_key}. Only .jpg and .jpeg are allowed.'}), 400
        # TODO: Run model processing with images['flair'], images['t1'], images['t1ce'], images['t2']
        processed_image = model.predict(images['flair'], images['t1'], images['t1ce'], images['t2'])  # Replace with model output
        return send_file(processed_image, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Max size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found.'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed.'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    model.load_ckpt()
    app.run(host=app.config['HOST'], port=app.config['PORT'], threaded=True)
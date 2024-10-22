from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unet_model import ModelUNet

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    PORT=5674,
    HOST='localhost',
    THREADS=4
)
executor = ThreadPoolExecutor(app.config['THREADS'])

model_instance = ModelUNet()
try:
    model_instance.load_ckpt()
except Exception as e:
    raise RuntimeError(f"Failed to load model checkpoint: {e}")

def process_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        result = model_instance.predict(img, threshold=0.68)
        img_io = io.BytesIO()
        result.save(img_io, format='PNG')
        img_io.seek(0)
        return img_io
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
async def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading.'}), 400

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty image file.'}), 400

        loop = asyncio.get_event_loop()
        img_io = await loop.run_in_executor(executor, process_image, image_bytes)
        return send_file(img_io, mimetype='image/png')
    except RuntimeError as re:
        return jsonify({'error': str(re)}), 500
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {e}"}), 500

if __name__ == '__main__':
    app.run(host=app.config['HOST'], port=app.config['PORT'])
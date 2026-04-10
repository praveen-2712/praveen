from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import io
from PIL import Image
from utils.inference_pipeline import run_pipeline

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Run the full hybrid AI pipeline
        results = run_pipeline(image)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

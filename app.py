import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    try:
        os.system("dvc repro")
        return jsonify({"message": "Training executed successfully via DVC pipeline!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image_data = None
        if request.is_json and request.json and 'image' in request.json:
            image_data = request.json['image']
        elif 'image' in request.files:
            file = request.files['image']
            file.save(clApp.filename)
            result = clApp.classifier.predict()
            return jsonify(result)
        
        if image_data:
            decodeImage(image_data, clApp.filename)
            result = clApp.classifier.predict()
            return jsonify(result)
        else:
            return jsonify({"error": "No image data provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

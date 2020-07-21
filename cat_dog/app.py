from flask import Flask, render_template, request
from testModel import load_model

app = Flask(__name__)

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.get_data()

@app.route('/upload', methods=['POST'])
def add_image():
    file = request.files['fileName']
    response = file.read()
    return response


if __name__ == "__main__":
    app.run(debug=True)
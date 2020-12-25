import base64
from io import BytesIO

from flask import Flask,request, jsonify, json
from PIL import Image

import neuro

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        result = ''
        buf = BytesIO()
        image = Image.open(request.files['image'])
        result = neuro.tn(image=image)
        if result['Prediction'] == 'hot_dog':
            result = 'Хот-Дог'
        else:
            result = 'Не Хот-Дог'
        image.save(buf, format='JPEG')
        return jsonify(result=result, img=base64.b64encode(buf.getvalue()).decode('utf-8'))
    else:
        return "You need use POST request"


if __name__ == '__main__':
    app.run()

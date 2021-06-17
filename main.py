from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import numpy as np
import cv2
import json
import os
import argparse
import datetime
import coco_labels

import tflite_detector

DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow object detection API')

    parser.add_argument('--debug', dest='debug',
                        help='Run in debug mode.',
                        required=False, action='store_true', default=True)

    parser.add_argument('--port', dest='port',
                        help='Port to run on.', type=int,
                        required=False, default=DEFAULT_PORT)

    parser.add_argument('--host', dest='host',
                        help='Host to run on, set to 0.0.0.0 for remote access', type=str,
                        required=False, default=DEFAULT_HOST)

    args = parser.parse_args()
    return args

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

#웹 서비스
@app.route('/')
def index():
    return render_template('index.html')

#디텍션 처리
@app.route('/detection', methods=['POST'])
def detection():
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    filename_first = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    rects, classes, scores = tflite_detector.inference(image)

    labels = []
    for cat in classes:
        label_txt = coco_labels.labels[int(cat)]
        labels.append(label_txt)

    #write json
    json_path = os.path.join('outputs', filename_first + '.json')
    result = {'rects':rects.tolist(), 'labels':labels, 'scores':scores.tolist()}
    json.dump(result, open(json_path, 'w'))
    
    #write image
    #drawing...
    dst = image
    tflite_detector.draw_boxes(dst, rects, classes, scores)
    dst_path = os.path.join('outputs', filename_first + '.jpg')
    cv2.imwrite(dst_path, dst)
    
    return redirect(url_for('show_detection', filename_first=filename_first))
    
#outputs 폴더를 일반 웹서버 형식으로 오픈
@app.route('/outputs/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    output_path = os.path.join(app.root_path, 'outputs')
    return send_from_directory('outputs', filename)

#디텍션 결과 보여주기
@app.route('/detection_result/<filename_first>')
def show_detection(filename_first):
    json_path = os.path.join('outputs', filename_first + '.json')
    dst_path = os.path.join('..', 'outputs', filename_first + '.jpg')

    result = json.load(open(json_path, 'r'))
    message = str(result['labels'])+'<br>'+str(result['scores'])
    return render_template("result.html", image_path=dst_path, message=message)
    # return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


# API 서비스
@app.route('/object_detection', methods=['POST'])
def infer():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rects, classes, scores = tflite_detector.inference(img)

    result = {'rects':rects.tolist(), 'classes':classes.tolist(), 'scores':scores.tolist()}
    response = json.dumps(result)

    return Response(response=response, status=200, mimetype="application/json")

# start flask app
def main():
    os.makedirs('outputs', exist_ok=True)
    tflite_detector.load_model("data/centernet_mobilenetv2_fpn_od/model.tflite")
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

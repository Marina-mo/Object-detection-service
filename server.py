import json
import torchvision
import torch
from flask import Flask, request, jsonify
import requests
import io
import torchvision.transforms as transforms
from PIL import Image
from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics


app = Flask(__name__, static_url_path="")
metrics = GunicornInternalPrometheusMetrics(app)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#model.load_state_dict(torch.load('./maskrcnn_resnet50.pth'))

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

def prepare(img_data):
    image = Image.open(io.BytesIO(img_data))
    img_data = transform_pipeline(image).unsqueeze(0)
    return img_data


@app.route("/predict", methods=['POST'])
@metrics.gauge("api_in_progress", "requests in progress")
@metrics.counter("app_http_inference_count", "number of invocations")
def predict():
    url = request.get_json(force=True)
    data = requests.get(url['url']).content
    image_data = prepare(data)


    model.eval()
    with torch.no_grad():
        output = model(image_data)[0]

    sel_labels = output['labels'][output['scores'] > 0.75]
    return jsonify({
        "objects": [inst_classes[index.item()] for index in sel_labels]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
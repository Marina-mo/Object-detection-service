## Object detection service

This service uses MaskRCNN to determine which objects are depicted in the image. It communicates with client through REST-API
 
**User guide**: 

1) Build a docker image with the following command:

    `docker build -t <server_name:version>`

2) Run this docker image with the following command:

    `docker run -p 8080:8080 `

3) Now you can send a url of image you want to processede:

    `curl -XPOST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"url": <"you url">}'`

4) You'll get a response in json format contains the list of objects in yout image:
    `{"objects": [
        "bird",
        "boat",
        "boat",
        "person",
        "person",
        "person",
        "person",
        "cell phone",
        "backpack",
        "handbag",
        "boat"
    ]}`

5) Service metrics can be found via /metrics on 8080 port. The most important metric for us is app_http_inference_count - the number of http endpoint invocations.
    `curl http://localhost:8080/metrics`    
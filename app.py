import cv2
from flask import Flask, Response

app = Flask(__name__)

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

#my_project_id = "/subscriptions/7c20a5c8-956e-4eb4-b5e4-946870fff696/resourceGroups/ClassificatinTest/providers/Microsoft.CognitiveServices/accounts/ClassificatinTest" 
my_project_id = "78c68f9f-7790-40da-ae8d-29b352b8c20c"
#prediction_key = "921e4a0dd5f04cd190e915cc04933deb"
prediction_key = "cff6e899bb3f4404af5abdc15eb23784"

ENDPOINT = "https://cv-morii-1.cognitiveservices.azure.com/"

#https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/quickstarts/image-classification?tabs=visual-studio&pivots=programming-language-python
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
#https://teratail.com/questions/262076
#predictor = CustomVisionPredictionClient(ENDPOINT, prediction_key)
my_iteration_name = "Iteration1"
#my_iteration_name = "376b0226-1d6e-45b3-80ef-2900a409aae6"
#https://dev.to/stratiteq/puffins-detection-with-azure-custom-vision-and-python-2ca5

webcamid = 0
cap = cv2.VideoCapture(webcamid)

def getFrames():
    while True:
        ret, frame = cap.read()
        #frame = cv2.flip(frame, 1) # horizontal flip
        img_h, img_w = frame.shape[:2]
        frame = cv2.resize(frame.copy(), (int(img_w/2), int(img_h/2)))
        img_h, img_w = frame.shape[:2]
        ret, jpeg = cv2.imencode('.jpg', frame)
        #jpeg = jpeg[1].tostring()
        #jpeg = jpeg.tobytes()

        #with open(jpeg, mode="rb") as test_data:
        #    results = predictor.detect_image(my_project_id, my_iteration_name, test_data)
        results = predictor.detect_image(my_project_id, my_iteration_name, jpeg)

        fontType = cv2.FONT_HERSHEY_COMPLEX

        result_image = frame
        for prediction in results.predictions:
            if prediction.probability > 0.7:
                bbox = prediction.bounding_box
                result_image = cv2.rectangle(result_image, (int(bbox.left * img_w), int(bbox.top * img_h)), (int((bbox.left + bbox.width) * img_w), int((bbox.top + bbox.height) * img_h)), (0, 255, 0), 3)
                label = prediction.tag_name
                x_text = int(bbox.left * img_w)
                y_text = int(bbox.top * img_h - 12)
                if y_text < 6:
                    y_text = 6
                if label == "makino":
                    cv2.putText(result_image, label,(x_text,  y_text), fontType, 1, (0, 0, 255),4)
                #print(prediction.tag_name)
                #cv2.imwrite('result.png', result_image)

        result_image = cv2.resize(result_image.copy(), (int(result_image.shape[1]*3), int(result_image.shape[0]*3)))

        ret, jpeg2 = cv2.imencode('.jpg', result_image)

        yield b'--boundary\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg2.tostring() + b'\r\n\r\n'

@app.route('/')
def video_feed():
    return Response(getFrames(), mimetype='multipart/x-mixed-replace; boundary=boundary')
    #return "hello"

if __name__ == '__main__':
    app.run(debug=True)
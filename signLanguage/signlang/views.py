from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import threading
import tensorflow as tf

@gzip.gzip_page
def Home(request):
    return render(request, 'index.html')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
        self.model = tf.keras.models.load_model('sign_language.h5')
    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        # print(image.shape)
        _, jpeg = cv2.imencode('.jpg', image)
        ri = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (28,28))
        ri = np.expand_dims(ri, axis = (0,3))
        out = self.model.predict(ri)
        out = out.astype('uint8')
        # print(out)        
        return jpeg.tobytes(),out

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

# Camera Object
cam = VideoCamera()

def test_stream(request):
    try:
        # cam = VideoCamera()
        return StreamingHttpResponse(generateImage(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def stream_response(request):
    yield "%s\n" % "A"

def text_stream(request):
    # cam = VideoCamera()
    resp = StreamingHttpResponse(predictSign(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    return resp

# Returns Image to html
def generateImage(camera):
    while True:
        frame,out = camera.get_frame()
        
        yield "<html><body>\n"
        yield "<div>%s</div>\n" % out
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        yield "</body></html>\n"

# Returns Predicted Handsign
def predictSign(camera):
    while True:
        frame,out = camera.get_frame()
        listt = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
        # print(len(out[0]),len(listt))
        if(1 not in out[0]):
            yield ""
        else:    
            yield "%s\n" % listt[list(out[0]).index(1)]


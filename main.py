from flask import Flask, render_template, Response
import cv2
import hiai
from hiai.nn_tensor_lib import DataType
from atlasutil import camera, ai, dvpp_process
import threading
import time
import os

class VideoCamera(object):
    def __init__(self):
        self.camera_width = 1280
        self.camera_height = 720
        presenter_config = './face_detection.conf'
        print("init camera")
        self.cap = camera.Camera(id=0, fps=5, width=self.camera_width, height=self.camera_height,
                            format=camera.CAMERA_IMAGE_FORMAT_YUV420_SP)
        if not self.cap.IsOpened():
            print("Open camera 0 failed-----")
            return
        self.dvpp_handle = dvpp_process.DvppProcess(self.camera_width, self.camera_height)
        self.graph = ai.Graph('./model/darknet.om')
        self.yolov3 = ai.Yolov3Manager()
        self.yolov3.labels = self.load_labels("coco.names")
    def load_labels(self, filename):
        labels=[]
        with open(filename,"r") as f:
            labels = f.read().splitlines()
        return labels  
    def __del__(self):
        self.video.release()

    def get_frame(self):
        yuv_img = self.cap.Read()
        #print('yuv shape:',yuv_img.shape)
        #print('yuv type:',type(yuv_img))
        #orig_image = self.dvpp_handle.Yuv2Jpeg(yuv_img)
        yuv_img = yuv_img.reshape((1080, 1280))
        rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV21)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb_img, (416, 416))
        result = self.graph.Inference(img)
        detection_list = self.yolov3.inference(result,0.7, 0.5, (416,416), (self.camera_width,self.camera_height))
        if len(detection_list) > 0:
            for detection in detection_list:
                cv2.rectangle(rgb_img, (detection.lt.x, detection.lt.y), (detection.rb.x, detection.rb.y), (0, 255, 0), 3)
                cv2.putText(rgb_img, detection.result_text, (detection.lt.x-5, detection.lt.y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0))
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', rgb_img)
        return jpeg.tobytes()


print("init camera object")
#video = VideoCamera()
            

app = Flask(__name__)


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen(camera):
    while True:
        time.sleep(0.01)
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = 5000)

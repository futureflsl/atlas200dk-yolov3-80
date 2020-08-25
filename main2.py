import hiai
import time
from hiai.nn_tensor_lib import DataType
from atlasutil import camera, ai, presenteragent, dvpp_process
import cv2 as cv

def load_labels(filename):
    labels=[]
    with open(filename,"r") as f:
        labels = f.read().splitlines()
    return labels

def main():
    camera_width = 1280
    camera_height = 720
    presenter_config = './face_detection.conf'
    cap = camera.Camera(id = 0, fps = 5, width = camera_width, height = camera_height , format = camera.CAMERA_IMAGE_FORMAT_YUV420_SP)
    if not cap.IsOpened():
        print("Open camera 0 failed")
        return
    chan = presenteragent.OpenChannel(presenter_config)
    if chan == None:
        print("Open presenter channel failed")
        return
    dvpp_handle = dvpp_process.DvppProcess(camera_width, camera_height)
    graph = ai.Graph('./model/darknet.om')
    yolov3 = ai.Yolov3Manager()
    yolov3.labels = load_labels("coco.names")
    while True:
        start = time.time()
        yuv_img = cap.Read()
        readtime = time.time()-start
        print("read time:",readtime)
        start = time.time()
        orig_image = dvpp_handle.Yuv2Jpeg(yuv_img)
        converttime = time.time()-start
        print("yuv conver jpeg time:", converttime)
        start = time.time()
        yuv_img = yuv_img.reshape((1080, 1280))
        img = cv.cvtColor(yuv_img, cv.COLOR_YUV2RGB_I420)
        img = cv.resize(img, (416, 416))
        resizetime = time.time()-start
        print("resize time:", resizetime)
        start = time.time()
        resultList = graph.Inference(img)
        inferencetime = time.time()-start
        print("inference time:", inferencetime)
        start = time.time()
        detection_list = yolov3.inference(resultList,0.7, 0.5, (416,416), (camera_height,camera_width))
        #print(result)
        resulttime = time.time()-start
        print("result time:",resulttime)
        chan.SendDetectionData(camera_width, camera_height, orig_image.tobytes(), detection_list)
        time.sleep(0.005)
  
if __name__ == "__main__":
    main()

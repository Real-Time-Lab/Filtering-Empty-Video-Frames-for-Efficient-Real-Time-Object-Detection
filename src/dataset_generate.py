# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
yolov5:
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''
yolo_vehicles={2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import sys
sys.path.append('../models') # sys.path.insert(0, '../models')
import cv2
import os
import torch
import numpy as np
import torchvision.transforms as transforms  # process image
import time
import erd_cnn
import yolov5
import ssd
from my_utils import mask_rect
# import efficientDet  ## comment it out when apply yolov5 since it conflicts with "import yolov5"
import pandas as pd

###########  setting #########################################
device_s1 = 'cuda'  # when use cpu to execute
device_s2 = 'cuda'
model_name_s2 = 'yolov5'  # 'ssd' or 'yolov5' or efficientDet
####################################################


interval = 1
# root = 'C:/dataset/traffic' ## win10
root = os.path.join(os.environ['HOME'],'./Documents/datasets/traffic')
file_name='Traffic count monitoring with computer vision 4K UHD HD.mp4'
# file_name ='Relaxing_highway_traffic.mp4'
video = os.path.join(root, file_name)  # home
is_mask = False

model_s2_zoo = ['yolov5','ssd', 'efficientDet']
assert model_name_s2 in model_s2_zoo, f'Model name is not correct, shall be one of {model_s2_zoo}'
model_s1 = erd_cnn.load_model(device_s1)
if model_name_s2 == 'efficientDet': import efficientDet  ## do this way cause it conflicts with "import yolov5"
model_s2 = eval(model_name_s2+'.load_model(device_s2)')
video_capturer = cv2.VideoCapture(video)
fps_video = video_capturer.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
erd_image_size= (640,360)
frame_id = 0
torch.no_grad()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy
exe_time_gpu, exe_time_cpu,  duration, duration_whole = [], [], [], []
i = 0
# video_h, video_w, c = 720, 1280, 3
results_erd, latency_s1, latency_s2, latency_whole = [], [], [], []
vehicle_num_list= []
data = pd.DataFrame(columns = ['empty_road_detection_result', 'latency_stage1', 'latency_stage2', 'latency_whole'])
data = pd.DataFrame(columns = [ 'latency_od', 'latency_whole'])
df = pd.DataFrame(columns = ['nonempty_road', 'vehicle_num'])
while (video_capturer.isOpened()):
    is_opened, frame = video_capturer.read()  # (720, 1280, 3)， H,W,C (BGR)
    if is_mask:
        frame = frame*mask_rect((frame_height,frame_width), upper_left = (0.1,0),lower_right = (1,1))
    start = time.time()
    if device_s1 == 'cuda':
        starter.record()

    if is_opened == True:
        frame_id += 1

        # Vehicle detection by ERD
        res_s1 = erd_cnn.predict(model_s1, frame, device_s1, erd_image_size)

        if device_s1 =='cuda':
            ender.record()
            torch.cuda.synchronize()
            t1 = starter.elapsed_time(ender) / 1000
        else:
            t1 = time.time() - start

        t1 = 0
        # res_s1 = 1
        vehicle_num = 0
        if i % interval == 0:  # interval =2: detect every 2 frames
            ## timer for start of stage2
            if device_s2 == 'cuda':
                starter.record()
            else: start_s2 = time.time()
            # if res_s1 == 1:  # object on the road detected
            if True:
                if model_name_s2 == 'yolov5':
                    res_s2 = yolov5.predict(model_s2, frame)
                    pred_objects =(res_s2.pred[0])[:,-1].tolist()  ## result is list of object classes detected, ,. e.g.[2,7] 'res_s2.pred[0]' is tensor, [:,-1] means object classes
                    for obj in pred_objects:
                        if obj in yolo_vehicles: vehicle_num+=1
                        # print("vehicle number:", vehicle_num)
                    cv2.putText(frame, '%s' % 'Vehicle number: '+str(vehicle_num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                elif model_name_s2 == 'ssd':
                    bboxes, classes, confidences = ssd.predict(model_s2, frame, device_s2)
                else: ## efficientDet
                    ori_imgs, framed_imgs, framed_metas = efficientDet.preprocess_video(frame, max_size=512)
                    res_s2 = efficientDet.predict(model_s2, framed_imgs, framed_metas, device_s2)

                start_imgshow = time.time()
                if res_s1 == 1:
                    cv2.putText(frame, '%s' % 'Nonempty Road', (780, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, '%s' % 'Empty Road', (780, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (155, 255, 55), 2)

                if model_name_s2 == 'yolov5':
                    # frame = np.squeeze(res_s2.render())
                    frame = yolov5.display(res_s2)  ## or: frame = np.squeeze(res_s2.render())
                elif model_name_s2 == 'ssd':
                    if len(classes) > 0:
                        for i in range(len(classes)):
                            xl, yl, xr, yr = int((bboxes[i][0]) * frame_width), int((bboxes[i][1]) * frame_height), \
                                             int((bboxes[i][2]) * frame_width), int((bboxes[i][3]) * frame_height)
                            cv2.rectangle(frame, (xl, yl), (xr, yr), (255, 0, 0), 1)
                            cv2.putText(frame, str(classes[i]), (xl, yl), 1, 1, (0, 255, 0))
                    else:
                        pass
                else: ## efficientDet
                    frame = efficientDet.display(res_s2, ori_imgs)

            # else:
            #     t2 = 0
            #     cv2.putText(frame, '%s' % 'Empty Road', (780, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (155, 255, 55),2)
            cv2.imshow('Result', frame)

            ## timer for end of stage2
            if device_s2 == 'cuda':
                ender.record()
                torch.cuda.synchronize()  ###
                t2 = starter.elapsed_time(ender) / 1000
            else:
                t2 = time.time() - start_s2

            results_erd.append(res_s1)
            latency_s1.append(t1)
            latency_s2.append(t2)
            latency_whole.append(time.time() - start)
            vehicle_num_list.append(vehicle_num)

            key = cv2.waitKey(2) & 0xFF  # or, waitKey(int(1000/fps_video))
            #         video_writer.write(frame)  # save to video
            if key == 27:  # 27: ESC to quite ， q: ord('q')
                is_open = False  # end of video
                break
            if key == ord('q'):
                time.sleep(3600)
        i+=1

    else:
        break
# video_writer.release()
video_capturer.release()
cv2.destroyAllWindows()

# data['empty_road_detection_result'] = results_erd
# data['latency_stage1'] =  latency_s1
data['latency_od'] =  latency_s2
data['latency_whole'] =  latency_whole
df['nonempty_road']=results_erd
df['vehicle_num']= vehicle_num_list
file_name = 'od_only_'+ model_name_s2+'_'+device_s2
# data.to_csv(os.path.join('../result/raw_data', (file_name+'.csv')), index=False)
# df.to_csv(os.path.join('../result/road_status.csv'), index=False)
print(df)

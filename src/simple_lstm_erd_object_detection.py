def count_vehicle_yolo(model, frame, device):
    yolo_vehicles = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    vehicle_num = 0
    res = yolov5.predict(model, frame)
    pred_objects = (res.pred[0])[:,
                   -1].tolist()  ## result is list of object classes detected, ,. e.g.[2,7] 'res_s2.pred[0]' is tensor, [:,-1] means object classes
    for obj in pred_objects:
        if obj in yolo_vehicles: vehicle_num += 1
    return vehicle_num, res  ## return both vehicle count and result of yolo


def pred_road(model_lstm, X, device):
    '''
    input: X is np.array, 2d
    '''
    if type(X) != torch.tensor: X = torch.Tensor(X).to(device)
    X = torch.unsqueeze(X, dim=0)
    y_pred_prob = model_lstm(X.to(device)).cpu()[:, 0]
    threshold = 0.5
    y_pred = [1 if y_pred_prob[j] > threshold else 0 for j in range(y_pred_prob.shape[0])]
    return y_pred[0]  ## batch size = 1, return the value instead of list


import os
import time
import torch
import cv2
import pandas as pd
import numpy as np
import sys

sys.path.append('../models')
import erd_cnn
import yolov5
import ssd
from lstm import LSTM_model

root = os.path.join(os.environ['HOME'], './Documents/datasets/traffic')
file_name = 'Traffic count monitoring with computer vision 4K UHD HD.mp4'
# file_name ='Relaxing_highway_traffic.mp4'
video = os.path.join(root, file_name)  # home
video_capturer = cv2.VideoCapture(video)

device_erd = 'cuda'
device_lstm = 'cuda'
device_od = 'cuda'
model_name_od = 'yolov5'

########### load models ##########
##1. load erd
model_erd = erd_cnn.load_model(device_erd)

##2. load od (yolov5)
model_od_zoo = ['yolov5', 'ssd', 'efficientDet']
assert model_name_od in model_od_zoo, f'Model name is not correct, shall be one of {model_od_zoo}'
if model_name_od == 'efficientDet': import efficientDet  ## do this way cause it conflicts with "import yolov5"
model_od = eval(model_name_od + '.load_model(device_od)')

##3. load lstm
model_lstm = LSTM_model().to(device_lstm)
model_lstm_state = '../pt_files/lstm20230726.pt'
model_lstm.load_state_dict(torch.load(model_lstm_state, map_location=torch.device(device_lstm)))
model_lstm.eval().to(device_lstm)

## initialize, configuration ##
erd_on = False
lstm_on = False
od_on = True

frame_id = 0
X = pd.DataFrame(columns=['nonempty_road', 'vehicle_num'])
data = pd.DataFrame(columns=['nonempty_road', 'vehicle_num', 'duration', 'fps'])

vehicle_num_list, erd_list = [1] * 10, [1] * 10
X.nonempty_road = erd_list
X.vehicle_num = vehicle_num_list
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy
dur_list, fps_list, erd_all_list, vehicle_num_all_list = [], [], [], []

while (video_capturer.isOpened()):
    start = time.time()
    ########### pause ###########
    #     print('Frame id:', frame_id)
    frame_id += 1
    if frame_id >= 100: break

    key1 = cv2.waitKey(2)
    if key1 == ord('p'):
        while (True):
            key2 = cv2.waitKey(5)
            if key2 == ord('o'):
                break

    is_opened, frame = video_capturer.read()

    ##============================##
    if erd_on:
        erd = erd_cnn.predict(model_erd, frame, device_erd, image_size=(640, 360))  ## return value of 0 or 1
    else:
        erd = -1
    erd_list.append(erd)

    if od_on:
        vehicle_num, res_od = count_vehicle_yolo(model_od, frame, device_od)
        vehicle_num_list.append(vehicle_num)
        frame = yolov5.display(res_od)
    else:
        vehicle_num = -1

    if lstm_on:
        if len(erd_list) >= 10:
            X.iloc[0:10, 0] = erd_list[-10:]  ## the newest one
            X.iloc[0:10, 1] = vehicle_num_list[-10:]  ## the newest one
        road_pred = pred_road(model_lstm, np.array(X), device_lstm)
    else:
        road_pred = -1
    #     print('Vehicle num:', vehicle_num)
    ##=============================##

    ### display ###

    if erd == 1:
        color_road_now = (0, 0, 255)
        text_road_now = 'Now: Nonempty'
    elif erd == -1:
        color_road_now = (215, 215, 215)
        text_road_now = 'ERD off'
    else:
        color_road_now = (155, 255, 55)
        text_road_now = 'Now: Empty'

    if road_pred == 1:
        color_road_future = (0, 0, 255)
        text_road_future = 'Next: Nonempty'
    elif road_pred == -1:
        color_road_future = (215, 215, 215)
        text_road_future = 'LSTM off'
    else:
        color_road_future = (155, 255, 55)
        text_road_future = 'Next: Empty'

    if od_on:
        color_od = (0, 255, 255)
    else:
        color_od = (215, 215, 215)

    cv2.putText(frame, '%s' % text_road_now, (780, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, \
                color_road_now, 2)
    cv2.putText(frame, '%s' % text_road_future, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color_road_future, 2)
    cv2.putText(frame, '%s' % 'Vehicle number: ' + str(vehicle_num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color_od, 2)

    # cv2.imshow(file_name, frame)
    key = cv2.waitKey(2)
    end = time.time()
    dur = end - start
    fps = 1 / dur
    dur_list.append(dur)
    fps_list.append(fps)
    vehicle_num_all_list.append(vehicle_num)
    erd_all_list.append(erd)

    if key == 27:  ## ord('q')
        break

    else:
        continue

## record and save data
data.nonempty_road = erd_all_list
data.vehicle_num = vehicle_num_all_list
data.duration = dur_list
data.fps = fps_list
print(f'Duration: {np.mean(dur_list)}, fps: {np.mean(fps_list)}.')

od_state = 'odOn' if od_on else 'odOff'
erd_state = 'erdOn' if erd_on else 'erdOff'
lstm_state = 'lstmOn' if lstm_on else 'lstmOff'
fun_state = od_state + '+' + erd_state + '+' + lstm_state

device_od_state = 'odCuda' if device_od == 'cuda' else 'odCpu'
device_erd_state = 'erdCuda' if device_erd == 'cuda' else 'erdCpu'
device_lstm_state = 'lstmCuda' if device_lstm == 'cuda' else 'lstmCpu'
device_state = device_od_state + '+' + device_erd_state + '+' + device_lstm_state

data.to_csv('../result/fps_result_' + fun_state + '_' + device_state + '.csv')
print(data)

video_capturer.release()
cv2.destroyAllWindows()
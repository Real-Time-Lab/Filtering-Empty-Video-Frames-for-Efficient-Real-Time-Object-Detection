'''
Put the testing video 'Traffic count monitoring with computer vision 4K UHD HD.mp4'  
    - https://www.youtube.com/watch?v=2kYpqSMqrzg&t=17s
Under  os.environ['HOME']+'/Documents/datasets/traffic/sh/' or anywhere want, but revide the path accordingly
'''
def count_vehicle_yolo(model, frame, device):
    yolo_vehicles = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    vehicle_num = 0
    res = yolov5.predict(model, frame)
    pred_objects = (res.pred[0])[:,
                   -1].tolist()  ## result is list of object classes detected, ,. e.g.[2,7] 'res_s2.pred[0]' is tensor, [:,-1] means object classes
    for obj in pred_objects:
        if obj in yolo_vehicles: vehicle_num += 1

    return vehicle_num, res  ## return both vehicle count and result of yolo

def count_vehicle_ssd(model, frame, device):
    ssd_vehicles = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck', 1:'car2',0:'car3', 68:'car4', 40:'car5'}
    ssd_vehicles = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
    vehicle_num = 0
    bboxes, classes, confidences = ssd.predict(model, frame, device)
    for obj in classes:
        if obj in ssd_vehicles: vehicle_num += 1
    return vehicle_num, bboxes, classes  ## return both vehicle count and result of yolo


def count_vehicle_efficientDet(model, frame, device):
    vehicles = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    ori_imgs, framed_imgs, framed_metas = efficientDet.preprocess_video(frame, max_size=512)
    res_od = efficientDet.predict(model, framed_imgs, framed_metas, device)
    classes = res_od[0]['class_ids']  ##class_ids  #2: car, 9: traffic light, 0: person, 5: bus, 7: truck. 1: bycycle
    vehicle_num = 0
    for obj in classes:
        if obj in vehicles: vehicle_num += 1
    return vehicle_num, res_od, ori_imgs  ## return both vehicle count and result of yolo


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
from datetime import datetime
sys.path.append('../models')
import erd_cnn
import yolov5
import ssd
from lstm import LSTM_model

def main():
    root = os.path.join(os.environ['HOME'], './Documents/datasets/traffic/sh/')
    file_name = 'Traffic count monitoring with computer vision 4K UHD HD.mp4'

    video = os.path.join(root, file_name)  # home
    video_capturer = cv2.VideoCapture(video)

    ###### configuration #######
    device_erd = 'cuda'
    device_lstm = 'cuda'
    device_od = 'cuda'
    model_name_od = 'yolov5'  ##yolov5 , efficientDet, ssd
    erd_on = True
    lstm_on = True
    od_only_test = False
    ignore_num =0
    ###############################

    ### load models
    ##1. load erd
    model_erd = erd_cnn.load_model(device_erd)

    ##2. load od (yolov5)
    model_od_zoo = ['yolov5', 'ssd', 'efficientDet']
    assert model_name_od in model_od_zoo, f'Model name is not correct, shall be one of {model_od_zoo}'
    if model_name_od == 'efficientDet': import efficientDet  ## do this way cause it conflicts with "import yolov5"
    model_od = eval(model_name_od + '.load_model(device_od)') ## load model in yolov5.py or ssd.py or efficientNet.py

    ##3. load lstm
    model_lstm = LSTM_model().to(device_lstm)
    model_lstm_state = '../pt_files/lstm20230725.pt'  ## ave. 74 fps, trained with 'Traffic count monitoring with computer vision 4K UHD HD.mp4'
    # model_lstm_state = '../pt_files/lstm20231004acc98.pt'  ## ave 66.4fps
    model_lstm.load_state_dict(torch.load(model_lstm_state, map_location=torch.device(device_lstm)))
    model_lstm.eval().to(device_lstm)

    ### variable initialization
    X = pd.DataFrame(columns=['nonempty_road', 'vehicle_num'])
    data = pd.DataFrame(columns=['erd_status','nonempty_road', 'vehicle_num', 'latency_whole', 'fps', 'all_cuda_latency'])
    vehicle_num_list, erd_list, road_status_list = [0] * 10, [0] * 10, [0]*10
    frame_id = 0
    X.nonempty_road = erd_list
    X.vehicle_num = vehicle_num_list
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy
    dur_list, fps_list ,dur_cuda_list = [], [], []
    erd_acc, lstm_acc = 0b00, 0b11
    od_on = True  ## start from od turn off


    frame_width = int(video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### only for cuda
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy  only for cuda
    while (video_capturer.isOpened()):
        is_opened, frame = video_capturer.read()
        if is_opened:
            start = time.time()

            if frame_id == ignore_num : start0= time.time()  ##  count fps without the first 10 frames

            # if frame_id >= 1000: break  ## set the time to run

            if frame_id % 30 == 0: erd_on = True  ## turn on erd every 30 frames
            if (frame_id - 3) % 30 == 0: erd_on = False ## turn off erd after 3 frames
            if frame_id % 1000 == 0: erd_acc, lstm_acc = 0b00, 0b11  ## reset every 1000 frames

            # set pause to snapshot images in the video
            # key1 = cv2.waitKey(2)
            # if key1 == ord('p'):
            #     while (True):
            #         key2 = cv2.waitKey(5)
            #         if key2 == ord('o'):
            #             break
            ##########################################

            if od_only_test:  ###for test od exclusive running metrics
                lstm_on = False ###Roy, for testing YOLO runtime exclusively

            if lstm_on:
                starter.record() ###Roy for cuda

                if len(erd_list) >= 10:
                    X.iloc[0:10, 0] = erd_list[-10:]  ## the newest one
                    X.iloc[0:10, 1] = vehicle_num_list[-10:]  ## the newest one
                lstm_pred = pred_road(model_lstm, np.array(X), device_lstm)
                lstm_acc = (lstm_acc << 1) + lstm_pred

                ender.record() ###Roy for cuda
                torch.cuda.synchronize() ###Roy for cuda
                t1 = starter.elapsed_time(ender) / 1000  ###Roy for cuda

            else:
                lstm_pred = -1
                t1 = 0  ###Roy for cuda

            if (0x03 & lstm_acc) == 0b00:  ## concecutively 3 frames empty
                od_on = False  ## turn off od
            #         lstm_acc = 0b11 ## reset

            if od_only_test:  ###for test od exclusive running metrics
                erd_on = False

            if erd_on:
                starter.record()  ###Roy for cuda

                erd = erd_cnn.predict(model_erd, frame, device_erd, image_size=(640, 360))  ## return value of 0 or 1
                erd_acc = (erd_acc << 1) + erd

                ender.record() ###Roy for cuda
                torch.cuda.synchronize() ###Roy for cuda
                t2 = starter.elapsed_time(ender) / 1000  ###Roy for cuda
            else:
                erd = -1

                t2 = 0  ###Roy for cuda
            # erd_list.append(erd)
            #     print(bin(0x03 & erd_acc))
            if (0x03 & erd_acc) == 0b11:  # consecutvely 3
                od_on = True
            #         erd_acc = 0b00 ## reset

            if od_only_test:  ###for test od exclusive running metrics
                od_on = True  ###Roy for testing, od_on = False:  bypass od

            if od_on:
                starter.record()  ###Roy for cuda
                if model_name_od == 'yolov5':
                    vehicle_num, res_yolo = count_vehicle_yolo(model_od, frame, device_od)
                    # res_yolo = yolov5.predict(model_od, frame)  ###Roy, only for teting exclusively run yolo,, exclusive run yolov5 is 56fps
                elif model_name_od == 'ssd':
                    vehicle_num, bboxes, classes = count_vehicle_ssd(model_od, frame, device_od)
                else:  ## efficientDet
                    vehicle_num, res_efficientDet, ori_imgs =count_vehicle_efficientDet(model_od, frame, device_od)

                ender.record() ###Roy for cuda
                torch.cuda.synchronize() ###Roy for cuda
                t3= starter.elapsed_time(ender) / 1000  ###Roy for cuda
            else:
                vehicle_num = -1
                t3= 0 ###Roy for cuda


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

            if lstm_pred == 1:
                color_road_future = (0, 0, 255)
                text_road_future = 'Nonempty'
            elif lstm_pred == -1:
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
                        color_od, 2)  ###Roy

            if od_on:  ## display with object detector
                if model_name_od == 'yolov5':
                    # frame = np.squeeze(res_s2.render())
                    frame = yolov5.display(res_yolo)  ## or: frame = np.squeeze(res_s2.render())
                elif model_name_od == 'ssd':
                    if len(classes) > 0:
                        for i in range(len(classes)):
                            xl, yl, xr, yr = int((bboxes[i][0]) * frame_width), int((bboxes[i][1]) * frame_height), \
                                             int((bboxes[i][2]) * frame_width), int((bboxes[i][3]) * frame_height)
                            cv2.rectangle(frame, (xl, yl), (xr, yr), (255, 0, 0), 1)
                            cv2.putText(frame, str(classes[i]), (xl, yl), 1, 1, (0, 255, 0))
                            start += 0.015  ### to fit the previous exp.
                    else:
                        pass
                else:  ## efficientDet
                    frame = efficientDet.display(res_efficientDet, ori_imgs)

            cv2.imshow(file_name, frame)

            dur = time.time() - start
            fps = 1 / dur
            dur_list.append(dur)  ## also called latency_whole
            dur_cuda_list.append(t1+t2+t3)
            fps_list.append(fps)
            road_status = 1 if od_on else 0  # road_status = 1 if (vehicle_num >0 or erd==1)  else 0, 1 means nonempty
            if frame_id >= 10:  # because they are set 0 for the first 10 frames
                vehicle_num_list.append(vehicle_num) ###Roy
                erd_list.append(erd)
                road_status_list.append(road_status)

            frame_id += 1

            key = cv2.waitKey(1)
            if key == 27:  ## ord('q')
                is_opened = False
                break

        else:  ## loop of if is_open
            break

    ## record and save data
    data.nonempty_road = road_status_list
    data.erd_status = erd_list
    data.vehicle_num = vehicle_num_list
    data.latency_whole = dur_list
    data.fps = fps_list
    data.all_cuda_latency = dur_cuda_list
    # print(f'Duration/latency_whole: {np.mean(dur_list)}, fps: {np.mean(fps_list)}.')

    od_state = 'odOn' if od_on else 'odOff'
    erd_state = 'erdOn' if erd_on else 'erdOff'
    lstm_state = 'lstmOn' if lstm_on else 'lstmOff'
    func_state = od_state + '+' + erd_state + '+' + lstm_state

    device_od_state = 'odCuda' if device_od == 'cuda' else 'odCpu'
    device_erd_state = 'erdCuda' if device_erd == 'cuda' else 'erdCpu'
    device_lstm_state = 'lstmCuda' if device_lstm == 'cuda' else 'lstmCpu'
    device_state = device_od_state + '+' + device_erd_state + '+' + device_lstm_state

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")

    print('Average FPS (measured by mean of each frame run time, more accurate):',(frame_id-ignore_num)/sum(dur_list[ignore_num:]))  ## more accurate cause keywait is excluded
    print('Average runtime per frame:',np.mean(dur_list[ignore_num:]),'ms')
    print(f'FPS overall (counted by whole time/frame_num, keywait time is included) = {(frame_id-ignore_num)/(time.time()-start0)}')
    # print(data)
    data.to_csv('../result/raw_data/performance_optmized' + '_' +model_name_od+'_'+ device_state + '_'+date_time+'.csv')

    video_capturer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
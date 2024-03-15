import sys
# sys.path.append('../../../efficientDet')
sys.path.insert(0,'../../efficientDet')  ##use this instead above to solve utils import confilit with environment default
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
import os

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

# function for display
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        return imgs[i]

def predict(model, framed_imgs, framed_metas, device):
    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    threshold = 0.2
    iou_threshold = 0.2
    if device == 'cuda':
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    # x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    x = x.to(device).permute(0, 3, 1, 2)

    ## model predict
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    # result
    out = invert_affine(framed_metas, out)
    return out

def load_model(device):
    compound_coef = 0
    force_input_size = None  # set None to use default size
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    # use_cuda = True
    # use_float16 = False
    # cudnn.fastest = True
    # cudnn.benchmark = True
    # tf bilinear interpolation is different from any other's, just make do

    ## load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(f'../../efficientDet/weights/efficientdet-d{compound_coef}.pth'))
    model.requires_grad_(False)
    model.eval()

    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda() # model.to('cuda')
    else: model.to('cpu')
    # if use_float16:
    #     model = model.half()

    return model

def main():
    root = '/home/royliu/Documents/datasets/traffic'  # ubuntu
    video_src = os.path.join(root, 'Traffic count, monitoring with computer vision. 4K, UHD, HD..mp4')  # home

    device = 'cuda'

    # Video capture
    video_capturer = cv2.VideoCapture(video_src)

    while True:
        ret, frame = video_capturer.read()
        if not ret:
            break

        # frame preprocessing
        model  = load_model(device)
        # ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=args['input_size'])
        ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=512)
        result = predict(model, framed_imgs, framed_metas, device)
        img_show = display(result, ori_imgs)
        # show frame by frame
        cv2.imshow('Result', img_show)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capturer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

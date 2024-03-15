import torch
import os
import cv2
import torchvision.transforms as transforms  # process image
import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([transforms.ToTensor()])  # value change to 0 to 1

def predict(model, frame, device):# in list format, e.g.[frame]
    utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils")
    threshold = 0.2
    frame_ssd = cv2.resize(frame, (300, 300))
    x_ssd = transform(frame_ssd)  # normalize value to 0 ~ 1， h,w,c convert to c,h,w
    x_ssd = torch.unsqueeze(x_ssd, dim=0)  # conver (c,h,w) to (1, c, h,w)
    x_ssd = torch.as_tensor(x_ssd).to(device)
    with torch.no_grad():
        detections_batch = model(x_ssd)

    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, threshold) for results in results_per_input]
    classes_to_labels = utils.get_coco_object_dictionary()

    for image_idx in range(len(best_results_per_input)):
        bboxes, classes, confidences = best_results_per_input[image_idx]

    return bboxes, classes, confidences


def load_model(device):
    model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd")
    # utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils")
    model.to(device)
    model.eval()
    return model

def main():
    root = '/home/royliu/Documents/datasets/traffic'  # ubuntu
    video_src = os.path.join(root, 'Traffic count, monitoring with computer vision. 4K, UHD, HD..mp4')  # home
    device = 'cuda'
    video_capturer = cv2.VideoCapture(video_src)
    frame_width = int(video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while video_capturer.isOpened():
        is_opened, frame = video_capturer.read()
        if not is_opened:
            break

        model  = load_model(device)
        bboxes, classes, confidences = predict(model, frame, device)
        if len(classes) > 0:
            for i in range(len(classes)):
                xl = int((bboxes[i][0]) * frame_width)
                yl = int((bboxes[i][1]) * frame_height)
                xr = int((bboxes[i][2]) * frame_width)
                yr = int((bboxes[i][3]) * frame_height)
                cv2.rectangle(frame, (xl, yl), (xr, yr), (255, 155, 0), 1)
                cv2.putText(frame, str(classes[i]), (xl, yl), 1, 1, (0, 255, 0))
        else:
            pass

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_capturer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main()
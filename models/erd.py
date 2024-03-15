## network loading (ERD)
import torch
import torch.nn as nn
import os
import cv2
import torchvision.transforms as transforms
# from torchstat import stat
from torchsummary import summary
class NET(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_channels = 3
        self.output_dim = 2
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, self.output_dim),
        )
        self.features = self.get_layers()  # must put here, not in forward method

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x

    def get_layers(self):
        net_config = [8, 8, 'M', 16, 16, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M']
        batch_norm = True
        layers = []
        # in_channels= self.in_channels
        in_channels = 3
        for c in net_config:
            assert c == 'M' or isinstance(c, int)
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = c
        return nn.Sequential(*layers)

def predict(model, frame, device, image_size):
    x = frame2tensor(frame, device, image_size)
    y_pre = model.forward(x)
    res = y_pre.argmax()
    res = res.cpu().numpy() if device== 'cuda' else res.numpy()  ## convert value in tensor to scaler
    return res.tolist()


def load_model(device):
    device_s1= device
    assert device_s1 == 'cuda' or device_s1 =='cpu', f'Device set to be "cuda" or "cpu".'
    state_file = '../pt_files/erd_20221214.pt'  
    model = NET()  # send model and weight to GPU as well, device(type='cuda')
    model = model.to(device_s1)  # SEND model to GPU or cpu
    model.load_state_dict(torch.load(state_file))
    return model

def frame2tensor(frame, device, image_size):
    transform = transforms.Compose([transforms.ToTensor()])  # value change to 0 to 1
    frame1 = cv2.resize(frame, image_size)
    x = transform(frame1)  # normalize value to 0 ~ 1， h,w,c convert to c,h,w
    x = torch.unsqueeze(x, dim=0)  # conver (c,h,w) to (1, c, h,w)
    x = torch.as_tensor(x).to(device)
    return x

def print_model_struct():
    from torchsummary import summary as model_summary
    model = NET()
    model_summary(model, input_size=(3, 360, 640))  # (c,h,w)

def display(result, frame, frame_id, frame_width, frame_height):
    left_x_up = int(frame_width / frame_id)
    word_x = frame_width - left_x_up - 400
    word_y = 50
    if result == 1:
        cv2.putText(frame, '%s' % 'Nonempty Road', (word_x - 200, word_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    else:
        cv2.putText(frame, '%s' % 'Empty Road', (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (155, 255, 55),
                    2)
    cv2.imshow('Result', frame)
    key = cv2.waitKey(2) & 0xFF 
    if key == 27:  # 27: ESC to quite ， q: ord('q')
        return False
    else: return True


def main():
    root = os.path.join(os.environ['HOME'],'./Documents/datasets/traffic') 
    device_s1 ='cuda'
    image_size= (640,360)
    model = load_model(device_s1)
    file_name = 'Traffic count monitoring with computer vision 4K UHD HD.mp4'
    video = os.path.join(root, file_name)  # home
    video_capturer = cv2.VideoCapture(video)
    frame_width = int(video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_id = 0

    while (video_capturer.isOpened()):
        is_opened, frame = video_capturer.read()  # (720, 1280, 3)， H,W,C (BGR)
        if is_opened:
            frame_id += 1
            result = predict(model, frame, device_s1, image_size) # 0 or 1
            is_opened = display(result, frame, frame_id, frame_width, frame_height)
        if not is_opened: break
        # else: break
    video_capturer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



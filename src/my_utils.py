import time
import os
import csv
import re
import numpy as np
import random

data_dir = root = os.path.join(os.environ['HOME'],'./Documents/datasets/traffic_od')

def mask_rect(sizes=(720,1280),upper_left = (0.4,0),lower_right = (0.9,0.5)):
    sizes= sizes  ## w*h
    upper_left = upper_left  # in percentage (h*w)
    lower_right = lower_right  ## in percentage
    upper_left = np.array(upper_left)*sizes  ## in pixel, (h*w), e.g. (288, 0)
    lower_right = np.array(lower_right)*sizes  ## e.g. (648,1280)
    mask = np.zeros(sizes,np.uint8)
    for y in range(int(upper_left[0]),int(lower_right[0])):
        for x in range(int(upper_left[1]),int(lower_right[1])):
            mask[y,x]=1.0
    mask = np.expand_dims(mask, axis=2)
    mask= np.repeat(mask,3, axis=2) ## (h,w,channel)
    return mask
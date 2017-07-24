import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import do_detect, plot_boxes, load_class_names
from darknet import Darknet
import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import struct # get_image_size
import imghdr # get_image_size
def yolo_detect(imgfile, cfgfile = "cfg/yolo.cfg", weightfile = "yolo.weights"):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    #print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
    class_names = load_class_names(namesfile)
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
	if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            if class_names[cls_id] == "person" and cls_conf >= 0.3:
		return True
    return False       
if __name__ == '__main__':
    if len(sys.argv) == 2:
        imgfile = sys.argv[1]
        print yolo_detect(imgfile)

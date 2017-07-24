# coding: utf-8

# In[3]:

#Clark modified code
import os
import numpy as np
import cv2
from glob import glob
import json
from os import path
from PIL import Image, ImageDraw, ImageFont
import urllib
import time
#from joblib import Parallel, delayed
import subprocess
import multiprocessing
import datetime
import os.path
from utils import do_detect, plot_boxes, load_class_names
from darknet import Darknet

# In[8]:

def motion_det_save_img(camurl, m, use_cuda, class_names):
    count = 0
    req_url = "rtsp://" + camurl + '/axis-media/media.amp'
    
    folder_name = camurl.replace(".", "_")
    cur_date = time.strftime("%m_%d_%y")
    cur_hour = time.strftime("%H")
    if not os.path.exists("./" + folder_name + "/" + cur_date + "/" + cur_hour):
        os.makedirs("./" + folder_name + "/" + cur_date + "/" + cur_hour)
    change_folder = False
    cap = cv2.VideoCapture(req_url)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    failed_time = 0
    while(True):
        ret, frame = cap.read()
	#print "start check frame"
        if (time.strftime("%H") != cur_hour):
            cur_hour = time.strftime("%H")
            change_folder = True
        if (time.strftime("%m_%d_%y") != cur_date):
            cur_date = time.strftime("%m_%d_%y")
            change_folder = True



        if count == 1000:
            print(str(camurl) + " is alive.")
            count = 0
        if ret:
            count += 1
            if change_folder:
                if not os.path.exists("./" + folder_name + "/" + cur_date + "/" + cur_hour):
                    os.makedirs("./" + folder_name + "/" + cur_date + "/" + cur_hour)
            failed_time = 0
            non_zero = 4
'''
            start_time = time.time()
            fgmask = fgbg.apply(frame)
            fgmask  = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            #     Check the percentage of non-zero (zero represents black pixel)
            total = fgmask.size
            temp_array = np.array(fgmask)
            cur_time = datetime.datetime.now().strftime("%H_%M_%S_%f")[:-3]


            non_zero = (1.0 * (temp_array != 0).sum() / total)
'''
            if non_zero >= 0.03:
                cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
		#global m
		#global use_cuda
		#global class_names
                sized = pil_im.resize((m.width, m.height))
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                for box in boxes:
                    thresh = 0.5
                    if box[5]>thresh and class_names[box[6]]=='person':
                        print("human detected")
                        #print 'human detected '+camurl
                        cv2.imwrite("./"+ folder_name+ "/" + cur_date + "/" + cur_hour  + "/" + cur_time + ".jpg", frame)
                #print 'save image from '  + camurl
            elapsed_time = time.time() - start_time
        #             print elapsed_time
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            failed_time += 1
            if failed_time >= 5000:                
                cmd = ['ffmpeg','-nostats','-loglevel', '0','-y','-i', req_url,'-vframes','1','snapshot/'+camurl +'.jpg']
                file_path = './snapshot/'+camurl+'.jpg'
                curProcess = subprocess.Popen(cmd)
                time.sleep(10)
                if os.path.isfile(file_path):
                    failed_time = 0
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                else:
                    print(str(camurl) + "Error, wait 0.5 hour")
                    time.sleep(1800)
                    motion_det_save_img(camurl)


def modified_urls(directory):
    orig_urls = glob(path.join(directory, '*.jpg'))
    prefix = len(directory)
    suffix = len('.jpg')
    urls = [orig[prefix:-suffix] for orig in orig_urls]
    return urls




# ## Method 2: Read from txt


# fname = "./song_list"

# with open(fname) as f:
#     content = f.readlines()

# urls = [x.strip() for x in content] 

# for url in urls:

#     rtspUrl = 'rtsp://' + url + '/axis-media/media.amp'
#     cmd = ['ffmpeg','-y','-i', rtspUrl,'-vframes','1','snapshot/'+url +'.jpg']
#     curProcess = subprocess.Popen(cmd)
#     time.sleep(1)

# ## Method 1: Read from picture

import random
num_cores = multiprocessing.cpu_count() * 2
base_path = "./base/"
base = modified_urls(base_path)
# print(base)

if not os.path.exists("./snapshot/"):
    os.makedirs("./snapshot/")

old_path = "./snapshot/"
to_del = modified_urls(old_path)
for url in to_del:
    file_path = 'snapshot/'+url +'.jpg'
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

def initialize(url):
    rtspUrl = 'rtsp://' + url + '/axis-media/media.amp'
    cmd = ['ffmpeg','-nostats','-loglevel', '0','-y','-i', rtspUrl,'-vframes','1','snapshot/'+url +'.jpg']
    curProcess = subprocess.Popen(cmd)
    time.sleep(1)
    
#Parallel(n_jobs=num_cores)(delayed(initialize)(url) for url in base)
for url in base:
    initialize(url) 
time.sleep(30)

url_path = "./snapshot/"
urls = modified_urls(url_path)
random.shuffle(urls)

print(urls)
#yolo model
#thresh = 0.8
cfgfile = 'cfg/yolo.cfg'
weightfile = 'yolo.weights'
m = Darknet(cfgfile)
#m.print_network()
m.load_weights(weightfile)

    
#print 'get frame'
if m.num_classes == 20:
    namesfile = 'data/voc.names'
elif m.num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/names'
class_names = load_class_names(namesfile)

use_cuda = 1
if use_cuda:
    m.cuda()
print ('get frame')
#
# num_cores = len(urls)
#Parallel(n_jobs=num_cores)(delayed(motion_det_save_img)(i, m, use_cuda, class_names) for i in urls)
for i in urls:
 motion_det_save_img(i,m, use_cuda, class_names)
'''
import threading
threads = []
for i in urls:
    t = threading.Thread(target = motion_det_save_img, args = (i, m, use_cuda, class_names,))
    threads.append(t)
for thr in threads:
    thr.start()
'''


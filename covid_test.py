import argparse
import logging
import time
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import pandas as pd
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from datetime import datetime

import re
import operator

FRAME_RATE = 25

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')

    
    f1=[]
    pipefar = rs.pipeline()
    farcfg=rs.config()
    farcfg.enable_device_from_file('i3d_keras/realsensedata/covid.bag')
    colz = rs.colorizer()
    pipefar.start(farcfg)

    while True:
        
        frame1 = pipefar.wait_for_frames()
        align= rs.align(rs.stream.color)
        frame1= align.process(frame1)
        
        cf1 = frame1.get_color_frame()
        df1 = frame1.get_depth_frame()
        depth_intrin = df1.profile.as_video_stream_profile().intrinsics
        farcolor = np.asanyarray(cf1.get_data())
        fardepth = np.asanyarray(colz.colorize(df1).get_data())
        #d8= (fardepth/256).astype(np.uint8)
        #d8= cv2.applyColorMap(d8,cv2.COLORMAP_JET)
        #plt.rcParams["axes.grid"] = False
        #plt.rcParams['figure.figsize'] = [12, 6]
        #plt.imshow(fardepth)
        
        logger.info('cam image=%dx%d' % (farcolor.shape[1], farcolor.shape[0]))
        logger.debug('image process+')
        humans = e.inference(farcolor, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        farcolor = TfPoseEstimator.draw_humans(farcolor, humans, imgcopy=False)
        allx=[]
        for human in humans:
            print("This is the value of a human for each loop:", human)
            #for i in range(len(humans)):
            try:
                a =str(human.body_parts) 
                b= re.findall(r'(\(.*?,.*?\))', a)
                c= [tuple(float(s) for s in i.strip("()").split(",")) for i in b]
                xcoor=[i[0]*640 for i in c]
                allx.append(xcoor)
                #print(' original a is:', a)
                print('list of coordinates is:',xcoor)

            except:
                pass
        if len(allx)>1:
            hu1,hu2=allx[0],allx[1]
            dist=[abs(i-j) for i,j in zip(hu1,hu2)]
            print("Distance between the 2 subjects is: ", dist)
            for i in dist:
                if i<160:
                    print(i)
                    danger="Please maintain distance!!"
                    cv2.putText(farcolor,danger,color=(255, 0, 0),org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
        logger.debug('postprocess+')
        farcolor = TfPoseEstimator.draw_humans(farcolor, humans, imgcopy=False)

        #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        farcolor= cv2.cvtColor(farcolor, cv2.COLOR_BGR2RGB)
        im2= cv2.imread("3.jpg")
        #vis = np.concatenate((farcolor, im2), axis=1)
        vis= farcolor
        logger.debug('show+')
        
        cv2.putText(vis,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow('tf-pose-estimation result', farcolor)
        
        #cv2.imshow('depth result', fardepth)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    pipefar.stop()
    #pipenear.stop()
    cv2.destroyAllWindows()

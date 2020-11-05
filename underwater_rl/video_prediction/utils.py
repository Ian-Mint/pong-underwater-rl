import cv2
import numpy as np
import torch

'''
read mnist_test_seq.npy
mnist_test_seq = np.load('mnist_test_seq.npy').swapaxes(0,1)    # shape:(10000, 20, 64, 64)
height = mnist_test_seq.shape[2]
width = mnist_test_seq.shape[3]
'''

def create_array(y_hat, y):
    y_hat = y_hat.cpu().numpy()
    y = y.cpu().numpy()
    video_frames = np.concatenate([y,y_hat],axis=3)
    return video_frames


def generate_video(video_array, video_filename, fps=5, width=168, height=84):
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    # new frame after each addition of water
    for batch in range(10):
        for frame in range(4):
        #add this array to the video
            gray = cv2.normalize(video_array[batch,frame,:,:], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            gray_3c = cv2.merge([gray, gray, gray])
            out.write(gray_3c)

    # close out the video writer
    out.release()
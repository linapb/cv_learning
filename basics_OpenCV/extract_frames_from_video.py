import cv2
import os

filename = '03102018-1700.mp4'
skip_frames = 10
dir_name = '{}_frames_skip{}'.format(filename[:-4], skip_frames)
os.mkdir(dir_name)

vidcap = cv2.VideoCapture(filename)
success, image = vidcap.read()
count = 0
extracted = 0

while success:
    if count % skip_frames == 0:
        cv2.imwrite("{}/frame{}.jpg".format(dir_name, count), image)     # save frame as JPEG file
        extracted += 1
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

print('Number of frames extracted: ', extracted)
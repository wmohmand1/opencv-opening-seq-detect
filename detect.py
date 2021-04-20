import cv2 as cv
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import logging
import threading
import time


start = []
end = []

def detect(video, template_start, template_end, method=cv.TM_CCOEFF_NORMED, sample_rate=4, threshold=0.9):
    # initialize
    global start
    global end
    threads_list = list()
    

    firstT = threading.Thread(target=threadDetect, args=(0, 1000, template_start, template_end, method, threshold))
    secondT = threading.Thread(target=threadDetect, args=(1000, 2000, template_start, template_end, method, threshold))
    thirdT = threading.Thread(target=threadDetect, args=(2000, 3000, template_start, template_end, method, threshold))
    fourthT = threading.Thread(target=threadDetect, args=(3000, 4000, template_start, template_end, method, threshold))
    fifthT = threading.Thread(target=threadDetect, args=(4000, 4320, template_start, template_end, method, threshold))

    firstT.start()
    secondT.start()
    thirdT.start()
    fourthT.start()
    fifthT.start()

    threads_list.append(firstT)
    threads_list.append(secondT)
    threads_list.append(thirdT)
    threads_list.append(fourthT)
    threads_list.append(fifthT)

    for t in threads_list:
        t.join()
    
    print('Selecting best match...')
    start = [ x for x in start if x[1] == max(np.transpose(start)[1])]
    end = [ x for x in end if x[1] == max(np.transpose(end)[1])]

    return start, end

def threadDetect (count, endCount, template_start, template_end, method, threshold):
    # start reading the video
    print('Detection started. This could take a while.')
    vidcap = cv.VideoCapture(video)
    while vidcap.isOpened():
        vidcap.set(cv.CAP_PROP_POS_FRAMES,count)
        success, img = vidcap.read()
        if count> endCount: # only checking the first 3min of the video
            break
        if count%200==0 and count!=0:
            print(f'{count} frames checked.')
        if count%1 == 0:
            if success:
                # replace with normalized image buffer
                cv.imwrite(os.path.join('out', '%d.png') % count, img)
                image = cv.imread(os.path.join('out', '%d.png') % count, 0)
                # detection
                detect_start = cv.matchTemplate(template_start, image,method)
                conf_start = cv.minMaxLoc(detect_start)[1]
                detect_end = cv.matchTemplate(template_end, image,method)
                conf_end = cv.minMaxLoc(detect_end)[1]
                # grab results
                if conf_start > threshold and conf_start < 1:
                    print(f'start candidate found at {count}')
                    start.append([count, conf_start])
                elif conf_end > threshold and conf_end < 1:
                    print(f'end candidate found at {count}')
                    end.append([count, conf_end])
                else:
                    os.remove(os.path.join('out', '%d.png') % count)
                count += 1
            else:
                raise ValueError('Cannot read video at frames '+str(count))
                break
        else:
            count += 1

    cv.destroyAllWindows()
    vidcap.release()


def result_gen(detect_start, detect_end, start_img, end_img):
    print('Creating report...')
    plt.rcParams['figure.figsize'] = [16, 9]
    height, width= start_img.shape
    blank = np.add(np.zeros((height,width,3), np.uint8), 255)

    plt.subplot(221),plt.imshow( start_img ,cmap = 'gray')
    plt.title('Start Template')
    if detect_start:
        plt.subplot(222),plt.imshow( cv.imread(os.path.join('out', '%d.png') % detect_start[0][0], 0), cmap = 'gray')
        plt.title('Detection {}, Confidence={}'.format( str(detect_start[0][0]) ,str(detect_start[0][1])[:5]))
    else:
        errorMessage = "Starting position not detected for the specified image template, method and confidence threshold."
        print(errorMessage)
        detect_error = cv.putText(img=np.copy(blank), text="Not detected.", org=(200,400),fontFace=2, fontScale=4, color=(255,0,0), thickness=4)
        plt.subplot(222),plt.imshow( detect_error ,cmap = 'gray')
        plt.title('Detection')
    plt.subplot(223),plt.imshow( end_img ,cmap = 'gray')
    plt.title('End Template')
    if detect_end:
        plt.subplot(224),plt.imshow( cv.imread(os.path.join('out', '%d.png') % detect_end[0][0], 0), cmap = 'gray')
        plt.title('Detection {}, Confidence={}'.format( str(detect_end[0][0]) ,str(detect_end[0][1])[:5]))
    else:
        errorMessage = "Ending position not detected for the specified image template, method and confidence threshold."
        print(errorMessage)
        detect_error = cv.putText(img=np.copy(blank), text="Not detected.", org=(200,400),fontFace=2, fontScale=4, color=(255,0,0), thickness=4)
        plt.subplot(224),plt.imshow( detect_error ,cmap = 'gray')
        plt.title('Detection')
    plt.suptitle('Opening Sequence Detection\n{}'.format(video))
    plt.savefig('result.png')
    print('Report created result.png')
    return

if len(sys.argv) !=4:
    print("\r\nInvalid or missing params. Please format your command as the following: \r\npython detect.py <video_path> <start_image> <end_image>\r\n e.g. python3 detect.py 'friends_s1e2.avi' 'templates/start.png' 'templates/end.png'\r\n")
else:
    script, video, startTemplate, endTemplate = sys.argv
    print(f"\r\n---\r\nvideo input is: {video}\r\nstart image selected: {startTemplate}\r\nend image selected: {endTemplate}\r\n---\r\n")

    start_img = cv.imread(str(startTemplate), 0)
    end_img = cv.imread(str(endTemplate), 0)

    detect_start, detect_end = detect(str(video), start_img, end_img)
    result_gen(detect_start, detect_end, start_img, end_img)
    sys.stdout.write(str(detect_start))
    sys.stdout.write(str(detect_end))

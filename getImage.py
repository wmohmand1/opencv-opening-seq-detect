import cv2 as cv
import sys
import os

def getImage(video, frameCount, name):
    vidcap = cv.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        if count > 4320:
            break # sanity check
        success, img = vidcap.read()
        if count == int(frameCount):
            print('yes')
            cv.imwrite(os.path.join('templates', '{}.png'.format(name)), img)
            cv.destroyAllWindows()
            vidcap.release()
            break
        count += 1
    cv.destroyAllWindows()
    vidcap.release()
    return

if len(sys.argv) != 4:
    print('Invalid or missing params. Please format your command as the following: \r\npython getImage.py <video_path> <frameCount> <outputName>')
else:
    script, video, frameCount, output = sys.argv
    print(f"\r\n---\r\nvideo input is: {video}\r\nframe count from start is:{frameCount}\r\noutput image name is:{output}\r\n---\r\n")
    # setup toolbar
    getImage(video, frameCount, str(output))
    if '{}.png'.format(str(output)) in os.listdir("templates"):
        sys.stdout.write('Image template_{}.png successfully generated.\n'.format(str(output)))
    else:
        sys.stdout.write('Image template_{}.png did not generate correctly. Please retry.\n'.format(str(output)))

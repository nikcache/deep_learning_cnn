import cv2
import numpy as np

def vidOut(vidpath, outfile):
    cap = cv2.VideoCapture(vidpath)
    output_file = outfile
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    is_begin = True

    while(1):
        ret, frame2 = cap.read()

        if frame2 is None:
            break
        processed = frame2

        if is_begin:
            h, w, _ = processed.shape
            out = cv2.VideoWriter(output_file, fourcc, 30, (w, h), True)
            is_begin = False

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, 10, 1, 7, 1.5, 0)
        
        # OPTIMAL 
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, 15, 3, 5, 1.2, 0)
        
        # OLD
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,90,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        out.write(rgb)
        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()

def main():

    vidOut('cnn_set/guitar/vid_2/IMG_9020.mp4', 'cnn_set/guitar/vid_2/IMG_9020_test.mp4')

main()

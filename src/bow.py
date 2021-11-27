import cv2
import paddlehub as hub
import paddlex as pdx
import numpy as np
import os
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class detUtils():
    def __init__(self):
        super(detUtils, self).__init__()
        self.module = hub.Module(name="yolov3_resnet50_vd_coco2017")
        self.last = None

    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def iou(self, bbox1, bbox2):

        b1left = bbox1['left']
        b1right = bbox1['right']
        b1top = bbox1['top']
        b1bottom = bbox1['bottom']

        b2left = bbox2['left']
        b2right = bbox2['right']
        b2top = bbox2['top']
        b2bottom = bbox2['bottom']

        area1 = (b1bottom - b1top) * (b1right - b1left)
        area2 = (b2bottom - b2top) * (b2right - b2left)

        w = min(b1right, b2right) - max(b1left, b2left)
        h = min(b1bottom, b2bottom) - max(b1top, b2top)

        dis = self.distance([(b1left+b1right)/2, (b1bottom+b1top)/2],[(b2left+b2right)/2, (b2bottom+b2top)/2])

        if w <= 0 or h <= 0:
            return 0, dis
        
        iou = w * h / (area1 + area2 - w * h)
        return iou, dis

    def do_det(self, frame):
        res = self.module.object_detection(images=[frame], use_gpu=True, visualization=False)
        result = res[0]['data']
        reslist = []
        for r in result:
            if r['label'] == 'person':
                reslist.append(r)
        if len(reslist) == 0:
            if self.last is not None:
                return self.last
            else:
                return None
        elif len(reslist) == 1:
            self.last = reslist[0]
            return reslist[0]
        else:
            if self.last is not None:
                maxiou = -float('inf')
                maxi = 0
                mind = float('inf')
                mini = 0
                for index in range(len(reslist)):
                    tiou, td = self.iou(self.last, reslist[index])
                    if tiou > maxiou:
                        maxi = index
                        maxiou = tiou
                    if td < mind:
                        mind = td
                        mini = index  
                if tiou == 0:
                    self.last = reslist[mini]
                    return reslist[mini]
                else:
                    self.last = reslist[maxi]
                    return reslist[maxi]
            else:
                return reslist[0]

class estUtils():
    def __init__(self):
        super(estUtils, self).__init__()
        self.module = hub.Module(name='human_pose_estimation_resnet50_mpii')

    def do_est(self, frame):
        # pair(x, y)
        res = self.module.keypoint_detection(images=[frame], use_gpu=True)
        return res[0]['data']

class FOClassifer():
    def __init__(self):
        super().__init__()
        self.model = pdx.deploy.Predictor('inference_model', use_gpu=True)

    def classifer(self, frame):
        result = self.model.predict(frame.astype('float32'))[0]['category_id']
        if result == 0:
            return True
        else:
            return False


class bodyController():
    def __init__(self, ratio, debug):
        super().__init__()
        self.clap = False
        self.release = 0
        self.DU = detUtils()
        self.EU = estUtils()
        self.FOC = FOClassifer()
        self.ratio = ratio
        self.cap = cv2.VideoCapture(0)
        self.debug = debug


    def control(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None
        
        frame = cv2.flip(frame, 1)
        if self.debug:
            cv2.imshow("action", frame)
            cv2.waitKey(1)

        if self.release > 7:
            self.release = 0
            self.clap = False

        det_res = self.DU.do_det(frame)
        if det_res is None:
            return None, None, None, None
            # return clap, distance, angle, release
        top = int(det_res['top'])
        left = int(det_res['left'])
        right = int(det_res['right'])
        bottom = int(det_res['bottom'])
        if self.debug:
            cv2.rectangle(frame, (left, top),(right, bottom), (0,255,0), 2)
            cv2.imshow("action", frame)
            cv2.waitKey(1)
        humanpart = frame[top:bottom, left:right]
        h, w = frame.shape[:2]
        pose_res = self.EU.do_est(humanpart)
        lwp = pose_res['left_wrist']
        rwp = pose_res['right_wrist']
        if lwp[0] > rwp[0]:
            tmp = rwp
            rwp = lwp
            lwp = rwp
        th = pose_res['thorax']
        ht = pose_res['head_top']

        if self.debug:
            for key, value in pose_res.items():
                cv2.circle(humanpart, (int(value[0]), int(value[1])), 8, (0,0,255), -1)
            cv2.imshow("actionhuman", humanpart)
            cv2.waitKey(1)

        unit = self.DU.distance(th, ht)
        lrwp = self.DU.distance(lwp, rwp)

        if not self.clap:
            if lrwp < unit:
                self.clap = True
            return self.clap, None, None, None
        else:
            angle = math.atan((lwp[1] - rwp[1]) / (lwp[0] - rwp[0] + 0.000000001))

            radius = unit * 1.5

            cx = lwp[0] + left
            cy = lwp[1] + top

            htop = int(cy - radius) if cy - radius > 0 else 0
            hbottom = int(cy + radius) if cy + radius < h else h-1
            hleft = int(cx - radius) if cx - radius > 0 else 0
            hright = int(cx + radius) if cx + radius < w else w - 1

            lefthand = frame[htop:hbottom, hleft:hright]
            if self.debug:
                cv2.imshow("lefthand", lefthand)
                cv2.waitKey(1)
            self.getRelease(lefthand)

            return True, lrwp / unit / self.ratio, angle, self.release > 5


    def getRelease(self, frame):
        if self.FOC.classifer(frame):
            self.release += 1
        else:
            self.release = 0


# bCer = bodyController()        
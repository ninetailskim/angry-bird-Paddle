import paddlehub as hub
import cv2
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class segUtil():
    def __init__(self):
        self.model = hub.Module(name="deeplabv3p_xception65_humanseg")

    def doSeg(self, frame):
        res = self.model.segmentation(images=[frame], use_gpu=True)
        return res[0]['data']

SU = segUtil()

def compose():
    cap = cv2.VideoCapture("xuebati.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("res.mp4", fourcc, fps, (width, height))

    back = cv2.imread("back.png")
    back = cv2.resize(back, (width, height))
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("test", frame)
            cv2.waitKey(1)
            seg_mask = np.around(SU.doSeg(frame) / 255)
            seg_mask3 = np.repeat(seg_mask[:,:,np.newaxis], 3, axis=2)
            result = frame * seg_mask3 + back * (1 - seg_mask3)
            out.write(result.astype(np.uint8))
        else:
            break
    out.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    compose()

import os
import cv2
import time
import yaml
import paddle
import numpy as np
from .keypoint_visualize import draw_pose
from .keypoint_postprocess import HRNetPostProcess
from paddle.inference import Config, create_predictor
from .preprocess import preprocess, NormalizeImage, Permute
from .utils import argsparser, Timer, get_current_memory_mb
from .keypoint_preprocess import EvalAffine, TopDownEvalAffine

# delete
KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}

class KeyPoint_Detector(object):
    def __init__(self, 
                pred_config,
                model_dir,
                use_gpu=False,
                run_mode='fluid',
                trt_calib_mode=False,
                cpu_threads=1,
                enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn
        )
        print("what?")

    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        inputs = create_inputs(im, im_info)
        return inputs

    def postprocess(self, np_boxes, np_masks, inputs, threshold=0.5):
        results = {}
        imshape = inputs['im_shape'][:,::-1]
        center = np.round(imshape / 2.)
        scale = imshape / 200
        keypoint_postprocess = HRNetPostProcess()
        results['keypoint'] = keypoint_postprocess(np_boxes, center, scale)
        return results

    def predict(self, image, threshold=0.5, warmup=0, repeats=1):
        # Start preprocess
        inputs = self.preprocess(image)
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        # End preprocess

        # Start inference
        ## warm up
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        if self.pred_config.tagmap:
            masks_tensor = self.predictor.get_output_handle(output_names[1])
            heat_k = self.predictor.get_output_handle(output_names[2])
            inds_k = self.predictor.get_output_handle(output_names[3])
            np_masks = [
                masks_tensor.copy_to_cpu(), heat_k.copy_from_cpu(),
                inds_k.copy_to_cpu()
            ]
        # End inference

        # Start postprocess
        results = self.postprocess(
            np_boxes, np_masks, inputs, threshold=threshold
        )
        # End postprocess
        
        return results


def create_inputs(im, im_info):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.array((im, )).astype('float32')
    inputs['im_shape'] = np.array((im_info['im_shape'], )).astype('float32')

    return inputs


class PredictConfig_KeyPoint():
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        # print(yml_conf)
        self.arch = yml_conf['arch']
        self.archcls = KEYPOINT_SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.tagmap = False
        if 'keypoint_bottomup' == self.archcls:
            self.tagmap = True
        # self.print_config()

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')
        # input()


def load_predictor(model_dir,
                    run_mode='fluid',
                    batch_size=1,
                    use_gpu=False,
                    min_subgraph_size=False,
                    trt_calib_mode=False,
                    cpu_threads=1,
                    enable_mkldnn=False):
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams')
    )
    if use_gpu:
        config.enable_use_gpu(200, 0)
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
        config.set_cpu_match_library_num_threads(cpu_threads)
        if enbale_mkldnn:
            try:
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass
        
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


def predict_video(detector, threshold, camera_id, video_file):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'output.mp4'
    else:
        capture = cv2.VideoCapture(video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]+'.mp4'
    fps = 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not os.path.exists("output"):
        os.makedirs("output")
    out_path = os.path.join("output", video_name + '.mp4')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    allTime = 0
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        index += 1
        startTime = time.time()
        results = detector.predict(frame, threshold)
        endTime = time.time()
        print("Index: ", index , "Time: ", endTime - startTime)
        allTime = allTime + (endTime - startTime)
        # draw results
        im = draw_pose(
            frame, results, visual_thread=threshold, returnimg=True)
        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    print("Time:", allTime / index)
    writer.release()


class MyDetector():
    def __init__(self, model_dir, use_gpu):
        self.pred_config = PredictConfig_KeyPoint(model_dir)
        self.detector = KeyPoint_Detector(
            self.pred_config,
            model_dir,
            use_gpu=use_gpu,
            run_mode='fluid',
            # use_dynamic_shape=False,
            trt_calib_mode=False,
            cpu_threads=1,
            enable_mkldnn=False
        )

    def predict(self, frame, threshold=None):
        results = self.detector.predict(frame, threshold)
        return results


def main(model_dir, use_gpu, threshold, camera_id=None, video_file=None):
    md = MyDetector(model_dir, use_gpu)

    predict_video(md.detector, threshold, camera_id, video_file)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    # print(FLAGS)
    main(FLAGS.model_dir, FLAGS.use_gpu, FLAGS.threshold, FLAGS.camera_id, FLAGS.video_file)

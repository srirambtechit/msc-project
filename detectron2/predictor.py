import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

class VisualizationDemo(object):

    def __init__(self, args, m1_cfg, m2_cfg, instance_mode=ColorMode.IMAGE):
        if args.classifier == 'umpire-classifier':
            self.classifier = UmpireClassifier(m1_cfg, instance_mode)
            self.MODE = 'UC'
        elif args.classifier == 'umpire-pose-classifier':
            self.classifier = UmpireSignsClassifier(m2_cfg, instance_mode)
            self.MODE = 'UPC'
        else:
            self.umpire_classifier = UmpireClassifier(m1_cfg, instance_mode)
            self.umpire_signs_classifier = UmpireSignsClassifier(m2_cfg, instance_mode)
            self.MODE = 'B'

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
            
    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        # initializing VideoVisualizer based on the classifier selected at runtime
        if self.MODE == 'B':
            comb_vid_vsulzr = self.umpire_signs_classifier.video_visualizer()
        else:
            sngl_vid_vsulzr = self.classifier.video_visualizer()

        def process_single_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = sngl_vid_vsulzr.draw_instance_predictions(frame, predictions)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        def process_combine_predictions(frame, predictions):
            vis_frame = None
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if "instances" in predictions:
                instances = predictions["instances"]
                # only interested in frame classified as "umpire"
                if 2 in instances.pred_classes:
                    vis_frame = self.umpire_signs_classifier.predict_and_draw(frame, comb_vid_vsulzr)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            if self.MODE == 'B':
                yield process_combine_predictions(frame, self.umpire_classifier.predictor(frame))
            else:
                yield process_single_predictions(frame, self.classifier.predictor(frame))

class UmpireClassifier(object):
    def __init__(self, cfg, instance_mode):
        assert cfg is not None, "UmpireClassifier's config cannot be None"
        print('UC dataset', cfg.DATASETS.TEST[0])

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.metadata.thing_classes = ['isumpire', 'non-umpire', 'umpire']
        self.metadata.thing_dataset_id_to_contiguous_id = ({0: 0, 1: 1, 2: 2})
        self.predictor = DefaultPredictor(cfg)
        self.instance_mode = instance_mode

    def video_visualizer(self):
        return VideoVisualizer(self.metadata, self.instance_mode)

class UmpireSignsClassifier(object):
    def __init__(self, cfg, instance_mode):
        assert cfg is not None, "UmpireSignsClassifier's config cannot be None"
        print('UC dataset', cfg.DATASETS.TEST[0])
        
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.metadata.thing_classes = ['umpire-signals', 'no-action', 'no-ball', 'out', 'six', 'wide']
        self.metadata.thing_dataset_id_to_contiguous_id = ({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        self.predictor = DefaultPredictor(cfg)
        self.instance_mode = instance_mode
        self.cpu_device = torch.device("cpu")

    def video_visualizer(self):
        return VideoVisualizer(self.metadata, self.instance_mode)

    def predict_and_draw(self, frame, viz):
        predictions = self.predictor(frame)
        if "instances" in predictions:
            instances = predictions["instances"]
            # only interested in frame not classified as NO-ACTION
            # such as OUT, SIX, NO-BALL, WIDE
            if 1 not in instances.pred_classes:
                predictions = instances.to(self.cpu_device)
                vis_frame = viz.draw_instance_predictions(frame, predictions)

                # Converts Matplotlib RGB format to OpenCV BGR format
                vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
                return vis_frame

"""
sudo code:
two task
 use c1 
 detect frame is umpire
 if yes
  use c2
  detect frame is not no-action
  if yes
   store that frame
   save as video
"""

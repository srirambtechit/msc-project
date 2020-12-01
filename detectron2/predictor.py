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
    def __init__(self, args, cfg, instance_mode=ColorMode.IMAGE):
        if args.classifier == 'umpire-classifier':
          self.umpire_classifier = UmpireClassifier(cfg, instance_mode)
        elif args.classifier == 'umpire-pose-classifier':
          self.umpire_signs_classifier = UmpireSignsClassifier(cfg, instance_mode)

        self.cpu_device = torch.device("cpu")

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
        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "instances" in predictions:
              instances = predictions["instances"]
              pred_classes = instances[ instances.pred_classes == 2]
              print('Pred_classes',pred_classes)
              predictions = instances.to(self.cpu_device)
              vis_frame = self.umpire_classifier.video_visualizer().draw_instance_predictions(frame, predictions)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.umpire_classifier.predictor(frame))
            
        # video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        # def process_predictions(frame, predictions):
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     if "instances" in predictions:
        #         predictions = predictions["instances"].to(self.cpu_device)
        #         vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

        #     # Converts Matplotlib RGB format to OpenCV BGR format
        #     vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        #     return vis_frame

        # frame_gen = self._frame_from_video(video)
        # for frame in frame_gen:
        #     yield process_predictions(frame, self.predictor(frame))

class UmpireClassifier(object):
  def __init__(self, cfg, instance_mode):
    self.metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    self.metadata.thing_classes = ['isumpire', 'non-umpire', 'umpire' ]
    self.metadata.thing_dataset_id_to_contiguous_id=({0: 0, 1: 1, 2: 2})
    self.predictor = DefaultPredictor(cfg)
    self.instance_mode = instance_mode

  def video_visualizer(self):
    return VideoVisualizer(self.metadata, self.instance_mode)

class UmpireSignsClassifier(object):
  def __init__(self, cfg, instance_mode):
    self.metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    self.metadata.thing_classes = ['umpire-signals', 'no-action', 'no-ball', 'out', 'six', 'wide']
    self.metadata.thing_dataset_id_to_contiguous_id=({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
    self.predictor = DefaultPredictor(cfg)
    self.instance_mode = instance_mode

  def video_visualizer(self):
    return VideoVisualizer(self.metadata, self.instance_mode)

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
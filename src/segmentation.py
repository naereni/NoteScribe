from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch.cuda import is_available

import helper_functions as help_fn


class SEGMpredictor:
    def __init__(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = model_path
        cfg.TEST.EVAL_PERIOD = 1000

        cfg.INPUT.MIN_SIZE_TRAIN = 2160
        cfg.INPUT.MAX_SIZE_TRAIN = 3130

        cfg.INPUT.MIN_SIZE_TEST = 2160
        cfg.INPUT.MAX_SIZE_TEST = 3130
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        cfg.INPUT.FORMAT = "BGR"
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.IMS_PER_BATCH = 3
        cfg.SOLVER.BASE_LR = 0.01
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.STEPS = (1500,)

        if not is_available():
            cfg.MODEL.DEVICE = "cpu"

        cfg.SOLVER.MAX_ITER = 17000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.OUTPUT_DIR = "./output"

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        outputs = self.predictor(img)
        prediction = outputs["instances"].pred_masks.cpu().numpy()
        contours = []
        for pred in prediction:
            contour_list = help_fn.get_contours_from_mask(pred)
            contours.append(help_fn.get_larger_contour(contour_list))
        return contours

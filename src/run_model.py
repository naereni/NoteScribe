from __future__ import annotations

import json
import logging
import warnings

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage as plt_image

import model.helper_functions as help_fn
import model.recognition as rec
import model.segmentation as segm

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
logger = logging.getLogger("detectron2")
logger.setLevel(logging.CRITICAL)


with open("model/config.json") as f:
    CONFIG_JSON = json.load(f)

CONFIG_JSON["ocr_config"]["device"] = (
    "cuda" if torch.cuda.is_available() else "cpu"
)


class PiepleinePredictor:
    def __init__(
        self,
        segm_model_path: str,
        ocr_model_path: str,
        beam_model_path: str,
        ocr_config: dict[str, str],
    ):
        self.segm_predictor = segm.SEGMpredictor(model_path=segm_model_path)
        self.ocr_predictor = rec.OcrPredictor(
            ocr_model_path=ocr_model_path,
            beam_model_path=beam_model_path,
            config=ocr_config,
        )

    def __call__(self, img: np.ndarray) -> dict[str, list[list[int]] | str]:
        output: dict[str, list[list[int]] | str] = {}
        contours = self.segm_predictor(img)
        for contour in contours:
            if contour is not None:
                crop = help_fn.crop_img_by_polygon(img, contour)
                pred_text = self.ocr_predictor(crop)
                pred_polygon = [[int(i[0][0]), int(i[0][1])] for i in contour]
                output["polygon"] = pred_polygon
                output["text"] = pred_text
        return output


def init_predictor() -> PiepleinePredictor:
    return PiepleinePredictor(
        segm_model_path=CONFIG_JSON["SEGM_MODEL_PATH"],
        ocr_model_path=CONFIG_JSON["OCR_MODEL_PATH"],
        beam_model_path=CONFIG_JSON["BEAM_MODEL_PATH"],
        ocr_config=CONFIG_JSON["ocr_config"],
    )


def htr_predict(
    img_name: str, pipeline_predictor: PiepleinePredictor
) -> plt_image:
    image = cv2.imread(img_name)
    output = pipeline_predictor(image)
    vis = help_fn.get_image_visualization(image, output, "model/font.ttf")
    plt.figure(figsize=(20, 20))
    pred_img = plt.imshow(vis)
    pred_name = "pred-" + img_name.split("/")[-1]
    plt.savefig(pred_name)
    return pred_img

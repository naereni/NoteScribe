import json
import logging
import os
import sys
import warnings

import cv2
import torch
from tqdm import tqdm

import helper_functions as help_fn
import recognition as rec
import segmentation as segm

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
logger = logging.getLogger("detectron2")
logger.setLevel(logging.CRITICAL)

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

with open("config.json") as f:
    CONFIG_JSON = json.load(f)

CONFIG_JSON["device"] = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class PiepleinePredictor:
    def __init__(
        self, segm_model_path, ocr_model_path, beam_model_path, ocr_config
    ):
        self.segm_predictor = segm.SEGMpredictor(model_path=segm_model_path)
        self.ocr_predictor = rec.OcrPredictor(
            ocr_model_path=ocr_model_path,
            beam_model_path=beam_model_path,
            config=ocr_config,
        )

    def __call__(self, img):
        output = {"predictions": []}
        contours = self.segm_predictor(img)
        for contour in contours:
            if contour is not None:
                crop = help_fn.crop_img_by_polygon(img, contour)
                pred_text = self.ocr_predictor(crop)
                output["predictions"].append(
                    {
                        "polygon": [
                            [int(i[0][0]), int(i[0][1])] for i in contour
                        ],
                        "text": pred_text,
                    }
                )
        return output


def main():
    pipeline_predictor = PiepleinePredictor(
        segm_model_path=CONFIG_JSON["SEGM_MODEL_PATH"],
        ocr_model_path=CONFIG_JSON["OCR_MODEL_PATH"],
        beam_model_path=CONFIG_JSON["BEAM_MODEL_PATH"],
        ocr_config=CONFIG_JSON["ocr_config"],
    )
    pred_data = {}
    for img_name in tqdm(os.listdir(TEST_IMAGES_PATH)):
        image = cv2.imread(os.path.join(TEST_IMAGES_PATH, img_name))
        pred_data[img_name] = pipeline_predictor(image)

    with open(SAVE_PATH, "w") as f:
        json.dump(pred_data, f)

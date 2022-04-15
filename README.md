# Telegram bot for handwritten text recognition

## Install

To install you must have:
- ```GNU/Linux```
- ```python3.9```
- ```g++```

### Poerty
```
curl -sSL https://install.python-poetry.org | python3 - --preview
export PATH="/home/$USER/.local/bin:$PATH"
```

### htr-tg-bot
```  bash
git clone --recursive https://github.com/naereni/htr-tg-bot.git
cd htr-tg-bot && sh install.sh
```

If you will you encounter a problem with ctcdecode - you should run several times ```python setup.py install``` in directory **htr-tg-bot/third_party/ctcdecode**

### How should the solution work?

> A sequence of two models: segmentation and recognition. First, the segmentation model predicts the mask polygons of each word in the photo. Then these words are cut out of the image along the contour of the mask (drops are obtained for each word) and fed into the recognition model. The result is a list of recognized words with their coordinates.

### Models

**Instance Segmentation**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lednik7/nto-ai-text-recognition/blob/main/train/detectron2_segmentation_latest.ipynb)

- [X101-FPN](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn) from detectron2.model_zoo + augmentation + high resolution

**Character Recognition**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lednik7/nto-ai-text-recognition/blob/main/train/ocr_model.ipynb)

- CRNN architecture with Resnet-34 backbone and BiLSTM, pre-trained for the top 1 models of the competition [Digital Peter](https://github.com/sberbank-ai/digital_peter_aij2020)

**Beam Search**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lednik7/nto-ai-text-recognition/blob/main/dataset/make_kenlm_dataset_latest.ipynb)

- [KenLM](https://github.com/kpu/kenlm), trained on competition data [Feedback](https://www.kaggle.com/c/feedback-prize-2021/data ), **"Решу ОГЭ/ЕГЭ"**, and also [CTCDecoder](https://github.com/parlance/ctcdecode)

### Resources
**Christofari** with **NVIDIA Tesla V100** and docker image **jupyter-cuda10.1-tf2.3.0-pt1.6.0-gpu:0.0.82**

### More about training [here](https://github.com/Lednik7/nto-ai-text-recognition)

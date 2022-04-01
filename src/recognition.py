import numpy as np
import torch
import torch.nn as nn
import torchvision
from ctcdecode import CTCBeamDecoder

import helper_functions as help_fn
import tokenizer
import transforms


def get_resnet34_backbone(pretrained=True):
    m = torchvision.models.resnet34(pretrained=pretrained)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [
        input_conv,
        m.bn1,
        m.relu,
        m.maxpool,
        m.layer1,
        m.layer2,
        m.layer3,
    ]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(self, number_class_symbols):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((512, 32))
        self.bilstm = BiLSTM(512, 256, 2)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, number_class_symbols),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    return output


class OcrPredictor:
    def __init__(self, ocr_model_path, beam_model_path, config):
        self.tokenizer = tokenizer.Tokenizer(config["alphabet"])
        self.device = torch.device(config["device"])

        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(
            torch.load(ocr_model_path, map_location=self.device)
        )
        self.model.to(self.device)

        self.transforms = transforms.InferenceTransform(
            height=config["image"]["height"],
            width=config["image"]["width"],
        )

        labels_for_bs = """_@|!"%'()+,-./0123456789:;=?AEFIMNOSTW[]
        abcdefghiklmnopqrstuvwxyАБВГДЕЖЗИКЛМНОПРСТУХЦЧШЭЮЯ
        абвгдежзийклмнопрстуфхцчшщъыьэюяё№"""

        self.decoder = CTCBeamDecoder(
            list(labels_for_bs),
            model_path=beam_model_path,
            alpha=0.22,
            beta=1.1,
            cutoff_top_n=5,
            cutoff_prob=1,
            beam_width=10,
            num_processes=4,
            blank_id=0,
            log_probs_input=True,
        )

    def __call__(self, images):
        assert type(images) in (
            list,
            tuple,
            np.ndarray,
        ), """Input must contain np.ndarray, tuple or list,
              but found {type(images)}."""

        images = help_fn.black2white(images)
        images = [help_fn.process_image(images)]
        images = self.transforms(images)
        output = predict(images, self.model, self.tokenizer, self.device)
        beam_results, _, _, out_lens = self.decoder.decode(
            output.permute(1, 0, 2)
        )
        encoded_text = beam_results[0][0][: out_lens[0][0]]
        text_pred = self.tokenizer.decode_after_beam([encoded_text.numpy()])[0]
        return text_pred

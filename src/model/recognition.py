import numpy as np
import torch
import torchvision
from ctcdecode import CTCBeamDecoder

from model import helper_functions as help_fn
from model import tokenizer, transforms


def get_resnet34_backbone(pretrained: bool = True) -> torch.nn.Sequential:
    m = torchvision.models.resnet34(pretrained=pretrained)
    input_conv = torch.nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [
        input_conv,
        m.bn1,
        m.relu,
        m.maxpool,
        m.layer1,
        m.layer2,
        m.layer3,
    ]
    return torch.nn.Sequential(*blocks)


class BiLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class CRNN(torch.nn.Module):
    def __init__(self, number_class_symbols: int) -> None:
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(pretrained=False)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((512, 32))
        self.bilstm = BiLSTM(512, 256, 2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, number_class_symbols),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = torch.nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


def predict(
    images: torch.Tensor, model: CRNN, device: torch.device
) -> torch.Tensor:
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    return output


class OcrPredictor:
    def __init__(
        self, ocr_model_path: str, beam_model_path: str, config: dict[str, str]
    ):
        self.tokenizer = tokenizer.Tokenizer(config["alphabet"])
        self.device = torch.device(config["device"])

        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(
            torch.load(ocr_model_path, map_location=self.device)
        )
        self.model.to(self.device)

        self.transforms = transforms.InferenceTransform()

        self.decoder = CTCBeamDecoder(
            list(config["labels_for_bs"]),
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

    def __call__(self, images: np.ndarray) -> str:
        images = help_fn.black2white(images)
        images = [help_fn.process_image(images)]
        images = self.transforms(images)
        output = predict(images, self.model, self.device)
        beam_results, _, _, out_lens = self.decoder.decode(
            output.permute(1, 0, 2)
        )
        encoded_text = beam_results[0][0][: out_lens[0][0]]
        text_pred = self.tokenizer.decode_after_beam([encoded_text.numpy()])[0]
        return text_pred

import numpy as np
import torch
import torchvision


class Normalize:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255


class ToTensor:
    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.moveaxis(image, -1, 0)


def get_val_transforms() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            MoveChannels(),
            Normalize(),
            ToTensor(),
        ]
    )


class InferenceTransform:
    def __init__(self) -> None:
        self.transforms = get_val_transforms()

    def __call__(self, images: list[np.ndarray]) -> torch.Tensor:
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor

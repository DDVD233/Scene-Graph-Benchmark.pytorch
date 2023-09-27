import torch as th
from torchvision import transforms
from . import transforms_video


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(th.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


# class ToCTHW(object):
#     """
#     Args:
#         clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
#     Return:
#         clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
#     """
#
#     def __init__(self):
#         pass
#
#     def __call__(self, tensor):
#         return tensor.permute(1, 0, 2, 3)
#
#     def __repr__(self):
#         return self.__class__.__name__


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        self.mean = self.mean.to(tensor.device)
        self.std = self.std.to(tensor.device)
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):

    def __init__(self, image_size=224, min_scale=0.5,
        max_scale=1.0):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.norm,
            ]
        )

    def __call__(self, tensor):
        return self.transform(tensor)

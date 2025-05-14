import torch
import numpy as np
import random
from PIL import Image


def rgb_read(filename):
    data = Image.open(filename)
    rgb = (np.array(data) / 255.).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb


def depth_read(filename):
    data = Image.open(filename)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    depth = (np.array(data) / 65535.).astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(0)
    data.close()
    return depth


class RGBPReader(object):
    def __init__(self):
        self.rel = False

    @staticmethod
    def read_rgbraw(workspace,rgb_path, raw):
        rgb = rgb_read(rgb_path)
        if workspace is not None:
            (x1, y1, x2, y2) = workspace
            rgb = rgb[:, y1:y2,x1:x2]
        rgb = torch.nn.functional.interpolate(rgb.unsqueeze(0).to(torch.float32), size=(320, 448), mode='bilinear', align_corners=False)
        rgb = rgb.squeeze(0)
      
        raw = (raw / 65535.).astype(np.float32)
        
        if workspace is not None:
            (x1, y1, x2, y2) = workspace
            raw = raw[y1:y2,x1:x2]
        raw = torch.from_numpy(raw).unsqueeze(0)

        raw = raw.unsqueeze(0)
        if raw.shape[2] != rgb.shape[1]:
            raw = torch.nn.functional.interpolate(raw, (rgb.shape[1], rgb.shape[2]), mode='nearest-exact')

        return rgb.unsqueeze(0).to(torch.float32), raw.to(torch.float32)

    def read_data(self,workspace, rgb_path, depth):
        rgb, raw = self.read_rgbraw(workspace,rgb_path, depth)
        hole_raw = torch.ones_like(raw)
        hole_raw[raw == 0] = 0
        return rgb, raw, hole_raw

    @staticmethod
    def __min_max_norm__(depth):
        max_value = np.max(depth)
        min_value = np.min(depth)
        norm_depth = (depth - min_value) / (max_value - min_value + 1e-6)
        return norm_depth

    def adjust_domain(self, pred):
        pred = pred.squeeze().cpu().detach().numpy()
        if self.rel:
            pred = self.__min_max_norm__(pred)
        pred = np.clip(pred * 65535., 0, 65535).astype(np.int32)
        return pred


class DepthEvaluation(object):
    @staticmethod
    def rmse(depth, ground_truth):
        residual = ((depth - ground_truth) / 256.) ** 2
        residual[ground_truth == 0.] = 0.
        value = np.sqrt(np.sum(residual) / np.count_nonzero(ground_truth))
        return value

    @staticmethod
    def absRel(depth, ground_truth):
        diff = depth - ground_truth
        diff[ground_truth == 0] = 0.
        rel = np.sum(abs(diff) / (ground_truth + 1e-6)) / np.count_nonzero(ground_truth)
        return rel

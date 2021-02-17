import glob
import os

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = torch.device('cuda')
BATCH_SIZE = 5

RAW_VIDEO_PATH = os.path.join('data', 'CASIA-B-RAW', 'video')
OUTPUT_PATH = '/tmp/CASIA-B-MRCNN-V2'
SCORE_THRESHOLD = 0.9
BOX_RATIO_THRESHOLD = (1.25, 5)
MASK_BOX_RATIO = 1.7


class CASIABClip(Dataset):

    def __init__(self, filename):
        super().__init__()
        video, *_ = torchvision.io.read_video(filename, pts_unit='sec')
        self.frames = video.permute(0, 3, 1, 2) / 255

    def __getitem__(self, index) -> tuple[int, torch.Tensor]:
        return index, self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.to(DEVICE)
model.eval()


def result_handler(frame_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    for (box, label, score, mask) in zip(*result.values()):
        x0, y0, x1, y1 = box
        height, width = y1 - y0, x1 - x0
        if (BOX_RATIO_THRESHOLD[0] < height / width < BOX_RATIO_THRESHOLD[1]) \
                and (score > SCORE_THRESHOLD and label == 1):
            mask_height_offset_0 = (0.043 * height) / 2
            mask_height_offset_1 = (0.027 * height) / 2
            mask_y0 = (y0 - mask_height_offset_0).floor().int()
            mask_y1 = (y1 + mask_height_offset_1).ceil().int()
            mask_half_width = ((mask_y1 - mask_y0) / MASK_BOX_RATIO) / 2
            mask_xc = (x0 + x1) / 2
            mask_x0 = (mask_xc - mask_half_width).floor().int()
            mask_x1 = (mask_xc + mask_half_width).ceil().int()

            # Skip incomplete frames
            if (height < 64 or width < 64 / MASK_BOX_RATIO) \
                    or (mask_x0 < 0 or mask_x1 > 320) \
                    or (mask_y0 < 0 or mask_y1 > 240):
                continue

            cropped_frame = frame_[:, mask_y0:mask_y1 + 1, mask_x0:mask_x1 + 1]
            cropped_mask = mask[:, mask_y0:mask_y1 + 1, mask_x0:mask_x1 + 1]
            filtered_frame = cropped_frame * cropped_mask

            return cropped_mask, filtered_frame


SIL_PATH = os.path.join(OUTPUT_PATH, 'SIL')
SEG_PATH = os.path.join(OUTPUT_PATH, 'SEG')
if not os.path.exists(SIL_PATH):
    os.makedirs(SIL_PATH)
if not os.path.exists(SEG_PATH):
    os.makedirs(SEG_PATH)
RAW_VIDEO_REGEX = os.path.join(RAW_VIDEO_PATH, '*-*-*-*.avi')
for clip_filename in sorted(glob.glob(RAW_VIDEO_REGEX)):
    clip_name, _ = os.path.splitext(os.path.basename(clip_filename))
    clip_sil_dir = os.path.join(SIL_PATH, clip_name)
    clip_seg_dir = os.path.join(SEG_PATH, clip_name)
    if os.path.exists(clip_sil_dir):
        if len(os.listdir(clip_sil_dir)) != 0:
            continue
    else:
        os.mkdir(clip_sil_dir)
    if os.path.exists(clip_seg_dir):
        if len(os.listdir(clip_seg_dir)) != 0:
            continue
    else:
        os.mkdir(clip_seg_dir)

    clip = CASIABClip(clip_filename)
    clip_loader = DataLoader(clip, batch_size=BATCH_SIZE, pin_memory=True)

    with torch.no_grad():
        for frame_ids, frames in tqdm(
                clip_loader, desc=clip_name, unit='batch'
        ):
            frames = frames.to(DEVICE)
            for frame_id, frame, result in zip(
                    frame_ids, frames, model(frames)
            ):
                if len(result['boxes']) == 0:
                    continue
                if processed := result_handler(frame):
                    sil, seg = processed
                    frame_basename = f'{frame_id:04d}.png'
                    sil_filename = os.path.join(clip_sil_dir, frame_basename)
                    seg_filename = os.path.join(clip_seg_dir, frame_basename)
                    torchvision.utils.save_image(sil, sil_filename)
                    torchvision.utils.save_image(seg, seg_filename)

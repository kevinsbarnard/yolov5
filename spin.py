import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox


def load_image(image_info, path, img_size, stride, auto):
    """
    Load an image.
    """
    # TODO: Use image_info instead of path
    
    # Read image
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, f'Image Not Found {path}'

    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img, img0


def get_image_info_blocking():
    """
    Get info of an image to process. Block until available.
    """
    raise NotImplementedError  # TODO: Read from Redis queue


@torch.no_grad
def run(
    weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='', 
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    """
    Main function. Runs a loop around the model inference function.
    """
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    
    # Run inference loop
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    while True:
        try:
            # Get the image info
            image_info = get_image_info_blocking()
            
            # Load the image and source image
            im, im0s = load_image(image_info, None, imgsz, stride, auto=False)
            
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            # Inference
            pred = model(im, augment=augment, visualize=False)
            
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0 = im0s.copy()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    # TODO: Parse into response

        except KeyboardInterrupt:
            print('Stopped.')
        


if __name__ == '__main__':
    run()

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from random import random

from utils import datasets
import cv2
import numpy
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
myselect_index = 0

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from PIL import Image
from torch.autograd.variable import Variable
from torchvision import transforms

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'my_date/rightFrame/images/',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    num_tip = 0
    num_tool = 0
    # Table header settings
    title = ['lightness',
             'center_x1', 'center_y1', 'area1', 'tip_1', 'theta_1', 'class1',
             'center_x2', 'center_y2', 'area2', 'tip_2', 'theta_2', 'class2',
             'center_x3', 'center_y3', 'area3', 'tip_3', 'theta_3', 'class3',
             'center_x4', 'center_y4', 'area4', 'tip_4', 'theta_4', 'class4',
             'center_x5', 'center_y5', 'area5', 'tip_5', 'theta_5', 'class5',
             'center_x6', 'center_y6', 'area6', 'tip_6', 'theta_6', 'class6']
    label_sort = ['Ligation', 'scissors', 'Separating', 'Right', 'electric', 'ultrasound', 'Intestinal', 'Bipolar']

    # Image processing pipeline
    pipeline = transforms.Compose([
        # 分辨率重置为256
        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Get a smoke model
    fog_model = torch.load("fog_binary.pt")
    fog_model.eval()
    set_fog = 0  # Detection of smoke when energy instruments are present
    fog = 0.0
    with open("detect_result.csv", "w+", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(title)
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Load the embedded detection model file-------------------------------------------------------------------------------
        weights_tip = ROOT / 'tip_detect.pt'
        half_tip = False

        model_tip = DetectMultiBackend(weights_tip, device=device, dnn=dnn)
        stride_tip, names_tip, pt_tip, jit_tip, onnx_tip, engine_tip = model_tip.stride, model_tip.names, model_tip.pt, model_tip.jit, model_tip.onnx, model_tip.engine

        # Half
        half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        # Half_tip----------------------------------------------
        half_tip &= (pt_tip or engine_tip) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt_tip:
            model_tip.model.half() if half_tip else model_tip.model.float()

        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        model_tip.warmup(imgsz=(1, 3, *imgsz), half=half_tip)

        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            one_start = time.time()
            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            for i, det in enumerate(pred):  # per image

                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # im0 = cv2.resize(im0, dsize=(1920, 1080))
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0
                csv_data = []
                temp_row = []
                # ===========================fog 检测=============================================
                if set_fog == 1:
                    temp_fog = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    temp_fog = Image.fromarray(temp_fog)
                    temp_fog = pipeline(temp_fog)
                    temp_fog = torch.unsqueeze(temp_fog, 0)
                    temp_fog = Variable(temp_fog).to(device)

                    fog_output = fog_model(temp_fog)

                    fog_output = torch.sigmoid(fog_output)

                    fog = fog_output.cpu().detach().numpy()[0][1]
                    fog = round(fog, 2)
                    temp_row.append(str(fog))

                    set_fog = 0
                else:
                    temp_row.append(str(fog))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    tip_list = []  # Where to put the tip, remove it if it's too close
                    one_tip = 0

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            num_tool += 1
                            tip_num = 0
                            # -------------------------- Capture the detection area ---------------------------------------------
                            # ------------------ Moderately enlarge the capture area -----------------------------------------
                            if xyxy[0].item() > 40:
                                temp_x1 = int(xyxy[0].item() - 30)
                            else:
                                temp_x1 = int(xyxy[0].item())

                            if xyxy[1].item() > 40:
                                temp_y1 = int(xyxy[1].item() - 30)
                            else:
                                temp_y1 = int(xyxy[1].item())

                            if xyxy[2].item() < 1850:
                                temp_x2 = int(xyxy[2].item() + 30)
                            else:
                                temp_x2 = int(xyxy[2].item())

                            if xyxy[3].item() < 1000:
                                temp_y2 = int(xyxy[3].item() + 30)
                            else:
                                temp_y2 = int(xyxy[3].item())
                            # --------------- to store data in csv_data ----------------
                            csv_data.append(str((temp_x1 + temp_x2) / 2))
                            csv_data.append(str((temp_y1 + temp_y2) / 2))
                            csv_data.append(str((temp_y2 - temp_y1) * (temp_x2 - temp_x1)))
                            temp_src = imc[temp_y1:temp_y2, temp_x1:temp_x2]   # Intercept box area Crop [y1,y2: x1,x2]
                            # cv2.imshow("gg", temp_src)
                            # cv2.waitKey(0)
                            # Padded resize
                            img_tip = datasets.letterbox(temp_src, 640, 32, True)[0]
                            # Convert
                            img_tip = img_tip.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                            img_tip = numpy.ascontiguousarray(img_tip)

                            img_tip = torch.from_numpy(img_tip).to(device)
                            img_tip = img_tip.half() if half else img_tip.float()  # uint8 to fp16/32
                            img_tip /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(img_tip.shape) == 3:
                                img_tip = img_tip[None]  # expand for batch dim

                            pred_tip = model_tip(img_tip, augment=augment, visualize=visualize)

                            pred_tip = non_max_suppression(pred_tip, 0.30, iou_thres, classes, agnostic_nms,
                                                           max_det=max_det)
                            for i_tip, det_tip in enumerate(pred_tip):  # per image
                                # ------------------- writes the result of the judgment tip
                                if len(det_tip):
                                    tip_xes = []
                                    tip_yes = []
                                    det_tip[:, :4] = scale_coords(img_tip.shape[2:], det_tip[:, :4], temp_src.shape).round()
                                    for *xyxy_tip, conf_tip, cls_tip in reversed(det_tip):
                                        tip_conf = conf_tip.item()
                                        tip_num += 1
                                        tip_x = int((xyxy_tip[0].item() + xyxy_tip[2].item()) / 2) + temp_x1
                                        tip_y = int((xyxy_tip[1].item() + xyxy_tip[3].item()) / 2) + temp_y1
                                        tip_r = int(min((xyxy_tip[2].item() - xyxy_tip[0].item()), (
                                                xyxy_tip[3].item() - xyxy_tip[1].item())))

                                        tip_xes.append((xyxy_tip[0].item() + xyxy_tip[2].item()) / 2 + temp_x1)
                                        tip_yes.append((xyxy_tip[1].item() + xyxy_tip[3].item()) / 2 + temp_y1)

                                        num_tip += 1



                                csv_data.append(str(tip_num))
                                if len(det_tip) > 0:
                                    # Recording instrument posture
                                    the_tip_x = numpy.array(tip_xes).mean() - (temp_x1 + temp_x2)/2
                                    the_tip_y = numpy.array(tip_yes).mean() - (temp_y1 + temp_y2)/2
                                    theta = math.atan2(the_tip_y, the_tip_x)
                                    csv_data.append(str(round(theta, 2)))

                                else:
                                    csv_data.append("None")

                            c = int(cls)  # integer class
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                            if label.startswith("e") or label.startswith("B"): # Smoke detection starts when energy device is present
                                set_fog = 1
                            csv_data.append(str(label))



                    # --------------------- Sort the data ----------------------------
                    # The order of arrangement：Ligation, scissors, Separating, Right, electric, ultrasound, Intestinal, Bipolar
                    len_csv_data = len(csv_data)
                    while len(temp_row) != len_csv_data + 1:
                        for temp_row_index in range(0, 8):
                            temp_row_i = 5
                            while temp_row_i < len_csv_data:
                                if str(csv_data[temp_row_i]).startswith(label_sort[temp_row_index]):
                                    temp_row.append(csv_data[temp_row_i - 5])
                                    temp_row.append(csv_data[temp_row_i - 4])
                                    temp_row.append(csv_data[temp_row_i - 3])
                                    temp_row.append(csv_data[temp_row_i - 2])
                                    temp_row.append(csv_data[temp_row_i - 1])
                                    temp_row.append(csv_data[temp_row_i])
                                temp_row_i += 6

                    writer.writerow(temp_row)
                # Print time (inference-only)
                else:
                    writer.writerow(temp_row)

                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print("检测尖端数：", num_tip)
    print("检测器具数：", num_tool)
    csv_file.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/202201/new0117_model7/weights/best.pt',
                        help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'my_date/video/model7.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect/detect_tip', help='save results to project/name')
    parser.add_argument('--name', default='test0414', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    start = time.time()
    opt = parse_opt()
    main(opt)
    print("this task take %d"% (time.time() - start))
S
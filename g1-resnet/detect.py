
from tqdm import tqdm
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
from models.common import DetectMultiBackend
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams,create_dataloader
from utils.datasets_g1T import create_dataloader
from utils.general import (LOGGER, NCOLS,check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

def visualize_grid_to_grid(args,index, attention_name, att_map, image, alpha=0.6):
    mask = Image.fromarray(att_map).resize((image.shape[1],image.shape[0]))
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    
    ax[0].imshow(image)
    ax[0].axis('off')
    
    ax[1].imshow(image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.savefig(os.path.join(args,'attention_visualize_%s_%d.png'%(attention_name,index)))

@torch.no_grad()
def run(weights=ROOT / 'yolov3.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
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
    source = str(source)
    save_img = True


    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    print(stride)
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    #dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    pad = 0.5
    task='test'
    image_shape=(320,320)
    batch_size=1
    single_cls=False
    #task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataset = create_dataloader('path',250000,5,image_shape,task,batch_size, stride, single_cls, pad=pad, rect=pt,
                                       prefix=colorstr(f'{task}: '))[0]
    bs = 1  # batch_size

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 5,3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    pbar = tqdm(dataset, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    for batch_i, (im, targets, paths) in enumerate(pbar):
    #for image,target,path in dataset:
        t1 = time_sync()
        #im = torch.from_numpy(im).to(device)#shape:[3,480,640]
        if pt:
            im = im.to(device, non_blocking=True)#
            targets = targets.to(device)
        #image = im.squeeze().permute(1,2,0).cpu().detach().numpy()
        
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(paths).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        #attention visualise
        # Process predictions
        print(ROOT/save_dir / 'pred')
        if Path(ROOT/ save_dir / 'pred').exists():
            pass 
        else:
            os.mkdir(ROOT/save_dir / 'pred')
        if Path(ROOT/ save_dir / 'gt').exists():
            pass 
        else:
            os.mkdir(ROOT/save_dir / 'gt')

        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0= paths[0], im[0,4,:,:,:].cpu().numpy().copy() #, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / 'pred'   / p.name)[:-3]+('jpg')  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('')  # im.txt
            s += '%gx%g ' % im[0].shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            im1 = np.transpose(im0,[1,2,0])*255
            im1 = np.ascontiguousarray(im1)
            im1 = cv2.resize(im1,dsize=(320,320),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
            im1 = np.ascontiguousarray(im1)
            annotator = Annotator(im1, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((320,320), det[:, :4], (320,320)).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im1 = annotator.result()
            # Save results (image with detections)
            save_img=True
            if save_img:
                if True:
                    cv2.imwrite(save_path, im1)
                else:
                    pass
        

        p1 = targets[:,1:].cpu()
        p2 = targets.cpu()
        #print(p1.shape)
        det = np.ones_like(p2)
        #b = torch.from_numpy(a)
        #det = np.array([[0,0,0,0,1,0]],dtype=np.float16)
        det[:,0]=(p1[:,1]*320-p1[:,3]*160)
        det[:,2]=(p1[:,1]*320+p1[:,3]*160)
        det[:,1]=(p1[:,2]*320-p1[:,4]*160)
        det[:,3]=(p1[:,2]*320+p1[:,4]*160)
        det[:,5]=(p1[:,0])
        det=torch.from_numpy(det)
        det=det.to(device)
        #det[1]=det[1]-det[3]/2
        #det[2]=det[2]-det[4]/2


        seen += 1
        p, im0= paths[0], im[0,4,:,:,:].cpu().numpy().copy() #, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / 'gt'   /p.name)[:-3]+('jpg')  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('')  # im.txt
        s += '%gx%g ' % im[0].shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        im2 = np.transpose(im0,[1,2,0])*255
        im2 = np.ascontiguousarray(im2)
        im2 = cv2.resize(im2,dsize=(320,320),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        im2 = np.ascontiguousarray(im2)
        annotator = Annotator(im2, line_width=line_thickness, example=str(names))
        if len(targets):
            (w,h,c)=im0.shape
            # Rescale boxes from img_size to im0 size
            det[:,:4] = scale_coords((320,320), det[:,:4], (320,320)).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im2 = annotator.result()
            # Save results (image with detections)
            save_img=True
            if save_img:
                if True:
                    cv2.imwrite(save_path, im2)
                else:
                    pass

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp24/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='', help='file/dir/URL/glob, 0 for webcam')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='7', help='cuda device, i.e. 0 or 0,1,2,3,4,5,6,7 or cpu')
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
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
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
    opt = parse_opt()
    main(opt)

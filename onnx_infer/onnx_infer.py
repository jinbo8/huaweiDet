# coding=utf-8

"""
role : yolov7 onnx inference
"""
import colorsys
import glob
import os
import cv2
import numpy as np
import onnxruntime
import torch
import torchvision
import time
import random


class YOLOV7_ONNX(object):
    def __init__(self, onnx_path):
        '''initialization onnx'''
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        '''git input name'''
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        '''git output name'''
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, image_tensor):
        '''git input tensor'''
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_tensor
        return input_feed

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        '''Image normalization'''
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms(self, prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
        if prediction.dtype is torch.float16:
            prediction = prediction.float()  # to FP32
        xc = prediction[..., 4] > conf_thres  # candidates
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]

        return output

    def clip_coords(self, boxes, img_shape):
        '''Check whether the boundary is crossed'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):

        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                    img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_classes(self, classes_txt_path):
        with open(classes_txt_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    def infer(self, img_path, classes_txt, img_name, iou_thres, conf_thres, anchor_list):
        # Hyperparameter setting
        img_size = (640, 640)
        class_name, class_num = self.get_classes(classes_txt)

        hsv_tuples = [(x / class_num, 1., 1.) for x in range(class_num)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        anchor = np.array(anchor_list).astype(np.float64).reshape(3, -1, 2)
        stride = [8, 16, 32]
        area = img_size[0] * img_size[1]
        size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]
        feature = [[int(j / stride[i]) for j in img_size] for i in range(3)]

        # Read pictures
        src_img = cv2.imread(img_path)
        src_size = src_img.shape[:2]

        # Images are filled and normalized
        img = self.letterbox(src_img, img_size, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # normalized
        img = img.astype(dtype=np.float32)
        img /= 255.0

        # dimension expansion
        img = np.expand_dims(img, axis=0)

        # forward reasoning
        start = time.time()
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)

        # extract features

        y = []
        y.append(
            torch.tensor(pred[2].reshape(1, 3, 5 + class_num, 80, 80)).permute(0, 1, 3, 4, 2).reshape(-1, size[0] * 3,
                                                                                                      5 + class_num).sigmoid())
        y.append(
            torch.tensor(pred[1].reshape(1, 3, 5 + class_num, 40, 40)).permute(0, 1, 3, 4, 2).reshape(-1, size[1] * 3,
                                                                                                      5 + class_num).sigmoid())
        y.append(
            torch.tensor(pred[0].reshape(1, 3, 5 + class_num, 20, 20)).permute(0, 1, 3, 4, 2).reshape(-1, size[2] * 3,
                                                                                                      5 + class_num).sigmoid())

        grid = []
        for k, f in enumerate(feature):
            grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

        z = []
        for i in range(3):
            src = y[i]
            xy = src[..., 0:2] * 2. - 0.5
            wh = (src[..., 2:4] * 2) ** 2
            dst_xy = []
            dst_wh = []
            for j in range(3):
                dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + torch.tensor(grid[i])) * stride[i])
                dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
            src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
            src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
            z.append(src.view(1, -1, 5 + class_num))

        results = torch.cat(z, 1)
        results = self.nms(results, conf_thres, iou_thres)
        cast = time.time() - start
        print("cast time:{}".format(cast))

        # Map to the original image
        img_shape = img.shape[2:]
        for det in results:  # detections per image
            if det is not None and len(det):
                det[:, :4] = self.scale_coords(img_shape, det[:, :4], src_size).round()

        if det is not None and len(det):
            self.draw(src_img, det, img_name, output_results_dir, class_name, colors)

    def plot_one_box(self, x, img, color=None, label=None, conf=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            information = label + " " + conf
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(information, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, information, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)

    def draw(self, img, boxinfo, img_name, output_results_dir, class_name, colors):
        for *xyxy, conf, cls in boxinfo:
            label = str(class_name[int(cls)])
            conf = str(format(float(conf), ".4f"))
            self.plot_one_box(xyxy, img, label=label, conf=conf, color=colors[int(cls)], line_thickness=2)
        # show and save image
        # cv2.namedWindow("dst",0)
        # cv2.imshow("dst", img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(output_results_dir, img_name), img)
        return 0


if __name__ == "__main__":

    onnx_path = r"model_data/models.onnx"  # Into the model
    img_dir = r"img"  # Import the directory of the picture you want to reason about
    output_results_dir = r"./"  # Where you want the output image to be saved
    classes_txt = "model_data/coco_classes.txt"  # contain classes name txt

    iou_thres = 0.2
    conf_thres = 0.3

    anchor_list = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]

    if os.path.isdir(output_results_dir) == False:
        os.mkdir(output_results_dir)
    img_list = glob.glob(os.path.join(img_dir, "*.jpg"))

    model = YOLOV7_ONNX(onnx_path=onnx_path)
    for img_path in img_list:
        img_name = img_path.split("\\")[-1]
        model.infer(img_path=img_path, classes_txt=classes_txt, img_name=img_name, iou_thres=iou_thres,
                    conf_thres=conf_thres, anchor_list=anchor_list)
import argparse
import time
import os
import copy
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import  non_max_suppression, scale_coords
from plate_recognition.plate_rec import allFilePath, init_model
from plate_recognition.double_plate_split_merge import get_split_merge
from utils.datasets import letterbox
from PIL import Image, ImageDraw, ImageFont
from models.colorNet import myNet_ocr_color
import json
from paddleocr import PaddleOCR


def roi_det(roi_json, img):
    """ 通过掩码方式获取图像的Roi区域进行车牌识别"""
    img_h, img_w = img.shape[0], img.shape[1]
    img_zero = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    with open(roi_json, "r", encoding="utf-8") as f:
        roi_size = json.load(f)
    cam_h_s = roi_size['camera1']["height_start"]  # 检测的ROI区域在图像上的 高度起始 位置
    cam_h_e = roi_size['camera1']["height_end"]  # 检测的ROI区域在图像上的 高度结束 位置
    cam_w_s = roi_size['camera2']["width_start"]  # 检测的ROI区域在图像上的 宽度起始 位置
    cam_w_e = roi_size['camera1']["width_end"]  # 检测的ROI区域在图像上的 宽度结束 位置

    img_zero[cam_h_s:cam_h_e, cam_w_s:cam_w_e, :] = 255
    img_mask = cv2.bitwise_and(img, img_zero)

    # save_img_path = os.path.join(opt.output, os.path.basename(pic_))
    # cv2.imwrite(save_img_path, img_mask)
    # time.sleep(10000)
    return img_mask


corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # colors of four corner points
plate_colors = ['黑色','蓝色','绿色','白色','黄色']
colors = {"蓝色": (255, 0, 0),  # blue
          "绿色": (0, 128, 0),  # green
          "黄色": (0, 128, 255),  # yellow
          "白色": (210, 210, 210),  # white
          "黑色": (0, 0, 0),  # black
          "其他": (255, 0, 255)  # other
          }
# colors = {0: (255, 0, 0),  # blue
#           1: (0, 128, 0),  # green
#           2: (0, 128, 255),  # yellow
#           3: (255, 255, 255),  # white
#           4: (0, 0, 0),  # black
#           -1: (255, 0, 255)  # other
#           }
plate_chr="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"


def order_points(pts):  # 关键点按照（左上，右上，右下，左下）排列
    """四个角点：左上角开始，顺时针"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """ 透视/矫正变换"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def decodePlate(preds):
    pre=0
    newPreds=[]
    index=[]
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index


def get_plate_result(img, device, model, img_size=(48,168)):
    # img = image_processing(img, device, img_size)
    preds, preds_color = model(img)
    preds = preds.argmax(dim=2)
    preds_color = preds_color.argmax()
    preds_color = preds_color.item()
    preds = preds.view(-1).detach().cpu().numpy()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate, plate_colors[preds_color]


def get_plate_chars(preds):
    """ 获取车牌字符 """
    preds = torch.softmax(preds,dim=-1)
    probs, indexs = preds.max(dim=-1)
    # print(probs, indexs)
    plate_chars = []
    plate_char_probs = []
    for i in range(indexs.shape[0]):
        index = indexs[i].view(-1).detach().cpu().numpy()
        prob = probs[i].view(-1).detach().cpu().numpy()
        new_preds, new_index = decodePlate(index)
        prob = prob[new_index]
        plate = ""
        for i in new_preds:
            plate += plate_chr[i]
        plate_chars.append(plate)
        plate_char_probs.append(prob)
    return plate_chars, plate_char_probs  # 返回车牌号以及每个字符的概率


def image_processing(img, device):
    """ 图像处理成固定的尺寸，归一化 """
    img = cv2.resize(img, (168, 48))

    # normalize
    mean_value, std_value = (0.588, 0.193)
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])

    return img

def draw_result(img, dict_list):
    # img = img.astype(np.uint64)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for result in dict_list:
        box = result['box']
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        (img_w, img_h) = img.size
        box[0] = max(0, int(x - padding_w))
        box[1] = max(0, int(y - padding_h))
        box[2] = min(img_w, int(box[2] + padding_w))
        box[3] = min(img_h, int(box[3] + padding_h))

        corners = result['corners']
        result_str = result['plate_char']
        if len(result_str) < 7:
            result_str = ""
        elif result['plate_layer'] == 0:
            result_str += "自定义"

        r = 3
        for i in range(4):
            draw.ellipse([int(corners[i][0])-r, int(corners[i][1])-r, int(corners[i][0])+r, int(corners[i][1])+r],
                         fill=corner_colors[i], outline=corner_colors[i])
        draw.rectangle(box, width=3, outline=tuple(colors[result['plate_color']]))

        fontsize = max(round(max(img.size) / 100), 12)
        font = ImageFont.truetype("data/platech.ttf", fontsize)
        txt_width, txt_height = font.getsize(result_str)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(colors[result['plate_color']]))
        draw.text((box[0], box[1] - txt_height + 1), result_str, fill=(255, 255, 255), font=font)

    return np.asarray(img)



def roi_position_value(roi_json):
    """ 通过掩码方式获取图像的Roi区域进行车牌识别"""

    with open(roi_json, "r", encoding="utf-8") as f:
        roi_size = json.load(f)
    cam_h_s = roi_size['camera1']["height_start"]  # 检测的ROI区域在图像上的 高度起始 位置
    cam_h_e = roi_size['camera1']["height_end"]  # 检测的ROI区域在图像上的 高度结束 位置
    cam_w_s = roi_size['camera2']["width_start"]  # 检测的ROI区域在图像上的 宽度起始 位置
    cam_w_e = roi_size['camera1']["width_end"]  # 检测的ROI区域在图像上的 宽度结束 位置

    return cam_h_s, cam_h_e, cam_w_s, cam_w_e

# 绘制多边形
def draw_polylines(path):
    # 多变形(或矩形)区域, 第一位为宽度值，第二位为高度值，左上角坐标为（0,0），右下角坐标（width, height）
    img = cv2.imread(path)
    point1 = [int(img.shape[1] * 0.1), int(img.shape[0] * 1)]
    point2 = [int(img.shape[1] * 0.4),int(img.shape[0] * 0.1)]
    point3 = [int(img.shape[1] * 0.65), int(img.shape[0] * 0.1)]
    point4 = [img.shape[1], int(img.shape[0] * 0.55)]
    point5 = [img.shape[1], int(img.shape[0] * 1)]
    points = np.array([point1, point2, point3, point4, point5], np.int32)
    cv2.polylines(img, [points], True, (0, 0, 255), 2)  # True表示该图形为封闭图形

    # 设置蒙版
    zeros = np.zeros((img.shape), dtype=np.uint8)
    mask = cv2.fillPoly(zeros, [points], color=(100, 100, 100))  # 矩形填充颜色
    # mask = cv2.fillPoly(img, [points], color=[0, 255, 0])
    img = mask * 0.6 + img
    # print(points)

    cv2.imwrite(f"alpha_{1}.jpg", img)
    # cv2.imshow('Polygon', img)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()


def draw_mask_img(image_path, alpha=0.45):
    # 矩形区域, 第一位为宽度值，第二位为高度值，左上角坐标为（0,0），右下角坐标（width, height）

    blue = (220, 0, 0)
    mask_color = (120, 100, 100)

    img = cv2.imread(image_path)
    print(img.shape)
    overlay = img.copy()
    output = img.copy()

    width_start = int(img.shape[1]*0.2)  # 最左边为0
    width_end = int(img.shape[1]*0.95)  # 最右边为最大值
    height_start = int(img.shape[0]*0.1)  # 左上角为0
    height_end = int(img.shape[0]*1)  # 左下角为最大值

    left_bottom = (width_start, height_end)
    right_top = (width_end, height_start)

    # 绘制矩形和蒙版
    cv2.rectangle(overlay, left_bottom, right_top, mask_color, -1)  # 蒙版区域设置
    cv2.rectangle(overlay, left_bottom, right_top, blue, 6)  # 蒙版区域设置
    cv2.putText(overlay, f"alpha:{alpha},lt:{left_bottom}, rb{right_top}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, blue, 6)  # 蒙版透明度显示字体
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # 结果显示与保存
    cv2.imshow("Mask", output)
    cv2.waitKey(3000)
    cv2.imwrite(f"alpha_{alpha}.jpg", output)


def draw_mask_img_roi(img, roi_json, alpha=0.45):
    cam_h_s, cam_h_e, cam_w_s, cam_w_e =  roi_position_value(roi_json)
    # 矩形区域, 第一位为宽度值，第二位为高度值，左上角坐标为（0,0），右下角坐标（width, height）

    blue = (220, 0, 0)
    mask_color = (120, 100, 100)

    overlay = img.copy()
    output = img.copy()

    width_start = int(cam_w_s)  # 最左边为0
    width_end = int(cam_w_e)  # 最右边为最大值
    height_start = int(cam_h_s)  # 左上角为0
    height_end = int(cam_h_e)  # 左下角为最大值

    left_bottom = (width_start, height_end)
    right_top = (width_end, height_start)

    # 绘制矩形和蒙版
    cv2.rectangle(overlay, left_bottom, right_top, mask_color, -1)  # 蒙版区域设置
    cv2.rectangle(overlay, left_bottom, right_top, blue, 6)  # 蒙版区域设置
    cv2.putText(overlay, f"alpha:{alpha},lt:{left_bottom}, rb{right_top}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                blue, 6)  # 蒙版透明度显示字体
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # 结果显示与保存
    # cv2.imshow("Mask", output)
    # cv2.waitKey(5000)
    return output

# roi绘制多边形+蒙版
def roi_draw_polylines(img):
    # 多变形(或矩形)区域, 第一位为宽度值，第二位为高度值，左上角坐标为（0,0），右下角坐标（width, height）
    point1 = [int(img.shape[1] * 0.1), int(img.shape[0] * 1-3)]
    point2 = [int(img.shape[1] * 0.4),int(img.shape[0] * 0.1)]
    point3 = [int(img.shape[1] * 0.65), int(img.shape[0] * 0.1)]
    point4 = [img.shape[1]-3, int(img.shape[0] * 0.55)]
    point5 = [img.shape[1]-3, int(img.shape[0] * 1-3)]
    points = np.array([point1, point2, point3, point4, point5], np.int32)
    cv2.polylines(img, [points], True, (0, 0, 255), 2)  # True表示该图形为封闭图形

    # 设置蒙版
    zeros = np.zeros((img.shape), dtype=np.uint8)
    zeros = np.zeros((img.shape))
    mask = cv2.fillPoly(zeros, [points], color=(100, 100, 100))  # 矩形填充颜色
    # mask = cv2.fillPoly(img, [points], color=[0, 255, 0])
    img = mask * 0.6 + img

    return img



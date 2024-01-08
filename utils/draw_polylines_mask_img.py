# # name: JinboWang
# # Dev time: 2022/9/14

import cv2
import numpy as np
import matplotlib.pyplot as plt



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
    print(f"img:{img.shape}")
    print(f"img:{type(img)}")
    cv2.imwrite(f"alpha_{12}.jpg", img)
    cv2.imshow('Polygon', img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


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


if __name__ == '__main__':

    path = r"/home/dell/桌面/license_plate_rec/licence_plate_rec_infer/data/test_imgs/2023-1-10_11_40_17_580_vehicleBackgroundImage_2.jpg"
    # draw_mask_img(path)  # 绘制矩形蒙版
    draw_polylines(path)  # 绘制多边形区域
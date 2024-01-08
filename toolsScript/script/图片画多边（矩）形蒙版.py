import os

import cv2
import numpy as np


def draw_mask_place(img_path,  save_dir,  x_min=0, y_min=527, x_max=0, y_max=1080):
    """
        功能： 为图片特定区域绘制蒙版，对图片进行保存。

    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        pass

    n = 0
    images = os.listdir(img_path)
    for pic_ in images:
        n += 1
        src_img_path = os.path.join(img_path, pic_)

        src_img = cv2.imread(src_img_path)
        if src_img.shape[-1] == 4:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)  # BGR
        black = np.zeros(src_img.shape, np.uint8)

        # 车牌识别ROI区域
        # x_min, y_min = 10, 10
        # x_max, y_max = 800, 800

        # 对特定区域绘制颜色,
        # 颜色(B,G,R), 线的粗细
        # mask_color = (255, 0, 0) # blue
        mask_color = (255, 105, 65) # blue
        cv2.rectangle(src_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.rectangle(black, (x_min, y_min), (x_max, y_max), mask_color, -1)  # 注意在 blk的基础上进行绘制；

        # black后的参数指定mask区域蒙版的透明度， 并保存
        picture = cv2.addWeighted(src_img, 1.0, black, 0.5, 1)
        res_save_path = os.path.join(save_dir, pic_)
        cv2.imwrite(res_save_path, picture)
        print(f"n:{n}")
        # 图片结果可视化
        # cv2.imshow('img', picture)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()



def draw_mask_with_polygen(img_path,  save_dir, x_y_position=[[100, 100], [200, 50], [300, 100], [300, 300], [200, 350], [100, 300]]):
    """
        功能： 为图片特定区域绘制多边形蒙版，对图片进行保存。
    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        pass

    # 定义多边形顶点坐标
    # x_y_position = [[100, 100], [200, 50], [300, 100], [300, 300], [200, 350], [100, 300]]
    # x_y_position = [[231, 527], [572, 526], [287, 899], [6, 899], [0,760]]  # R9
    x_y_position = [[810, 316], [988, 316], [827, 1080], [330, 1080]]  #r12
    # x_y_position = [[615, 500], [760, 500], [867, 1080], [510, 1080]]  #r17
    # x_y_position = [[397, 450], [750, 450], [382, 1080], [0, 1080],[0,742]]  #r71
    # x_y_position = [[331, 539], [531, 539], [283, 1080], [0, 1080],[0,938]]  #r5
    points = np.array(x_y_position, np.int32).reshape((-1, 1, 2))
    mask_color = (255, 105, 65)   # mask区域绘制颜色, 颜色(B,G,R), 线的粗细,(255, 0, 0)：blue
    n = 0
    images = os.listdir(img_path)
    for pic_ in images:
        n += 1
        src_img_path = os.path.join(img_path, pic_)
        src_img = cv2.imread(src_img_path)
        image = np.zeros(src_img.shape, np.uint8) # 创建一个黑色背景的图像
        if src_img.shape[-1] == 4:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)  # BGR

        # 绘制多边形和填充颜色，第一个参数是图像，第二个参数是多边形的顶点坐标，第三个参数表示是否闭合多边形，第四个参数是颜色，第五个参数是线宽
        # cv2.polylines(image, [points, points2], isClosed=True, color=mask_color, thickness=2)
        cv2.polylines(image, [points], isClosed=True, color=mask_color, thickness=2)
        cv2.fillPoly(image, [points], color=mask_color)
        picture2 = cv2.addWeighted(src_img, 1.0, image, 0.5, 1)  # image 后面的值是mask的颜色透明度
        res_save_path = os.path.join(save_dir, pic_)
        cv2.imwrite(res_save_path, picture2)
        print(f'n:{n}')
        # 显示绘制的图像
        # cv2.imshow('Polygon', picture2)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()


if __name__ == '__main__':

    img_path = '/media/dell/Elements/tyjt/ITS/r12/r12out/car_draw_line_res3'
    save_dir = '/media/dell/Elements/tyjt/ITS/r12/r12out/draw_mask_plate_res4'

    # 绘制矩形和填充颜色
    # draw_mask_place(img_path, save_dir)

    # 绘制多边形和填充颜色
    draw_mask_with_polygen(img_path, save_dir)

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
import paddle

from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
paddle.get_device()


corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # colors of four corner points
#jwang:现在版本 0821
# plate_colors = ['黑色', '蓝色', '绿色', '白色', '黄色']  # {0：'黑色', 1：'蓝色', 2：'绿色', 3：'白色', 4：'黄色'}
# colors = {"蓝色": (255, 0, 0),  # blue
#           "绿色": (0, 128, 0),  # green
#           "黄色": (0, 128, 255),  # yellow
#           "白色": (210, 210, 210),  # white
#           "黑色": (0, 0, 0),  # black
#           "其他": (255, 0, 255)  # other
#           }

# 颜色参考：https://blog.csdn.net/m0_49789433/article/details/120077880
# 白色：rgb(255,255,255)
# 黑色：rgb(0,0,0)
# 红色：rgb(255,0,0)
# 绿色：rgb(0,255,0)
# 蓝色：rgb(0,0,255)

plate_colors = ['黑色', '蓝色', '绿色', '白色', '黄色']  # {0：'黑色', 1：'蓝色', 2：'绿色', 3：'白色', 4：'黄色'}

# rgb
colors = {"蓝色": (255, 0, 0),     # blue
          "绿色": (0, 128, 0),     # green
          "黄色": (0, 128, 255),   # yellow
          "白色": (210, 210, 210), # white
          "黑色": (0, 0, 0),       # black
          }

plate_chr = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"


def order_points(pts):  # 关键点按照（左上，右上，右下，左下）排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):  #透视变换
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
    # print(img.shape)
    # new = Image.fromarray(np.uint8(img[:, :, :]))  # RGB
    # new = Image.fromarray(np.uint8(img[:, :, ::-1])) #
    # new.show()

    img = cv2.resize(img, (168, 48))
    # normalize
    mean_value, std_value = (0.588, 0.193)
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])

    return img


def test_1_3_car_ocr(ocr):
    # ocr.ocr：预测车牌字符和置信度
    t1 = time.time()
    path_img = '/home/dell/桌面/ccp-lpr_v2/result/3840_2160.png'
    img = Image.open(path_img)
    img = np.asarray(img)
    img = torch.tensor(img)
    img.permute(2, 0, 1)
    img = np.asarray(img)
    print(f"\n图片名称：{path_img}")
    t1 = time.time()
    plate_chars = ocr.ocr(img, cls=False, det=False)  # 默认即可， 也可以单独注释本行使用以下plate_rec_color_model(torch.Tensor(ims).to(device)) 进行车牌字符识别
    print(f"车牌识别结果：{plate_chars[0][0][0]}, ocr推理时间：{time.time() - t1}")


def img_show(img):
    new = Image.fromarray(np.uint8(img[:, :, :]))  # 原图显示BGR
    new.show()
    # print(f"type:{type(img)}")
    # print(f"img:{img}")
    # new1 = Image.fromarray(np.uint8(img[:, :, ::-1]))  #变换通道后显示RGB
    # print(new1)
    # print('new1:',type(new1))
    # new1.show()
    # time.sleep(50000)

def img_save(img):
    save_img = img
    plt.imshow(save_img[:,:,:])  # （w,h,channel）通道反转实现BGR转RGB
    plt.savefig("./result1/src.jpg")  # 图片存储
    plt.imshow(save_img[:,:,::-1])  # （w,h,channel）通道反转实现BGR转RGB
    plt.savefig("./result1/src_exchange_channels.jpg")  # 图片存储
    cv2.waitKey(0)


def img_channel_show(img):
    # b, g, r = cv2.split(img)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    # img_matplotlib = cv2.merge([r, g, b])  #修改通道顺序
    plt.imshow(B)
    plt.show()
    plt.savefig("./result1/B.jpg")  # 图片存储
    cv2.waitKey(0)

    # plt.imshow(g)
    plt.show()
    plt.savefig("./result1/G.jpg")  # 图片存储
    cv2.waitKey(0)

    plt.imshow(R)
    plt.show()
    plt.savefig("./result1/R.jpg")  # 图片存储
    cv2.waitKey(0)


def detect(orgimg, device, plate_bbox_corner_model, plate_rec_color_model, img_size):
    # 传入的是BGR 图像
    # orgimg: BGR
    conf_thres = 0.3
    iou_thres = 0.5
    dict_list = []

    # Process img
    im0 = copy.deepcopy(orgimg)
    imgsz = (img_size, img_size)
    img = letterbox(im0, new_shape=imgsz)[0]  # BGR

    # 图片通道变换前后可视化与保存
    # img_show(img)  # 显示图片
    # img_save(img)  # 保存图片
    # img_channel_show(img)

    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x640X640

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # # 中间结果保存
    # img_new = np.asarray(img.cpu())
    # with open("img_res.txt", 'wb') as f:
    #     f.write(img.cpu().numpy().astype('float32'))

    # Inference & Apply NMS
    pred = plate_bbox_corner_model(img)[0]  # jwang：输入的是RGB图像
    # 预测结果保存到txt进行查看
    res = np.asarray(pred.cpu())
    res = res.squeeze()
    # np.savetxt("pred_res.txt", res)
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, kpt_label=4)

    # box_res = []  #用作保存检测bbox结果

    # Process detections
    t_total_start = time.time()
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=4, step=3)

            ims = []
            roi_imgs = []
            for j in range(det.size()[0]):
                result_dict = {}

                # 以下四行用作bbox和置信度结果保存
                # xyxy_s = det[j, :4].view(-1).tolist()
                # conf_s = det[j, 4].cpu().numpy()
                # box_save = [int(xyxy_s[0]), int(xyxy_s[1]), int(xyxy_s[2]), int(xyxy_s[3]), conf_s.item()]
                # box_res.append(box_save)
                # print(f"box:{box_save}")

                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 6:].view(-1).tolist()
                # 为四个脚点
                landmarks = [landmarks[0],landmarks[1],landmarks[3],landmarks[4],landmarks[6],landmarks[7],landmarks[9],landmarks[10]]
                cls = det[j, 5].cpu().numpy()  # 0 self-defined, 1 single, 2 double
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                landmarks_np = np.zeros((4, 2))

                # 格式化处理四个脚点
                for m in range(4):
                    point_x = int(landmarks[2 * m])
                    point_y = int(landmarks[2 * m + 1])
                    landmarks_np[m] = np.array([point_x, point_y])

                roi_img = four_point_transform(orgimg, landmarks_np)  # 矩形矫正, return: 矫正后的车牌
                """ ***** import  测试车牌颜色通道变换是否影响车牌颜色的识别 ***** """
                # img_show(roi_img)  # 显示图片 rgb
                # roi_img = roi_img[:, :, ::-1]  # 测试输入图片的通道变换RGB后是否会影响车牌颜色识别，实验结果表明，颜色通道变换会影响车牌颜色识别和字符识别。
                # img_show(roi_img)
                # img_save(img)  # 保存图片

                if int(cls) == 2:  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
                    roi_img = get_split_merge(roi_img)

                # 处理成固定的尺寸，归一化， 中间版本使用，结合OCR一起使用效果更好，也可以不要，以下两行和“ preds_chars, preds_color = plate_rec_color_model(torch.Tensor(ims).to(device))”一起使用
                im = image_processing(roi_img, device)   # 图像处理成固定的尺寸，归一化

                # print('color_rec_shape', im.shape)

                ims.append(im)
                # roi_imgs.append(roi_img)

                # 测试 车身下半1/3直接OCR结果
                # test_1_3_car_ocr(ocr)
                # print(f"ocr_images_channel:{roi_img.shape}")
                # img_show(roi_img)
                plate_chars = ocr.ocr(roi_img, cls=False, det=False)  # 默认即可， 也可以单独注释本行使用以下plate_rec_color_model(torch.Tensor(ims).to(device)) 进行车牌字符识别
                # print(plate_chars[0][0][1])  #

                # --- 20231009 功能： 查看车牌字符ocr预测的所有结果，字符+字符置信度
                for line in plate_chars:
                    print(f"line lp::{line}")


                # --- update 20231008  功能： 去除非第一位出现的汉字
                lp_chars = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新民航危险品'
                plate_char_res = plate_chars[0][0][0]  # 车牌字符

                new_char = plate_char_res[1:]
                for repeat in lp_chars:
                    if repeat in new_char:
                        new_char = new_char.replace(repeat, "")
                plate_char_res = plate_char_res.replace(plate_char_res[1:], new_char)
                result_dict['plate_char'] = plate_char_res  # 车牌所有字符
                # result_dict['plate_char'] = plate_chars[0][0][0]  # 车牌所有字符
                result_dict['plate_char_conf'] = plate_chars[0][0][1]  # 车牌所有字符置信度，输出车牌置信度20231103
                # print(f"plate_chars[0][0]:{plate_chars[0][0][1]}, plate_chars[0]:{plate_chars[0]}")
                # ---20231008 end ---

                # result_dict['plate_char_conf'] = float(plate_chars[0][0][1])  # 车牌所有字符检测结果的置信度
                result_dict['box'] = box  # 车牌bbox坐标
                result_dict['detect_conf'] = float(conf)  # 检测区域得分，置信度
                result_dict['corners'] = landmarks_np.tolist()  # 车牌角点坐标
                result_dict['plate_layer'] = int(cls)  # 0自定义车牌 1单层车牌 2双层车牌
                # print(f"infer-time: {time.time()-t1}")
                # print(result_dict)  # 打印检测结果
                # time.sleep(1000000)
                dict_list.append(result_dict)

            # 中间版本使用，结合OCR一起使用效果更好，也可以不要，得到车牌的颜色和字符， 进行了批处理
            # preds_chars, preds_color = plate_rec_color_model(torch.Tensor(ims).to(device))  # 对所有的车牌区域进行字符识别， # 中间版本使用，结合OCR一起使用效果更好，也可以不要

            print('color_rec', torch.Tensor(ims).shape)

            preds_color = plate_rec_color_model(torch.Tensor(ims).to(device))
            # plate_chars, plate_char_probs = get_plate_chars(preds_chars)
            for w in range(len(dict_list)):
                # 如果不适应paddleOCR 需要打开以下两行代码，与以上代码preds_chars, preds_color = plate_rec_color_model(torch.Tensor(ims).to(device)) 一起使用
                # dict_list[w]['plate_char'] = plate_chars[w]  # 车牌字符
                # dict_list[w]['plate_char_prob'] = plate_char_probs[w].tolist()  # 每个字符的概率
                pred_color = torch.softmax(preds_color[w], dim=-1)
                color_conf, color_index = torch.max(pred_color, dim=-1)

                # 车牌颜色检测结果二次校验 20230911 update-----start-----------------
                plate_color = plate_colors[color_index]  # 颜色
                if plate_color == "黑色":
                    if (not (dict_list[w]['plate_char'][0] == '使')) and (not (dict_list[w]['plate_char'][-1] in ['领', '港', '澳'])) and (not (dict_list[w]['plate_char'][0:2] in ['粤Z'])):
                        if len(dict_list[w]['plate_char']) == 7:
                            plate_color = '蓝色'
                        if len(dict_list[w]['plate_char']) == 8:
                            plate_color = '绿色'
                if plate_color == "白色":
                    if not (dict_list[w]['plate_char'][-1]=='警'):
                        if len(dict_list[w]['plate_char']) == 7:
                            plate_color = '蓝色'
                        if len(dict_list[w]['plate_char']) == 8:
                            plate_color = '绿色'
                if plate_color == '黄色':  # dict_list[w]['plate_char'][-1] in ['挂', '学']:
                    if len(dict_list[w]['plate_char']) == 8:
                        plate_color = '绿色'
                if int(dict_list[w]['plate_layer'])==1:
                    plate_color = '黄色'

                # update  --20230920--
                if plate_color == '蓝色':
                    if len(dict_list[w]['plate_char']) == 8:
                        plate_color = '绿色'
                if plate_color == '绿色':
                    if len(dict_list[w]['plate_char']) == 7:
                        plate_color = '蓝色'

                dict_list[w]['plate_color'] = plate_color
                dict_list[w]['color_conf'] = color_conf.item()  # 颜色置信度
                # 车牌颜色检测结果二次校验 20230911 update-----end-----------------
            # time.sleep(10000)

    print(f"{dict_list}\n")
    # return dict_list, box_res  #用作保存检测bbox结果
    print(f"耗时：{time.time()-t_total_start}")
    return dict_list


def img_box_res_save(abs_path, pic_, box_res):
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)

    res_name = pic_.split('/')[-1][:-3] + 'txt'
    new_path = os.path.join(abs_path, res_name)
    with open(new_path, 'w', encoding='utf-8') as f:
        for i in box_res:
            for j in i:
                f.write(str(j)+' ')
            f.write('\n')


def img_res_save(abs_path, pic_):
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)

    res_name = pic_.split('/')[-1][:-3] + 'txt'
    new_path = os.path.join(abs_path, res_name)
    with open(new_path, 'w', encoding='utf-8') as f:
        for i in dict_list:
            f.write(str(i) + '\n')


def draw_result(img, dict_list):
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
        if len(result_str) < 7:  #--20230923测试
        # if len(result_str) < 4:
            result_str = ""
        elif result['plate_layer'] != 0 and result['plate_layer'] != 1:
            result_str += "自定义"

        # print(f"result_str:{result_str}")
        # 加入len(result_str)!=0判断后，同时有车牌字符识别结果才进行绘制到图像上，不加以下判断，只有车牌检测框也会绘制到图像上
        if len(result_str)!=0:
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


if __name__ == '__main__':

    """
        1.全类别NMS计算，不按照按类别的方式进行NMS处理
        https://blog.csdn.net/sinat_41942180/article/details/129280379
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_det_deploy/yolov7-lite-s.pt', help='model.pt path(s)')  # 1.目标检测模型
    parser.add_argument('--rec_model', type=str, default='weights/color_rec_deploy/plate_rec_color0325.pth', help='model.pt path(s)')  # REC颜色识别模型测试
    # parser.add_argument('--source', type=str, default='inferdata/testFullImgS17_BLOCK', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='/media/dell/sata4t/s17/gpu1/R12_M_CamE', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='/home/dell/桌面/eval_sample/imageSample/DetOCRSample/suzhou15km', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='/home/dell/桌面/ccp-lpr_new_v3/data/tmp', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=1280, help='inference size (pixels)')  # 部署版本640*640
    parser.add_argument('--output', type=str, default='/home/dell/桌面/ccp-lpr_new_v3/runs', help='source')
    # parser.add_argument('--output', type=str, default='/media/dell/Elements/tyjt/s17/R12_M_CamE', help='source')
    # parser.add_argument('--output', type=str, default='/media/dell/Elements/tyjt/s17/R15_M_CamE', help='source')
    parser.add_argument('--kpt-label', type=int, default=4, help='number of keypoints')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    abs_path = opt.source+'-plate-license-res'  # 保存检测识别结果的文件夹

    """ 初始模型加载 """
    # 1.目标检测模型
    # plate_bbox_corner_model = attempt_load(opt.detect_model, map_location=device)  # 目标检测模型加载
    # plate_bbox_corner_model.to(device)
    # print(f"model:{plate_bbox_corner_model}")

    # 20230616  new 车牌检测模型
    plate_bbox_corner_model = attempt_load(opt.detect_model, map_location=device)
    plate_bbox_corner_model.to(device)
    # print(f"model:{plate_bbox_corner_model}")

    # 2.字符识别模型+颜色分类模型
    plate_rec_color_model = init_model(device, opt.rec_model)
    # print(f"plate_rec_color_model:{plate_rec_color_model}")
    plate_rec_color_model.to(device)

    # 3.PaddleOCR模型，字符识别
    ocr = PaddleOCR(rec_model_dir='weights/paddleocr_deploy',
                    rec_char_dict_path='weights/paddleocr_deploy/chinese_plate_dict.txt',
                    use_angle_cls=False,
                    use_gpu=False)  # use_gpu=False 不使用GPU 进行代码调试，调试结束后再设为True

    # define roI
    file_list = []
    allFilePath(opt.source, file_list)  # load all imgs in dir to file_list
    time_b = time.time()
    start_time = time.time()
    for pic_ in file_list:
        print(pic_, end=" ")  # 图片的绝对路径
        img = cv2.imread(pic_)  # 得到BGR图像
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # BGR

        # 进入检测函数进行计算
        # dict_list, box_res = detect(img, device, plate_bbox_corner_model, plate_rec_color_model, opt.img_size) # #用作保存检测bbox结果
        dict_list = detect(img, device, plate_bbox_corner_model, plate_rec_color_model, opt.img_size) # 部署版本
        print(f"dict_list:{dict_list}")

        # 保存检测结果到本地
        # img_res_save(abs_path, pic_)
        # img_box_res_save(abs_path, pic_, box_res)  # 用作保存检测bbox结果


        # 检测结果绘制在图像上并保存本地
        save_img_path = os.path.join(opt.output, os.path.basename(pic_))
        img = draw_result(img, dict_list)
        cv2.imwrite(save_img_path, img)
    print(f"耗时：{time.time() - start_time}")
    # print(f"time is {time.time()-time_b} s")

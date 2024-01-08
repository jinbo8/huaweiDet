import time
import os
import cv2
from PIL import Image


def create_dir_exist(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def splite_movie(path, FramesPath, ave_fre=1, stop_frame=36000):
    create_dir_exist(FramesPath)  # 创建空文件夹存储分帧图片

    vidcap = cv2.VideoCapture(path)
    count = 0
    while True:
        success, image = vidcap.read()
        img_path_name = os.path.join(FramesPath, str(str(f'{count}'.zfill(10))+'.jpg'))
        if count % ave_fre == 0:
            cv2.imwrite(img_path_name, image)
        count += 1
        print(f"img_frames：{count}, save_path:{FramesPath}")
        if count==stop_frame:  # 保存10min的视频结果，每两帧保存一帧。
            break
        else:
            continue


def make_video(mp4_save_Path, FramesPath, ImgPath, fps, merge_frames=36000):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    im = Image.open(ImgPath)
    vw = cv2.VideoWriter(mp4_save_Path, fourcc, fps, im.size)
    filenames = os.listdir(FramesPath)
    filenames.sort(key=lambda x:int(float(x.split('.')[0])))  # 按照图片名字顺序排序 必须是数字不能有字符串
    images_len = len(filenames)
    num = 0
    for i in filenames:
        num += 1
        frame = cv2.imread(FramesPath + '/' + i)
        print(f"merge_process:{num}/{images_len}")
        vw.write(frame)
        if num==merge_frames:  # 合并成视频的总帧数
            break


if __name__ == "__main__":

    # ***** 1.视频分帧为图像并保存 ******
    # path = '/media/dell/Elements/tyjt/ITS/r5/r5in/R5_Aw_CamN_in.mp4'  # 视频路径71
    # FramesPath = '/media/dell/Elements/tyjt/ITS/r5/r5in/src_img'  # 图片保存路径
    # path = '/media/dell/Elements/tyjt/ITS/r5/r5out/R5_Aw_CamN_out.mp4'  # 视频路径
    # FramesPath = '/media/dell/Elements/tyjt/ITS/r5/r5out/src_img'  # 图片保存路径
    # #
    #
    # start_time = time.time()
    # splite_movie(path, FramesPath, ave_fre=1, stop_frame=36000)
    # print(f"耗时：{time.time() - start_time}")
    #

    #***** 2.图像转视频*******
    # 第一帧图像名称
    ImgPath0 = '/media/dell/Elements/tyjt/ITS/r12/r12out/draw_mask_plate_res4/0000000000.jpg'
    # 分帧图像文件夹
    FramesPath = '/media/dell/Elements/tyjt/ITS/r12/r12out/draw_mask_plate_res4'
    # 生成的视频存放位置、名称
    mp4_save_Path = '/media/dell/Elements/tyjt/ITS/r12/r12out/R12_Aw_CamN_out_res.mp4'
    # 图片转视频
    make_video(mp4_save_Path, FramesPath, ImgPath0, 60, merge_frames=4200)

    # 使用 ffmep 进行压缩，压缩后大小为原来的1/4
    #  ffmpeg - i road_back_R15_M_CamE_2.mp4 -vcodec libx265 -crf  25 compress_video.mp4


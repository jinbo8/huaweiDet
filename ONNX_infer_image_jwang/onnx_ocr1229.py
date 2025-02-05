'''
测试转出的onnx模型
'''
import cv2
import numpy
import numpy as np

import torch
import onnxruntime as rt
import math
import os


class TestOnnx:
    def __init__(self, onnx_file, character_dict_path, use_space_char=True):
        self.sess = rt.InferenceSession(onnx_file)
        # 获取输入节点名称
        self.input_names = [input.name for input in self.sess.get_inputs()]
        # 获取输出节点名称
        self.output_names = [output.name for output in self.sess.get_outputs()]

        self.character = []
        self.character.append("blank")
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character.append(line)
        if use_space_char:
            self.character.append(" ")

    def resize_norm_img(self, img, image_shape=[3, 48, 168]):
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    # # 准备模型运行的feed_dict
    def process(self, input_names, image):
        feed_dict = dict()
        for input_name in input_names:
            feed_dict[input_name] = image

        return feed_dict

    def get_ignored_tokens(self):
        return [0]

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                                                                 batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[int(text_id)].replace('\n', '')
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def test(self, image_path):
        img_onnx = cv2.imread(image_path)
        # img_onnx = cv2.resize(img_onnx, (320, 32))
        # img_onnx = img_onnx.transpose((2, 0, 1)) / 255
        img_onnx = self.resize_norm_img(img_onnx)
        onnx_indata = img_onnx[np.newaxis, :, :, :]
        onnx_indata = torch.from_numpy(onnx_indata)
        # print('diff:', onnx_indata - input_data)
        # print('image shape: ', onnx_indata.shape)
        onnx_indata = np.array(onnx_indata, dtype=np.float32)
        feed_dict = self.process(self.input_names, onnx_indata)

        output_onnx = self.sess.run(self.output_names, feed_dict)
        # print('output_onnx[0].shape: ', output_onnx[0].shape)
        # print(' output_onnx[0]: ', output_onnx[0])

        output_onnx = numpy.asarray(output_onnx[0])

        preds_idx = output_onnx.argmax(axis=2)
        preds_prob = output_onnx.max(axis=2)
        post_result = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        print(f"post_result:{post_result}")
        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                    }
            print(image_path, rec_info)
        else:
            if len(post_result[0]) >= 2:
                # info = post_result[0][0] + "\t" + str(post_result[0][1])
                info = post_result[0][0]
                info_conf = post_result[0][1]
            print(image_path, info, info_conf)


if __name__ == '__main__':
    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5

    image_dir = "/home/dell/桌面/eval_sample/imageSample/DetOCRSample/LPSmallImg"
    onnx_file = '/home/dell/桌面/ccp-lpr_new_v3/weights/paddleocr_deploy/20231113_deploy/deploy_onnx/2023111300.onnx'
    character_dict_path = '/home/dell/桌面/ccp-lpr_new_v3/weights/paddleocr_deploy/20231113_deploy/deploy_onnx/chinese_plate_dict.txt'

    testobj = TestOnnx(onnx_file, character_dict_path)

    files = os.listdir(image_dir)
    for file in files:
        image_path = os.path.join(image_dir, file)
        result = testobj.test(image_path)
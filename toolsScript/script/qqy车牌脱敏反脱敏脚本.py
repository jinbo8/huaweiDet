import os
import shutil


def list2dict(cam_id):
    num = 9
    for i in cam_id:
        num += 1
        cam_id_dic[i] = 'a'+str(num)
    print(cam_id_dic)


def rename_files(rename_dir):
    names = os.listdir(rename_dir)
    tmp = 0
    total = len(names)
    for name in names:
        tmp += 1
        name_split = name.split('_')[:-1]
        cam_name = '_'.join(name_split)
        new_name = cam_id_dic[cam_name] + '_' + name.split('_')[-1]
        os.rename(os.path.join(rename_dir,name), os.path.join(rename_dir, new_name))
        print(f"rename process:{tmp}/{total}")




if __name__ == '__main__':

    # 标注图片脱敏
    # cam_id_dic = {'R3_Aw_CamN': 'a10', 'R3_Bn_CamE': 'a11', 'R3_Ds_CamW': 'a12', 'R4_Aw_CamN': 'a13',
    #               'R4_Bn_CamE': 'a14', 'R4_Ds_CamE': 'a15', 'R5_Aw_CamN': 'a16', 'R5_Bn_CamE': 'a17',
    #               'R5_Ds_CamW': 'a18', 'R6_Aw_CamN': 'a19', 'R6_Bn_CamE': 'a20', 'R6_Ds_CamW': 'a21',
    #               'R2_Aw_CamN': 'a22', 'R2_Bn_CamE': 'a23', 'R2_Ce_CamN': 'a24', 'R2_Ds_CamW': 'a25',
    #               'R9_Aw_CamN': 'a26', 'R9_Bn_CamE': 'a27', 'R9_Ce_CamN': 'a28', 'R9_Ds_CamE': 'a29',
    #               'R10_Aw_CamN': 'a30', 'R10_Bn_CamE': 'a31', 'R10_Ce_CamN': 'a32', 'R10_Ds_CamE': 'a33',
    #               'R11_Aw_CamN': 'a34', 'R11_Bs_CamW': 'a35', 'R11_Ce_CamN': 'a36', 'R11_Dn_CamW': 'a37',
    #               'R12_Aw_CamS': 'a38', 'R12_Bn_CamE': 'a39', 'R12_Ce_CamN': 'a40', 'R12_Ds_CamW': 'a41',
    #               'R16_Aw_CamS': 'a42', 'R16_Bn_CamE': 'a43', 'R16_Ce_CamS': 'a44', 'R16_Ds_CamE': 'a45',
    #               'R1_Bs_CamW': 'a46', 'R1_Bs_CamE': 'a47', 'R1_Ce_CamN': 'a48', 'R1_Dn_CamE': 'a49'}

    # rename_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/license20230823/license_plate'
    # rename_files(rename_dir)  # 图片rename

    # 1. 获取字典的key,value的反向映射
    # dic_id2cam = {}
    # for i, j in id2cam.items():
    #     dic_id2cam[j] = i
    # print(dic_id2cam)
    #
    # rename_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/license20230823/license_plate'
    # rename_files(rename_dir)  # 图片rename



    # 2. json标签重命名并合并到同一个文件夹
    # id2cam = {'a10': 'R3_Aw_CamN', 'a11': 'R3_Bn_CamE', 'a12': 'R3_Ds_CamW', 'a13': 'R4_Aw_CamN',
    #           'a14': 'R4_Bn_CamE', 'a15': 'R4_Ds_CamE', 'a16': 'R5_Aw_CamN', 'a17': 'R5_Bn_CamE',
    #           'a18': 'R5_Ds_CamW', 'a19': 'R6_Aw_CamN', 'a20': 'R6_Bn_CamE', 'a21': 'R6_Ds_CamW',
    #           'a22': 'R2_Aw_CamN', 'a23': 'R2_Bn_CamE', 'a24': 'R2_Ce_CamN', 'a25': 'R2_Ds_CamW',
    #           'a26': 'R9_Aw_CamN', 'a27': 'R9_Bn_CamE', 'a28': 'R9_Ce_CamN', 'a29': 'R9_Ds_CamE',
    #           'a30': 'R10_Aw_CamN', 'a31': 'R10_Bn_CamE', 'a32': 'R10_Ce_CamN', 'a33': 'R10_Ds_CamE',
    #           'a34': 'R11_Aw_CamN', 'a35': 'R11_Bs_CamW', 'a36': 'R11_Ce_CamN', 'a37': 'R11_Dn_CamW',
    #           'a38': 'R12_Aw_CamS', 'a39': 'R12_Bn_CamE', 'a40': 'R12_Ce_CamN', 'a41': 'R12_Ds_CamW',
    #           'a42': 'R16_Aw_CamS', 'a43': 'R16_Bn_CamE', 'a44': 'R16_Ce_CamS', 'a45': 'R16_Ds_CamE',
    #           'a46': 'R1_Bs_CamW', 'a47': 'R1_Bs_CamE', 'a48': 'R1_Ce_CamN', 'a49': 'R1_Dn_CamE'}

    # dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2023带预测结果的送标数据54792/TS_2D_车牌标注_20230921-1'
    # new_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2023带预测结果的送标数据54792/names_labels'
    # names = os.listdir(dir)
    # for i in names:
    #     json_dir = os.path.join(dir, i)
    #     for j in os.listdir(json_dir):
    #         json_abs_path = os.path.join(json_dir, j)
    #         shutil.move(json_abs_path, new_dir)

            # old_json_path = os.path.join(json_dir, j)
            # j_new = j.replace(i, id2cam[i])
            # new_label_name = os.path.join(new_dir, j_new)
            # shutil.copy(old_json_path, new_label_name)

    # 3. 获取标签图片
    old_images = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2023带预测结果的送标数据54792/license_plate_未脱敏原始图片'
    new_images = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2023带预测结果的送标数据54792/images'
    labels = '/media/dell/sata4t/jwang/datasets/items_datasets/plate_det/天翼交通标注数据集/2023带预测结果的送标数据54792/labels'
    lb_names= []
    for i in os.listdir(labels):
        lb_names.append(i[:-5])
    for j in os.listdir(old_images):
        if j[:-4] in lb_names:
            old_path = os.path.join(old_images, j)
            shutil.copy(old_path, os.path.join(new_images, j))
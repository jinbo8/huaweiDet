import os
from PIL import Image


def png2jpg(png_dir, jpg_dir):
    """ 遍历当前文件夹下的所有png文件并转换成jpg """
    n = 0
    total = len(os.listdir(png_dir))
    for filename in os.listdir(png_dir):
        n+=1
        file_img = os.path.join(png_dir, filename)
        img_png = Image.open(file_img)
        new_img = os.path.join(jpg_dir, filename.replace('.png', '.jpg'))
        os.remove(file_img)
        img_png.save(new_img)
        print(f"{n}/{total}")

if __name__ == '__main__':
    png_dir = '/media/dell/Elements/batch3_20880'
    jpg_dir = '/media/dell/sata4t/jwang/datasets/items_datasets/danger_drive/dangerous2022batch123/batch3/image3'
    png2jpg(png_dir, jpg_dir)
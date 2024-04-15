import glob
import os
import random

import numpy as np
from PIL import Image


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    # ratio = random.randint(1, 10) / 10
    ratio = 0.6
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img, amp_trg, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img  # .cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)


def extract_amp_spectrum(trg_img):
    """提取图像的振幅谱"""
    fft_trg_np = np.fft.fft2(trg_img, axes=(-2, -1))
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target


if __name__ == '__main__':
    # 定义本地图像文件夹和目标图像文件夹
    local_image_folder = "/home/temp58/dataset/biyanai/exp3/date1/FS/images"
    target_image_folder = "/home/temp58/dataset/biyanai/exp3/date3/FS/images"
    output_folder = "/home/temp58/dataset/biyanai/exp3/date1/FS/freq_images_06"

    # 获取本地图像文件列表
    local_image_files = glob.glob(os.path.join(local_image_folder, "*.jpg"))

    # 获取目标图像文件列表
    target_image_files = glob.glob(os.path.join(target_image_folder, "*.jpg"))

    im_local_list = []
    im_trg_list = []
    for local_image_file in local_image_files:
        # 读取本地图像
        im_local = Image.open(local_image_file).convert('RGB')
        im_local = im_local.resize((512, 512), Image.BICUBIC)
        im_local = np.asarray(im_local, np.float32)
        # im_local = im_local.transpose((2, 0, 1))
        im_local_list.append(im_local)

    for target_image_file in target_image_files:
        # 读取目标图像
        im_trg = Image.open(target_image_file).convert('RGB')
        im_trg = im_trg.resize((512, 512), Image.BICUBIC)
        im_trg = np.asarray(im_trg, np.float32)
        im_trg = im_trg.transpose((2, 0, 1))
        amp_tar = extract_amp_spectrum(im_trg)
        im_trg_list.append(amp_tar)

    # 执行插值和保存操作
    con = 0
    for im_local,im_trg in zip(im_local_list, im_trg_list):
        tar_img = source_to_target_freq(im_local, im_trg, L=0.0001)
        tar_img = np.clip(tar_img, 0, 255)
        generated_image = Image.fromarray(np.uint8(tar_img))
        con += 1
        # Define the output file path
        output_file = os.path.join(output_folder, f'{con}.jpg')
        print(con)
        # Save the generated image
        generated_image.save(output_file)
# coding=utf-8

import glob
import os
import imageio
import shutil
import numpy as np
from PIL import Image


def copy(a, b):
    with open(a, "r", encoding="utf-8") as fp:
        # for a1 in fp:
        #     print(a1)
        a = fp.readlines()
        with open(b, "w", encoding="utf-8") as fp1:
            fp1.writelines(a)


def SubPath(path_info, file_name, out_path, source_root_path, source_out_path, source_path):
    """

    Args:
        path_info: 'D:\\jpg_mask\\jpg_mask\\AH'
        file_name:
        out_path:
        source_root_path:
        source_out_path:
        source_path:

    Returns:

    """
    # print(file_name)

    dirNames = os.listdir(path_info)
    for dir_name in dirNames:
        if "ROI-3_Mask" == dir_name:
            file_name_info = file_name + "_" + "date3_jpg"
            source_file_path = source_path + "/" + "date3_jpg"
        # elif "ROI-2_Mask" == dir_name:
        #    file_name_info = file_name + "_" + "date2_jpg"
        #    source_file_path = source_path + "/" + "date2_jpg"
        # elif "ROI-3_Mask" == dir_name:
        #    file_name_info = file_name + "_" + "date3_jpg"
        #    source_file_path = source_path + "/" + "date3_jpg"
        elif dir_name.endswith("_Merge"):
            tmp_name = dir_name
            tmp_name = tmp_name.replace("_Merge", "")
            file_name_info = file_name + "_" + tmp_name
            source_file_path = source_path + "/" + tmp_name
        elif dir_name.endswith("_Mere"):
            tmp_name = dir_name
            tmp_name = tmp_name.replace("_Mere", "")
            file_name_info = file_name + "_" + tmp_name
            source_file_path = source_path + "/" + tmp_name
        else:
            file_name_info = file_name + "_" + dir_name
            source_file_path = source_path + "/" + dir_name
        # print(path_info)

        sub_path_info = path_info + "/" + dir_name

        if os.path.isfile(sub_path_info):
            # print(sub_path_info)
            out_path_target = out_path + "/" + file_name_info

            out_source_path_source = source_root_path + "/" + source_file_path
            out_source_path_target = source_out_path + "/" + file_name_info

            if os.path.exists(sub_path_info) and os.path.exists(out_source_path_source):
                print("**********************************")
                print(sub_path_info)
                mask = Image.open(sub_path_info)
                mask = mask.convert("L")
                mask = np.array(mask)
                if (np.sum(mask >= 128) < 5):
                    print("disable")
                    continue
                print(out_path_target)

                shutil.copyfile(sub_path_info, out_path_target)

                print(out_source_path_source)
                print(out_source_path_target)

                shutil.copyfile(out_source_path_source, out_source_path_target)
            else:
                print("------------------------------------------")
                print(sub_path_info)
                print(out_path_target)
                print(out_source_path_source)
                print(out_source_path_target)

            continue
            # aaaa

        SubPath(sub_path_info, file_name_info, out_path, source_root_path, source_out_path, source_file_path)


def SplitData(file_path, out_path, source_file_path, source_out_path):
    if os.path.exists(out_path) is False:
        os.mkdir(out_path)
    if os.path.exists(source_out_path) is False:
        os.mkdir(source_out_path)
    # os.path.join(cur_out_path, '{}.jpg'.format(n + 1))

    c = []
    dirNames = os.listdir(file_path)
    for dir_name in dirNames:
        path_info = os.path.join(file_path, dir_name)

        SubPath(path_info, dir_name, out_path, source_file_path, source_out_path, dir_name)

        # Dcm2jpg(input_path + "/" + dirName, out_path + "/" + dirName)


SplitData("/home/temp58/dataset/jpg_mask/AH",
          "/home/temp58/dataset/biyanai/exp3/date3/AH/masks",
          "/home/temp58/dataset/jpg_sort/AH",
          "/home/temp58/dataset/biyanai/exp3/date3/AH/images", )

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: voyager
# Date: 20190413,20190414,20190418,20190427
# Function: The program extracts pics from gnt files(CASIA HWDB)
#           and resize pics to 113*113 pixel,
#           save to trn, tst, val (8:1:1)

import os
import time
import random
import struct
import PIL.Image
import numpy as np

gnt_dir = "./HWDBgnt"
pic_dir = "./HWDBpic"
counter = 0


# read the pics from gnt files
def read_from_gnt_dir(gnt_dir):
    def one_file(f):
        # Refer to http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
        header_size = 10
        while True:
            header = np.fromfile(f, dtype = 'uint8', count = header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count = width*height).reshape((height, width))
            yield image, tagcode
 
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode, file_name


def resize_pic(img):
    img_arr = np.array(img)
    img_arr = 255 - img_arr

    x = img_arr.shape[0]
    y = img_arr.shape[1]
    if x < y:
        A = int((y-x)/2)
        img_tem = np.zeros((y, y))
        img_tem[A:A+x,:] = img_arr
    else:
        A = int((x-y)/2)
        img_tem = np.zeros((x, x))
        img_tem[:,A:A+y] = img_arr

    img_new = PIL.Image.fromarray(img_tem.astype('uint8'))
    img_new = img_new.resize((113, 113))
    return img_new


if __name__ == '__main__':
    start_time = time.time()

    for image, tagcode, file_name in read_from_gnt_dir(gnt_dir):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')

        img = PIL.Image.fromarray(image)
        img_new = resize_pic(img)

        # train:valid:test=8:1:1
        r_num = random.randint(1,100)
        if r_num < 81:
            newfilepath = pic_dir + '/trn/' + file_name[:-6] + tagcode_unicode + '.png'
        elif r_num > 90:
            newfilepath = pic_dir + '/tst/' + file_name[:-6] + tagcode_unicode + '.png'
        else:
            newfilepath = pic_dir + '/val/' + file_name[:-6] + tagcode_unicode + '.png'

        img_new.save(newfilepath)
        counter += 1
        if counter%10000 == 0:
            print("Processed",counter,"pics")

    end_time = time.time()
    # number of samples
    print("Program time comsuing:", end_time - start_time, "counter:", counter)


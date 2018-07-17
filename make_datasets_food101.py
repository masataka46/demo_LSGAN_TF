import numpy as np
from PIL import Image
import utility as Utility
import os
import random

class Make_datasets_food101():

    def __init__(self, dir_name, img_width, img_height, seed, mean, stddev, data_num, unit_num):
        self.dir_name = dir_name
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed

        self.file_list = self.get_file_names(self.dir_name)
        self.data_num = len(self.file_list)

        print("self.data_num", self.data_num)
        print("self.file_list[0], ", self.file_list[0])
        print("self.file_list[:10], ", self.file_list[:10])

        random.seed(self.seed)

        self.initial_noise = self.make_random_z_with_norm(mean, stddev, data_num, unit_num)

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files


    def read_data(self, filename_list, width, height):
        images = []
        for num, filename in enumerate(filename_list):

            pilIn = Image.open(filename)
            pilIn_size = pilIn.size #(width, height)

            if pilIn_size[0] >= pilIn_size[1]:
                margin_w = pilIn_size[0] - pilIn_size[1]
                margin_left = random.randint(0, margin_w)
                pil_crop = pilIn.crop((margin_left, 0, margin_left + pilIn_size[1], pilIn_size[1]))
                # print("pil_crop.size(case 1), ", pil_crop.size)
                pil_Resize = pil_crop.resize((width, height))
            else:
                margin_h = pilIn_size[1] - pilIn_size[0]
                margin_upper = random.randint(0, margin_h)
                pil_crop = pilIn.crop((0, margin_upper, pilIn_size[0], margin_upper + pilIn_size[0]))
                # print("pil_crop.size(case 2), ", pil_crop.size)
                pil_Resize = pil_crop.resize((width, height))

            image = np.asarray(pil_Resize, dtype=np.float32)
            images.append(image)

        return np.asarray(images)


    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0

        return data_norm


    def make_data_for_1_epoch(self):
        self.filename_1_epoch = random.sample(self.file_list, self.data_num)

        return len(self.filename_1_epoch)


    def get_data_for_1_batch(self, i, batchsize):
        filename_batch = self.filename_1_epoch[i:i + batchsize]
        images = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)

        return images_n


    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        return np.random.normal(mean, stddev, (data_num, unit_num))


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")

        return target


if __name__ == '__main__':
    #debug
    dir_name = '/media/webfarmer/HDCZ-UT/dataset/food101/food-101/images/hot_dog/'
    img_width = 64
    img_height = 64
    make_datasets_food101 = Make_datasets_food101(dir_name, img_width, img_height)
    num = make_datasets_food101.make_data_for_1_epoch()
    filename_1_epoch = make_datasets_food101.filename_1_epoch
    print("filename_1_epoch[:10], ", filename_1_epoch[:10])
    images_n = make_datasets_food101.get_data_for_1_batch(4, 3)

import os
import random
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle as rectangle
import matplotlib.pyplot as plt


class AnnotationRect:
    dic = dict()

    def __str__(self):
        return "x1: " + str(self.x1) \
               + ", y1: " + str(self.y1) \
               + ", x2: " + str(self.x2) \
               + ", y2: " + str(self.y2)

    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
        # print("TEMKENG")

    def get_width(self):
        return self.x2 - self.x1

    def get_height(self):
        return self.y2 - self.y1

    def get_area(self):
        return self.get_height() * self.get_width()

    def reader_ground_truth(self, filename):
        reader = open(filename, "r")
        annotation_list = list()
        for i in reader:
            d = i.split()
            annotation_list.append(AnnotationRect(*list(map(int, d[:4]))))
        return annotation_list

    def dico(self, image_names):
        for image in image_names:
            annotation_files = self.reader_ground_truth(image.replace("jpg", "gt_data.txt"))
            self.dic.update({image: annotation_files})

    def read_images(self):
        for r, d, f in os.walk("./dataset_mmp/train/"):
            for file in f:
                if ".jpg" in file:
                    image = os.path.join(r,file)
                    annotation_files = self.reader_ground_truth(image.replace("jpg", "gt_data.txt"))
                    self.dic.update({image: annotation_files})

    def select_and_draw(self):
        self.read_images()
        image = random.choice(list(self.dic.keys()))
        annotation_names = self.dic.get(image)
        image =Image.open(image)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.axis('off')
        for annotation_name in annotation_names:
            ax.add_patch(rectangle((annotation_name.x1, annotation_name.y1), annotation_name.get_width(),
                                   annotation_name.get_height(),linewidth=3,edgecolor='r',facecolor='none'))
        fig.savefig('TEMKENG_with_rectangle.png')
        plt.show()

    def anchor_grid(self, fmap_cols, fmap_rows, scale_factor=1.0, scale=[], aspect_ratios=[]):
        anchor = np.zeros((fmap_rows, fmap_cols, len(scale), len(aspect_ratios), 4))
        image_size = (scale_factor * fmap_rows, scale_factor * fmap_cols)
        # widht, heigth = image_size.size
        return image_size

    def area_interection(self, rect1, rect2):
        dx = min(rect1.x2, rect2.x2) - max(rect1.x1, rect2.x1)
        dy = min(rect1.y2, rect2.y2) - max(rect1.y1, rect2.y1)
        if (dx >= 0) and (dy >= 0):
            return dx * dy

    def area_union(self, rect1, rect2):
        return rect1.get_area() + rect2.get_area() - self.area_interection(rect1, rect2)

    def IoU(self, rect1, rect2):
        return self.area_interection(rect1, rect2) / self.area_union(rect1, rect2)

    def anchor_max_gt_overlaps(self,anchor_grid, gts=[]):
        print()


b = AnnotationRect(10, 15, 2, 3)
b.select_and_draw()
f = b.anchor_grid(1, 2, 8, )
d = np.zeros((2, 2, 1, 2, 4))
ds = [1, 5, 5, 5, 5, 5, 8, 5]


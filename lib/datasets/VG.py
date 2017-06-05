from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob

from collections import defaultdict
import random
import math
import json
import cv2
from matplotlib import pyplot as plt


class VG(object):
    def __init__(self, vg_path):
        self.vg_path = vg_path
        self.object_annotation_file = vg_path + '/annotations/' + 'objects.json'
        self.imageid_anno_dict = self._create_imageid_anno_dict()

    def loadCats(self):
        count = 0
        with open(self.object_annotation_file) as anno_file:
            data = json.load(anno_file)
            print('Yeah!')
            object_names = defaultdict(lambda: 0)
            for image_id in data:
                count += 1
                print(count)
                for object in image_id['objects']:
                    object_names[object['names'][0]] += 1
        return object_names.keys()

    def getImagesIds(self, category):
        with open(self.object_annotation_file) as anno_file:
            data = json.load(anno_file)
            print('Yeah!')
            image_list = []
            for image_id in data:
                for object in image_id['objects']:
                    if object['names'][0] == category:
                        image_list = image_list + [image_id["image_id"]]

    def getImagesIDs_from_category_list(self, categories_list):
        with open(self.object_annotation_file) as anno_file:
            data = json.load(anno_file)
            image_list = defaultdict(lambda: [])
            for image_id in data:
                for object in image_id['objects']:
                    if object['names'][0] in categories_list:
                        image_list[object['names'][0]] += [image_id["image_id"]]

        return image_list

    def train_val_splitting(self, percentage, categories_list):
        image_list = self.getImagesIDs_from_category_list(categories_list)
        train_cat = []
        val_cat = []
        for cat in categories_list:
            id = set(image_list[cat])
            num_train = math.floor(len(id) * percentage)
            train_cat = train_cat + random.sample(id, num_train)
            val_cat = val_cat + list((set(id) - set(train_cat)))
        train_val_intersect = list(set(train_cat).intersection(set(val_cat)))
        train_overlab = random.sample(train_val_intersect, math.floor(0.5 * len(train_val_intersect)))
        val_overlab = set(train_val_intersect) - set(train_overlab)
        train_list = list(set(train_cat) - set(train_val_intersect)) + list(train_overlab)
        val_list = list(set(val_cat) - set(train_val_intersect)) + list(set(val_overlab))
        return (train_list, val_list)

    def getImagesIds_from_folder(self,folder_path):
        file_list = glob.glob(folder_path + '/*.jpg')
        return [f.rsplit('/', 1)[1] for f in file_list]

    def plot(self, image_id, cats):
        with open(self.object_annotation_file) as anno_file:
            data = json.load(anno_file)
        im_file = self.vg_path + '/images/' + str(image_id) + '.jpg'
        im = cv2.imread(im_file)
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        objects = (x['objects'] for x in data if x['image_id'] == image_id)
        for obj in objects:
            for ob in obj:
                if ob['names'][0] in cats:
                    x = ob["x"]
                    y = ob["y"]
                    width = ob["w"]
                    height = ob["h"]
                    ax.add_patch(
                        plt.Rectangle((x, y),
                                      width,
                                      height, fill=False,
                                      edgecolor='red', linewidth=3.5)
                    )
    def _create_imageid_anno_dict(self):
        anno_image_dict = defaultdict(lambda: None)
        with open(self.object_annotation_file) as anno_file:
            data = json.load(anno_file)
            for im in data:
                anno_image_dict[str(im['image_id'])] = im
        return anno_image_dict

    def loadImgsAnno(self,index):
        im = self.imageid_anno_dict[index]
        return im

    def getImageWidthHeights(self,index):
        im_file = self.vg_path + '/images/' + str(index) + '.jpg'
        im = cv2.imread(im_file)
        h, w, c = im.shape
        return h, w


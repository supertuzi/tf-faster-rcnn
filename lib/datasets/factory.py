# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.VG import VG
from datasets.visual_genome import visual_genome

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

#Set up visual_genome_2016<split>
vg_path = './data/visual_genome'
vg_instance = VG(vg_path)
categories_lists = ['categories_1']
categories_1 = ['sunglasses', 'pants', 'jeans', 'shirt', 'tie', 'suit', 'shoes', 'skirt', 'jacket', 'dress', 'coat', 'shorts']
for categories in categories_lists:
 for split in ['train','val']:
      train_dir = './data/visual_genome/' + categories + '_train'
      val_dir = './data/visual_genome/' + categories + '_val'

      train_list, val_list = vg_instance.train_val_splitting(0.7, eval(categories))
      if not os.path.exists(train_dir):
            os.system('mkdir ' + train_dir)
            for f in train_list:
                os.system('cp ' + vg_path + '/images/' + str(f) + '.jpg ' + train_dir)
      if not os.path.exists(val_dir):
            os.system('mkdir ' + val_dir)
            for f in val_list:
                os.system('cp ' + vg_path + '/images/' + str(f) + '.jpg ' + val_dir)
      name = 'visual_genome_{}_{}'.format(categories,split)
      __sets[name] = (lambda split=split, categories=categories: visual_genome(categories, eval(categories), split))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())

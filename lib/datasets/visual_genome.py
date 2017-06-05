# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

from datasets.VG import VG
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.config import cfg
from .vg_eval import vg_eval
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
from matplotlib import pyplot as plt
import cv2

class visual_genome(imdb):
  def __init__(self, categories_name, categories, split):
    imdb.__init__(self, 'visual_genome_' + categories_name + '_' + split)
    # visual_genome specific config options
    self.config = {'use_salt': True,
                   'cleanup': True}
    # name, paths
    self._data_path = osp.join(cfg.DATA_DIR, 'visual_genome')
    self._data_name = categories_name + '_' + split
    self.categories_name = categories_name
    # load VG API, class <-> id mappings
    self._VG= VG(self._get_ann_file())
    self.cats = categories
    self._classes = tuple(['__background__'] + [c for c in self.cats])
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_index = self._load_image_set_index()
    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val')
    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

  def _get_ann_file(self):

    return osp.join(self._data_path)

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_files = self._VG.getImagesIds_from_folder(self._data_path + '/' + self._data_name)
    image_ids = [x.split('.jpg')[0] for x in image_files]
    return image_ids

  def _get_widths(self):
    widths = []
    for id in self._image_index:
      im = cv2.imread(self.image_path_at(id))
      h, w, c = im.shape
      widths = widths + [w]
    return widths

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = (str(index) + '.jpg')
    image_path = osp.join(self._data_path, self._data_name, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_vg_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_vg_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    t=time.time()
    object_anno = self._VG.loadImgsAnno(index)
    #print(time.time()-t)
    height, width = self._VG.getImageWidthHeights(index)
    #print(time.time()-t)
    objs = object_anno['objects']
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for obj in objs:
      if obj['names'][0] in self.cats:
          x1 = np.max((0, obj['x']))
          y1 = np.max((0, obj['y']))
          x2 = np.min((width - 1, x1 + np.max((0, obj['w'] - 1))))
          y2 = np.min((height - 1, y1 + np.max((0, obj['h'] - 1))))
          obj['area'] = obj['w'] * obj['h']
          if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
            obj['clean_bbox'] = [x1, y1, x2, y2]
            valid_objs.append(obj)
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    for ix, obj in enumerate(objs):
      cls = self._class_to_ind[obj["names"][0]]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    #print('processing time:{}'.format(time.time()-t))
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _get_box_file(self, index):
    # first 14 chars / first 22 chars / all chars + .mat
    # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
    file_name = ('COCO_' + self._data_name +
                 '_' + str(index).zfill(12) + '.mat')
    return osp.join(file_name[:14], file_name[:22], file_name)

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

  def _do_detection_eval(self,output_dir):
      annopath = os.path.join(
        self._data_path, 'annotations', 'objects.json')
      imagesetfile = os.path.join(
        self._data_path, self._data_name)

      cachedir = os.path.join(self._data_path, 'annotations_cache')
      aps = []
      if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
      for i, cls in enumerate(self._classes):
        if cls == '__background__':
          continue
        filename = self._get_vg_results_file_template().format(cls)
        rec, prec, ap = vg_eval(
          filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
        aps += [ap]
        print(('AP for {} = {:.4f}'.format(cls, ap)))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
          pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
      print(('Mean AP = {:.4f}'.format(np.mean(aps))))
      print('~~~~~~~~')
      print('Results:')
      for ap in aps:
        print(('{:.3f}'.format(ap)))
      print(('{:.3f}'.format(np.mean(aps))))
      print('~~~~~~~~')
      print('')
      print('--------------------------------------------------------------')

  def _vg_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      results.extend(
        [{'image_id': index,
          'category_id': cat_id,
          'bbox': [xs[k], ys[k], ws[k], hs[k]],
          'score': scores[k]} for k in range(dets.shape[0])])
    return results

  def _get_vg_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 'det_' + self.categories_name + '_{:s}.txt'
    dir = os.path.join(self._data_path,
      'results',
      'VG' , self.categories_name,
      'Main')
    if not os.path.exists(dir):
      os.mkdirs(dir)
    path = os.path.join(
      self._data_path,
      'results',
      'VG', self.categories_name,
      'Main',
      filename)
    return path

  def _write_vg_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_vg_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def evaluate_detections(self, all_boxes, output_dir):
    res_file = osp.join(output_dir, ('detections_' +
                                     self.categories_name +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.txt'
    with open(res_file, 'wb') as f:
        pickle.dump(all_boxes, f)
    self._write_vg_results_file(all_boxes)
    # Only do evaluation on non-test sets
    self._do_detection_eval(output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True




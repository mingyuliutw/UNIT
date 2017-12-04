#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
mypath="img_align_celeba"
myoutpath="img_align_crop_resize_celeba"
os.mkdir(myoutpath)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
  infile=join(mypath,f)
  outfile=join(myoutpath,f)
  img=cv2.imread(infile)
  crop_img=img[20:218-20,:,:]
  resized_img=cv2.resize(crop_img,dsize=(132,132))
  cv2.imwrite(outfile,resized_img)



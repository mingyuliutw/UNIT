"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import torch

# Get model list for resume
def get_model_list(dirname, key):
  if os.path.exists(dirname) is False:
    return None
  gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                os.path.isfile(os.path.join(dirname, f)) and key in f and "pkl" in f]
  if gen_models is None:
    return None
  gen_models.sort()
  last_model_name = gen_models[-1]
  return last_model_name

def _compute_true_acc(predictions):
  predictions = torch.ge(predictions.data, 0.5)
  if len(predictions.size()) == 3:
    predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
  acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
  return acc

def _compute_fake_acc(predictions):
  predictions = torch.le(predictions.data, 0.5)
  if len(predictions.size()) == 3:
    predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
  acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
  return acc

def _compute_true_acc2(predictions):
  _, true_predicts = torch.max(predictions.data, 1)
  acc = (true_predicts == 1).sum() / (1.0 * true_predicts.size(0))
  return acc

def _compute_fake_acc2(predictions):
  _, true_predicts = torch.max(predictions.data, 1)
  acc = (true_predicts == 0).sum() / (1.0 * true_predicts.size(0))
  return acc

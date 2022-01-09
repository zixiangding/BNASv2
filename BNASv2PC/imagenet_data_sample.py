import numpy as np
import os
import shutil
import sys, time

class ShowProcess():
  i = 0
  max_steps = 0
  max_arrow = 50
  infoDone = 'done'
  def __init__(self, max_steps, infoDone = 'Done'):
      self.max_steps = max_steps
      self.i = 0
      self.infoDone = infoDone
  def show_process(self, i=None):
      if i is not None:
          self.i = i
      else:
          self.i += 1
      num_arrow = int(self.i * self.max_arrow / self.max_steps)
      num_line = self.max_arrow - num_arrow
      percent = self.i * 100.0 / self.max_steps
      process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                    + '%.2f' % percent + '%' + '\r'
      sys.stdout.write(process_bar)
      sys.stdout.flush()
      if self.i >= self.max_steps:
          self.close()
  def close(self):
      print('')
      print(self.infoDone)
      self.i = 0

data_dir = '/home/user/datasets/imagenet/imagenet'
data_sub_dir = './imagenet_search'

assert data_dir or data_sub_dir is not None, 'data dir must be given!'

train_all_dir = os.path.join(data_dir, 'train')
train_label_dir = os.path.join(data_dir, 'train_label.txt')
val_label_dir = os.path.join(data_dir, 'validation_label.txt')

if not os.path.exists(data_sub_dir):
  os.mkdir(data_sub_dir)

train_sub_dir = os.path.join(data_sub_dir, 'train')

if not os.path.exists(train_sub_dir):
  os.mkdir(train_sub_dir)

val_sub_dir = os.path.join(data_sub_dir, 'val')

if not os.path.exists(val_sub_dir):
  os.mkdir(val_sub_dir)

train_label_sub_dir = os.path.join(data_sub_dir, 'train_label.txt')
val_label_sub_dir = os.path.join(data_sub_dir, 'validation_label.txt')

if not os.path.exists(train_label_sub_dir):
  shutil.copy(train_label_dir, train_label_sub_dir)

if not os.path.exists(val_label_sub_dir):  
  shutil.copy(val_label_dir, val_label_sub_dir)

train_class_all = os.listdir(train_all_dir)

max_steps_cls = len(train_class_all)
process_bar_cls = ShowProcess(max_steps_cls, 'OK')

for train_cls in train_class_all:
  process_bar_cls.show_process()
  time.sleep(0.0001)
  #random smaple 12.5% train data
  train_cls_dir = os.path.join(train_all_dir, train_cls)
  train_cls_dir_target = os.path.join(train_sub_dir, train_cls)

  if not os.path.exists(train_cls_dir_target):
    os.mkdir(train_cls_dir_target)

  val_cls_dir_target = os.path.join(val_sub_dir, train_cls)

  if not os.path.exists(val_cls_dir_target):
    os.mkdir(val_cls_dir_target)

  cls_files_all = os.listdir(train_cls_dir)
  num_files = len(cls_files_all)
  sub_index = np.random.randint(0, num_files, round(num_files * 0.125))
  partiton_index = round(len(sub_index)/5)
  val_indices = sub_index[:partiton_index]
  train_indices = sub_index[partiton_index:]

  for idx in train_indices:
    file_name = cls_files_all[idx]
    file_dir = os.path.join(train_cls_dir, file_name)
    file_dir_target = os.path.join(train_cls_dir_target, file_name)
    shutil.copy(file_dir, file_dir_target)

  for idx in val_indices:
    file_name = cls_files_all[idx]
    file_dir = os.path.join(train_cls_dir, file_name)
    file_dir_target = os.path.join(val_cls_dir_target, file_name)
    shutil.copy(file_dir, file_dir_target)

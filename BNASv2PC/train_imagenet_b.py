import os
import sys
import numpy as np
import time
import torch
import utils
import shutil
import glob
import random
import warnings
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import genotypes

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from torch.autograd import Variable
from model_b import NetworkImageNet_b2 as Network
#from thop import profile

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default=None, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=44, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--en_layers', type=int, default=2, help='total number of en layers')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default=None, help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default=None, help='')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=35, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
parser.add_argument('--tencrop', action='store_true', default=False, help='use 10-crop testing')
parser.add_argument('--num_readers', type=int, default=16, help='total number of layers')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler, linear or cosine')
parser.add_argument('--reset_save_dir', action='store_true', default=False, help='')
parser.add_argument('--best', action='store_true', default=False, help='')
args = parser.parse_args()

warnings.filterwarnings('ignore')
args.save = args.arch
args.resume = args.save

class DataLoaderX(DataLoader):
  def __iter__(self):
    return BackgroundGenerator(super().__iter__())

if os.path.isdir(args.save):
  if args.reset_save_dir:
    print("Path {} exists. Remove it.".format(args.save))
    shutil.rmtree(args.save)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
else:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  best_acc_top1 = 0
  best_acc_top5 = 0

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, False, genotype, en_layers=args.en_layers, drop_prob=0)
  #model = resnet50(pretrained=False)
  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()

  # flops, _ = profile(model, (1, 3, 224, 224))
  #
  # print("FLOPS of {} is {} G.".format(args.arch, flops/(1e9)))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  
  if args.best:
    dir_model = 'model_best.pth.tar'
  else:
    dir_model = 'checkpoint.pth.tar'

  if args.resume:
    args.resume = os.path.join(args.resume, dir_model)
    if os.path.isfile(args.resume):
      print("==============> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc_top1 = checkpoint['best_acc_top1']
      best_acc_top5 = checkpoint['best_acc_top5']
      model.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()
  
  # miles = [100, 180, 240, 250]
  #
  # if args.start_epoch < miles[0]:
  #   args.learning_rate *= 1
  #   milestones = [miles[0] - args.start_epoch, miles[1] - args.start_epoch, miles[2] - args.start_epoch]
  # elif args.start_epoch < miles[1]:
  #   args.learning_rate *= 0.1
  #   milestones = [miles[1] - args.start_epoch, miles[2] - args.start_epoch]
  # elif args.start_epoch < miles[2]:
  #   args.learning_rate *= 0.01
  #   milestones = [miles[2] - args.start_epoch]
  # else:
  #   args.learning_rate *= 0.001
  #   milestones = [miles[miles[3]] - args.start_epoch]
 
  def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  crops = transforms.TenCrop(224)
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  if args.tencrop:
    valid_data = dset.ImageFolder(
      validdir,
      transforms.Compose([
        transforms.Resize(256),
        crops,
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
      ]))
  else:
    valid_data = dset.ImageFolder(
      validdir,
      transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
      ]))

  # train_queue = torch.utils.data.DataLoader(
  #   train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_readers)
  # valid_queue = torch.utils.data.DataLoader(
  #     valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_readers)

  train_queue = DataLoaderX(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_readers)
  valid_queue = DataLoaderX(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_readers)

  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
  #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  lr = args.learning_rate
  for epoch in range(args.start_epoch, args.epochs):
    if args.lr_scheduler == 'cosine':
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
    elif args.lr_scheduler == 'linear':
        current_lr = adjust_lr(optimizer, epoch)
    else:
        print('Wrong lr type, exit')
        sys.exit(1)
    logging.info('Epoch: %d lr %e', epoch, current_lr)
    if epoch < 5 and args.batch_size > 256:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (epoch + 1) / 5.0
        logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)

  #for epoch in range(args.start_epoch, args.epochs):
    scheduler.step()
      #logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    logging.info('epoch %d drop path prob %e', epoch, model.drop_path_prob)

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)
    logging.info('valid_loss %f', valid_obj)
    # scheduler.step(train_obj)
    # logging.info('epoch %d', epoch)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      best_acc_top5 = valid_acc_top5
      is_best = True
      logging.info('Save the best model at epoch: %d, top1: %f, top5: %f', epoch, valid_acc_top1, valid_acc_top5)
    elif valid_acc_top1 == best_acc_top1:
      if valid_acc_top5 > best_acc_top5:
        best_acc_top5 = valid_acc_top5
        is_best = True
        logging.info('Save the best model at epoch: %d, top1: %f, top5: %f', epoch, valid_acc_top1, valid_acc_top5)
    else:
      pass

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'best_acc_top5': best_acc_top5,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    target = target.cuda(async=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    logits, _ = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  log_freq = args.report_freq

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    if args.tencrop:
      batch_size, num_crops, c, h, w = input.size()
      logits,_ = model(input.view(-1, c, h, w))
      logits = logits.view(batch_size, num_crops, -1).mean(1)
      loss = criterion(logits, target)
    else:
      logits, _ = model(input)
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % log_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 


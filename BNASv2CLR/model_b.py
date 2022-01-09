import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, enhancement=False):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if enhancement:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, genotype, set, en_layers=3, pool_time=2):
    super(Network, self).__init__()
    self._layers = layers
    self.C_list_gap = []
    self.C_list_en = []
    self.pool_distances = self._layers // pool_time
    self.pool_layers = []
    for pool in range(pool_time):
      self.pool_layers.append((pool+1)*self.pool_distances-1)

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    if set in ['cifar10', 'cifar100', 'svhn', 'norb']:
      C_in = 3
    elif set in ['mnist', 'fashion']:
      C_in = 1
    else:
      raise ValueError('Unknown dataset type!')

    self.stem = nn.Sequential(
      nn.Conv2d(C_in, C_curr, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.conv_cells = nn.ModuleList()  # conv cells and deep cells
    self.en_cells = nn.ModuleList()  # enhancement cells
    self.reduction_convs_gap = nn.ModuleList()  # factorreduce of conv used for GAP
    self.reduction_convs_en = nn.ModuleList()  # factorreduce of conv used for En
    self.restricted_convs_gap = nn.ModuleList()  # 1 conv knowledge between conv and gap
    self.restricted_convs_en = nn.ModuleList()  # 1 conv knowledge between conv and enhancement cell
    self.restricted_ens_gap = nn.ModuleList()  # 1 conv knowledge between conv and gap
    reduction_prev = False

    for i in range(layers):
      if i in self.pool_layers:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      cell = Cell(genotype, C_prev_prev, C_prev, C_curr,
                  reduction, reduction_prev, enhancement=False)

      # store the input of broad cells for gap layer
      if i + 1 in self.pool_layers:
        C_in = cell.multiplier * C_curr
        C_out = C_curr
        self.C_list_gap.append(C_out)
        restricted_conv_gap = ReLUConvBN(C_in, C_out, 1, 1, 0)
        self.restricted_convs_gap += [restricted_conv_gap]
        if len(self.restricted_convs_gap) > 1:  # for GAP
          for reduce_gap in range(len(self.restricted_convs_gap) - 1):
            reduction_conv_gap = FactorizedReduce(self.C_list_gap[reduce_gap], self.C_list_gap[reduce_gap])
            self.reduction_convs_gap.append(reduction_conv_gap)

      reduction_prev = reduction
      self.conv_cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

      # store the output of broad cells for enhancement block
      if i in self.pool_layers:
        if i == self.pool_layers[-1]:
          C_out_conv = C_curr * cell.multiplier
        else:
          C_out_conv = C_curr
        self.C_list_en.append(C_out_conv)
        restricted_conv_en = ReLUConvBN(C_prev, C_out_conv, 1, 1, 0)
        self.restricted_convs_en += [restricted_conv_en]
        if len(self.restricted_convs_en) > 1:  # for GAP
          for reduce_en in range(len(self.restricted_convs_en) - 1):
            reduction_conv_en = FactorizedReduce(self.C_list_en[reduce_en], self.C_list_en[reduce_en])
            self.reduction_convs_en.append(reduction_conv_en)

      if i + 1 == layers:
        for reduce_gap in range(len(self.restricted_convs_gap)):
          reduction_conv_gap = FactorizedReduce(self.C_list_gap[reduce_gap], self.C_list_gap[reduce_gap])
          self.reduction_convs_gap.append(reduction_conv_gap)

    for j in range(en_layers):  # enhancement block
      reduction = False
      reduction_prev = False
      if j == 0:
        en_cell = Cell(genotype, self.C_list_en[0], C_prev, C_curr,
                  reduction, reduction_prev, enhancement=True)
      else:
        en_cell = Cell(genotype, C_prev_prev, C_prev, C_curr,
                  reduction, reduction_prev, enhancement=True)
      self.en_cells += [en_cell]
      C_prev_prev, C_prev = C_prev, en_cell.multiplier * C_curr

      # store the output of enhancement cells for GAP layer
      if j + 1 == en_layers:
        C_out = C_prev
      else:
        C_out = C_curr
      restricted_en_gap = ReLUConvBN(C_prev, C_out, 1, 1, 0)
      self.restricted_ens_gap += [restricted_en_gap]

    C_final = sum(self.C_list_gap) + (en_layers - 1) * C_curr + C_prev

    #print(len(self.conv_cells)) # conv cells and deep cells
    #print(len(self.en_cells)) # enhancement cells
    # print(self.reduction_convs_gap)  # factorreduce of conv used for GAP
    # print(self.reduction_convs_en)  # factorreduce of conv used for En
    # print(self.restricted_convs_gap)  # 1 conv knowledge between conv and gap
    # print(self.restricted_convs_en)  # 1 conv knowledge between conv and enhancement cell
    # print(self.restricted_ens_gap)  # 1 conv knowledge between conv and gap

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_final, num_classes)

  def forward(self, input):
    conv_inputs = []
    en_inputs = []
    en_outputs = []
    reduce_time_conv = 0
    reduce_time_en = 0
    pool_time = 0
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.conv_cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if (i+1) in self.pool_layers: # store the input of broad cells for gap layer
        conv_input = self.restricted_convs_gap[pool_time](s1)
        conv_inputs.append(conv_input)
        if len(conv_inputs) > 1:
          for k in range(len(conv_inputs) - 1):
            conv_inputs[k] = self.reduction_convs_gap[reduce_time_conv](conv_inputs[k])
            reduce_time_conv += 1

      if i in self.pool_layers:
        # store the output of broad cells for enhancement block
        en_input = self.restricted_convs_en[pool_time](s1)
        en_inputs.append(en_input)
        if len(en_inputs) > 1:
          for j in range(len(en_inputs) - 1):
            en_inputs[j] = self.reduction_convs_en[reduce_time_en](en_inputs[j])
            reduce_time_en += 1
        if i == self.pool_layers[-1]:
          for m in range(len(conv_inputs)):
            conv_inputs[m] = self.reduction_convs_gap[reduce_time_conv](conv_inputs[m])
            reduce_time_conv += 1
        pool_time += 1

    for i, cell in enumerate(self.en_cells):  # enhancement blocks
      if i == 0:
        s0, s1 = s1, cell(en_inputs[0], en_inputs[1], self.drop_path_prob)
      else:
        s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      en_output = self.restricted_ens_gap[i](s1)
      en_outputs.append(en_output)

    final_layer = []
    for out in conv_inputs:
      final_layer.append(out)
    for out in en_outputs:
      final_layer.append(out)
    final_out = torch.cat(final_layer, dim=1)
    out = self.global_pooling(final_out)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux


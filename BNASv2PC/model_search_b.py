import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    self.k = 4
    for primitive in PRIMITIVES:
      op = OPS[primitive](C //self.k, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
      self._ops.append(op)


  def forward(self, x, weights):
    #channel proportion k=4  
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)
    #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C,
               reduction, reduction_prev, enhancement=False):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.enhancement = enhancement

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights,weights2):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, en_layers=1, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self.en_layers = en_layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.C_gap_list = []
    self.C_en_list = []

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
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
      #store the input of broad cells for gap layer
      if i == 0:
        C_out = self._C
        C_in = stem_multiplier * self._C
      else:
        C_out = C_curr
        C_in = multiplier * C_curr
      restricted_conv_gap = ReLUConvBN(C_in, C_out, 1, 1, 0)
      self.restricted_convs_gap += [restricted_conv_gap]
      self.C_gap_list.append(C_out)

      if len(self.restricted_convs_gap) > 1:  # for GAP
        for reduce_time in range(len(self.restricted_convs_gap) - 1):
          reduction_conv_gap = FactorizedReduce(self.C_gap_list[reduce_time], self.C_gap_list[reduce_time])
          self.reduction_convs_gap.append(reduction_conv_gap)

      # convolution block
      C_curr *= 2
      reduction = True
      conv_cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                       reduction, reduction_prev, enhancement=False)
      reduction_prev = reduction
      self.conv_cells += [conv_cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr

      # store the output of broad cells for enhancement block
      if i + 1 == layers:
        C_out_conv = C_curr * multiplier
      else:
        C_out_conv = C_curr
      restricted_conv_en = ReLUConvBN(C_prev, C_out_conv, 1, 1, 0)
      self.restricted_convs_en += [restricted_conv_en]
      self.C_en_list.append(C_out_conv)
      if len(self.restricted_convs_en) > 1:  # for GAP
        for reduce_time in range(len(self.restricted_convs_en) - 1):
          reduction_conv_en = FactorizedReduce(self.C_en_list[reduce_time], self.C_en_list[reduce_time])
          self.reduction_convs_en.append(reduction_conv_en)

      if i + 1 == layers:
        for reduce_time in range(len(self.restricted_convs_gap)):
          reduction_conv_gap = FactorizedReduce(self.C_gap_list[reduce_time], self.C_gap_list[reduce_time])
          self.reduction_convs_gap.append(reduction_conv_gap)
    reduction = False
    for j in range(en_layers):  # enhancement block
      # if j == 0:
      #   reduction_prev = True
      # else:
      reduction_prev = False

      if j == 0:
        en_cell = Cell(steps, multiplier, self.C_en_list[0], self.C_en_list[1], C_curr, reduction, reduction_prev, enhancement=True)
      else:
        en_cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, enhancement=True)
      self.en_cells += [en_cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr

      # store the output of enhancement cells for GAP layer
      if j + 1 == en_layers:
        C_out = C_prev
      else:
        C_out = C_curr
      restricted_en_gap = ReLUConvBN(C_prev, C_out, 1, 1, 0)
      self.restricted_ens_gap += [restricted_en_gap]

    C_final = sum(self.C_gap_list) + (en_layers-1)*C_curr + C_prev 

    # print(self.conv_cells) # conv cells and deep cells
    #print(self.en_cells) # enhancement cells
    # print(self.reduction_convs_gap)  # factorreduce of conv used for GAP
    # print(self.reduction_convs_en)  # factorreduce of conv used for En
    # print(self.restricted_convs_gap)  # 1 conv knowledge between conv and gap
    # print(self.restricted_convs_en)  # 1 conv knowledge between conv and enhancement cell
    # print(self.restricted_ens_gap)  # 1 conv knowledge between conv and gap

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_final, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, en_layers=self.en_layers).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    conv_inputs = []
    en_inputs = []
    en_outputs = []
    reduce_time_conv = 0
    reduce_time_en = 0
    pool_time = 0
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.conv_cells):
      # store the input of broad cells for gap layer
      conv_input = self.restricted_convs_gap[pool_time](s1)
      conv_inputs.append(conv_input)

      if len(conv_inputs) > 1:
        for k in range(len(conv_inputs) - 1):
          conv_inputs[k] = self.reduction_convs_gap[reduce_time_conv](conv_inputs[k])
          reduce_time_conv += 1
      # process all elements in conv_inputs again for GAP layer
      if i + 1 == self._layers:
        for m in range(len(conv_inputs)):
          conv_inputs[m] = self.reduction_convs_gap[reduce_time_conv](conv_inputs[m])
          reduce_time_conv += 1

      if cell.enhancement:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for _ in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for _ in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights, weights2)

      # store the output of broad cells for enhancement block
      en_input = self.restricted_convs_en[pool_time](s1)
      en_inputs.append(en_input)
      if len(en_inputs) > 1:
        for j in range(len(en_inputs) - 1):
          en_inputs[j] = self.reduction_convs_en[reduce_time_en](en_inputs[j])
          reduce_time_en += 1
      # for i in range(len(en_inputs)):
      #   print(en_inputs[i].size())
      pool_time += 1

    for j, cell in enumerate(self.en_cells):  # enhancement blocks
      if cell.enhancement:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for _ in range(self._steps - 1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2, tw2], dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for _ in range(self._steps - 1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2, tw2], dim=0)

      if j == 0:
        s0, s1 = s1, cell(en_inputs[0], en_inputs[1], weights, weights2)
      else:
        s0, s1 = s1, cell(s0, s1, weights, weights2)

      en_output = self.restricted_ens_gap[j](s1)
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

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


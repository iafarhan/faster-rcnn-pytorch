import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from tools import *
import matplotlib.pyplot as plt
from single_stage_detector import GenerateAnchor, GenerateProposal, IoU


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")

class ProposalModule(nn.Module):
  def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
    super().__init__()

    assert(num_anchors != 0)
    self.num_anchors = num_anchors

    self.pred_layer = None      
    # 6 = 4 bb offsets + 2 class scores
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_channels=in_dim,out_channels=hidden_dim,kernel_size=3,padding=1),
      nn.Dropout(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=hidden_dim,out_channels=self.num_anchors*6,kernel_size=1)
    )


  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Extracted_anchors: giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors

  def forward(self, features, pos_anchor_coord=None, \
              pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the proposal module.
    """
    if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
      mode = 'eval'
    else:
      mode = 'train'
    B, _, H, W = features.shape
    conf_scores, offsets, proposals = None, None, None

    activations = self.pred_layer(features)
    anchors_data = activations.clone().reshape(B,self.num_anchors,6,H,W) 
    offsets = anchors_data[:,:,2:].clone()
    conf_scores = anchors_data[:,:,:2].clone()
    if mode == 'train':
      offsets = offsets.permute(0,1,3,4,2).reshape(-1,4)
      offsets = offsets[pos_anchor_idx] 
      conf_scores = conf_scores.permute(0,1,3,4,2).reshape(-1,2)
      idxs = torch.cat([pos_anchor_idx,neg_anchor_idx],dim=0)
      conf_scores = conf_scores[idxs]
      proposals = GenerateProposal(pos_anchor_coord,offsets,method='FasterRCNN')
    
    if mode == 'train':
      return conf_scores, offsets, proposals
    elif mode == 'eval':
      return conf_scores, offsets


def ConfScoreRegression(conf_scores, batch_size):
  """
  Binary cross-entropy loss

  """
  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1. #positive anchors'0ind(object) =1
  GT_conf_scores[M:, 1] = 1. #neg anchors' 1ind(background)=1

  conf_score_loss = nn.functional.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum') * 1. / batch_size
  return conf_score_loss


def BboxRegression(offsets, GT_offsets, batch_size):
  """"
  Use SmoothL1 loss as in Faster R-CNN
  """
  bbox_reg_loss = nn.functional.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
  return bbox_reg_loss



class RPN(nn.Module):
  def __init__(self):
    super().__init__()

    # READ ONLY
    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
    self.feat_extractor = FeatureExtractor()
    self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0])

  def forward(self, images, bboxes, output_mode='loss'):
    """
    Training-time forward pass for the Region Proposal Network.

    """
    # weights to multiply to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 5 # for offsets

    assert output_mode in ('loss', 'all'), 'invalid output mode!'
    total_loss = None
    conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = \
      None, None, None, None, None, None

    image_feats = self.feat_extractor(images)
    B = images.shape[0]
    self.anchor_list=self.anchor_list.to(image_feats.device)
    grid_list = GenerateGrid(B)
    anc_list = GenerateAnchor(self.anchor_list,grid_list)
    iou_mat = IoU(anc_list,bboxes)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, \
      GT_offsets, GT_class, \
        activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anc_list,bboxes,grid_list,iou_mat,neg_thresh=0.2)
    
    conf_scores, offsets, proposals = self.prop_module(image_feats,activated_anc_coord,activated_anc_ind,negative_anc_ind)
    
    conf_loss = ConfScoreRegression(conf_scores, B)
    reg_loss = BboxRegression(offsets, GT_offsets,B)
    
    total_loss = w_conf * conf_loss + w_reg * reg_loss
    anc_per_img = torch.prod(torch.tensor(anc_list.shape[1:-1]))
    print(reg_loss)


    if output_mode == 'loss':
      return total_loss
    else:
      return total_loss, conf_scores, proposals, image_feats, GT_class, activated_anc_ind, anc_per_img


  def inference(self, images, thresh=0.5, nms_thresh=0.7, mode='RPN'):

    assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'

    features, final_conf_probs, final_proposals = None, None, None
    pass
    if mode == 'RPN':
      features = [torch.zeros_like(i) for i in final_conf_probs] # dummy class
    return final_proposals, final_conf_probs, features


class TwoStageDetector(nn.Module):
  def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
               roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0)
    self.num_classes = num_classes
    self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h

    self.rpn = None
    self.cls_layer = None

  def forward(self, images, bboxes):
    """
    Outputs:
    - total_loss: Torch scalar giving the overall training loss.
    """
    total_loss = None

    return total_loss

  def inference(self, images, thresh=0.5, nms_thresh=0.7):

    final_proposals, final_conf_probs, final_class = None, None, None

    pass

    return final_proposals, final_conf_probs, final_class

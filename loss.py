class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
	此类计算目标和网络预测之间的分配
    出于效率原因，目标不包括no_object。正因为如此，一般来说，
    预测多于目标。在这种情况下，我们对最佳预测进行 1 对 1 匹配，
    而其他的则不匹配（因此被视为非对象）。
    """
 
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
			cost_class：这是分类误差在匹配成本中的相对权重
            cost_bbox：这是匹配成本中边界框坐标的 L1 误差的相对权重
            cost_giou：这是匹配成本中边界框的 giou 损失的相对权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
 
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
		执行匹配
        参数：
            输出：这是一个至少包含以下条目的字典：
                 “pred_logits”：带有分类对数的暗 [batch_size、num_queries、num_classes] 张量
                 “pred_boxes”：具有预测框坐标的暗 [batch_size， num_queries， 4] 张量
            目标：这是一个目标列表（len（target） = batch_size），其中每个目标都是一个字典，其中包含：
                 “标签”：暗 [num_target_boxes] 的张量（其中 num_target_boxes 是真
                           目标中的对象）包含类标签
                 “盒子”：包含目标盒子坐标的 dim [num_target_boxes， 4] 张量
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
		返回：
            大小batch_size的列表，包含 （index_i， index_j） 元组，其中：
                - index_i是所选预测的索引（按顺序）
                - index_j是相应选定目标的索引（按顺序）
            对于每个批处理元素，它包含：
                len（index_i） = len（index_j） = min（num_queries， num_target_boxes）
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
 
        # We flatten to compute the cost matrices in a batch我们展平以批量计算成本矩阵
        # 将batch维度合并,out_prob shape为[200,92],out_bbox shape为[200,4]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
 
        # Also concat the target labels and boxes 同时连接目标标签和框
        # 将目标的ground truth id和bbox的batch维度合并,假设此处有22个类
        # 即假设第一张图像有20个类,第二张图像有2个类
        # 那么tgt_ids的shape为22,tgt_bbox的shape为[22,4]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
 
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
		# 计算分类成本。与损失相反，我们不使用 NLL，
        # 但近似为 1 - proba[目标 class
        # The 1 is a constant that doesn't change the matching, it can be ommitted. 1 是一个常量，不会改变匹配，可以省略。
        # 取出out_prob每一行对应索引中的元素,此时cost_class的shape为[200,22]
        cost_class = -out_prob[:, tgt_ids]
 
        # Compute the L1 cost between boxes
        # 计算out_bbox和tgt_bbox的L1距离,此时cost_bbox的shape为[200,22]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
 
        # Compute the giou cost betwen boxes
        # 计算giou,此时cost_giou的shape为[200,22]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
 
        # Final cost matrix
        # C [200,22]->[2,100,22]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        # size [20,2]
        sizes = [len(v["boxes"]) for v in targets]
        # 匈牙利算法的实现,指派最优的目标索引,输出一个二维列表,第一维是batch为0,即一个batch中第一张图像通过匈
        # 牙利算法计算得到的最优解的横纵坐标,第二维是batch为1,即一个batch中第二张图像,后面的batch维度以此类推
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
 
        for i, c in enumerate(C.split(sizes, -1)):
            import numpy as np
            cost_matrix = np.asarray(c[i])
            # print('cost_matrix:', cost_matrix)
            row_ind, col_ind = linear_sum_assignment(c[i])
 
            for (row, col) in zip(row_ind, col_ind):
                print(row, col, '***', cost_matrix[row][col])
        print('11:', indices)
        # 由于indices调用的是scipy.optimize库,输出的是一个numpy数组,最后输出的indices需要转换为torch tensor
        now = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return now



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
		此类计算 DETR 的损失。
    该过程分两步进行：
        1） 我们计算真实方框和模型输出之间的匈牙利赋值
        2）我们监督每对匹配的地面真相/预测（监督类和框）
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
			创建条件。
        参数：
            num_classes：对象类别的数量，省略特殊的无对象类别
            匹配器：能够计算目标和建议之间的匹配的模块
            weight_dict：包含损失名称及其相对权重值的字典。
            eos_coef：应用于无对象类别的相对分类权重
            损失：要应用的所有损失列表。有关可用损失的列表，请参阅get_loss。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
 
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		分类丢失 （NLL）
        目标字典必须包含包含 dim [nb_target_boxes] 张量的关键“标签”
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
 
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        # batch_idx得到的就是匈牙利算法得到的索引是属于哪一张图像,如tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # 前20属于第一张,最后两个属于第二张
        # src_idx则表示匈牙利算法得到的横坐标信息,如tensor([14, 20, 24, 28, 32, 37, 42, 46, 50, 52, 60, 64, 67, 70, 79, 87, 91, 93, 94, 97, 6, 31])
        # idx = (batch_idx,src_idx)
        idx = self._get_src_permutation_idx(indices)
        # target_classes_o由targets["labels"] 根据 indices的纵坐标重新排序得到
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 根据idx将target_classes_o中的值映射到[2,100]值为91的张量中
        target_classes[idx] = target_classes_o
        # 计算预测输出的类别损失
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
 
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here  TODO这可能应该是一个单独的损失，而不是在这里被黑客入侵
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
 
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
		计算基数误差，即预测非空框数量的绝对误差
        这不是真正的损失，它仅用于日志记录目的。它不传播梯度
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class) 
		#计算不是“无对象”的预测数量（这是最后一个类）
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
 
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		   计算与边界框、L1 回归损失和 GIoU 损失相关的损耗
           目标字典必须包含包含 dim 张量的关键“框” [nb_target_boxes， 4]
           目标框的格式（center_x、center_y、w、h）应按图像大小进行标准化。
        """
        assert 'pred_boxes' in outputs
 
        # _get_src_permutation_idx(indices)返回两个值batch_idx和src_idx
        # batch_idx得到的就是匈牙利算法得到的索引是属于哪一张图像,如tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # 前20属于第一张,最后两个属于第二张
        # src_idx则表示匈牙利算法得到的横坐标信息,如tensor([14, 20, 24, 28, 32, 37, 42, 46, 50, 52, 60, 64, 67, 70, 79, 87, 91, 93, 94, 97, 6, 31])
        # idx = (batch_idx,src_idx)
        idx = self._get_src_permutation_idx(indices)
        # 根据indices的横坐标提取预测输出outputs['pred_boxes']中的对应bbox
        src_boxes = outputs['pred_boxes'][idx]
        # target_boxes由targets['boxes'] 根据 indices的纵坐标重新排序得到
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
 
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
 
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
 
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
 
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
		   计算与蒙版相关的损失：焦点损失和骰子损失。
           目标字典必须包含包含 dim [nb_target_boxes， h， w] 张量的关键“掩码”
        """
        assert "pred_masks" in outputs
 
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
		#TODO 使用有效来掩盖由于填充丢失而导致的无效区域
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
 
        # upsample predictions to the target size 将预测上采样到目标大小
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
 
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
 
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices 跟踪指数的排列预测
        # 输入参数indices是匹配的预测（query）索引与GT的索引，其形式在上述SetCriterion(iv)
        # 图中注释已有说明。该方法返回一个tuple，代表所有匹配的预测结果的batch
        # index（在当前batch中属于第几张图像）和
        # query
        # index（图像中的第几个query对象）。
 
        # batch_idx得到的就是匈牙利算法得到的索引是属于哪一张图像,如tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # 前20属于第一张,最后两个属于第二张
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # src_idx则表示匈牙利算法得到的横坐标信息,如tensor([14, 20, 24, 28, 32, 37, 42, 46, 50, 52, 60, 64, 67, 70, 79, 87, 91, 93, 94, 97, 6, 31])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
 
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
 
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
 
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
		这将执行损失计算。
        参数：
             输出：张量字典，格式见模型的输出规范
             目标：字典列表，使得 len（目标） == batch_size。
                      每个字典中的预期键取决于应用的损失，请参阅每个损失的文档
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
  
        # Retrieve the matching between the outputs of the last layer and the targets  检索最后一层的输出与目标之间的匹配
        # 假设这是其中一组输出[(tensor([14, 20, 24, 28, 32, 37, 42, 46, 50, 52, 60, 64, 67, 70, 79, 87, 91, 93,94, 97]),
        # tensor([13,  1, 19,  0, 18,  2,  7,  4, 17,  3, 14,  6,  8,  9, 12, 11, 16, 15, 10, 5])),
        # (tensor([ 6, 31]), tensor([1, 0]))]
        # 其中的(14,13)即为匈牙利算法求出的第一张图像中最优解的横纵坐标
        indices = self.matcher(outputs_without_aux, targets)
 
        # Compute the average number of target boxes accross all nodes, for normalization purposes计算所有节点上目标框的平均数量，以实现规范化目的
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
  
        # Compute all the requested losses 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
 
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.  在辅助损耗的情况下，我们对每个中间层的输出重复此过程。
        if 'aux_outputs' in outputs: 
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them. 中间掩模损失成本太高，无法计算，我们忽略它们。
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer 仅为最后一层启用日志记录
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
 
        return losses
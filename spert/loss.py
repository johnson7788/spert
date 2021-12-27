from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        """
        计算损失
        :param entity_logits: 预测值
        :type entity_logits:
        :param rel_logits: 关系预测值
        :type rel_logits:
        :param entity_types: ground_truth
        :type entity_types:
        :param rel_types: ground_truth
        :type rel_types:
        :param entity_sample_masks:
        :type entity_sample_masks:
        :param rel_sample_masks:
        :type rel_sample_masks:
        :return:
        :rtype:
        """
        # 合并前2个维度entity_logits shape [batch_size, entiti_num, entity_label_num] -->[batch_size * entiti_num, entity_label_num]
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        # 合并前2个维度 [batch_size, entiti_num] --> [batch_size*entiti_num,], entity_types是ground_truth
        entity_types = entity_types.view(-1)
        # entity_sample_masks --> [batch_size, entiti_num] eg: [2,106] -->[batch_size*entiti_num,]
        entity_sample_masks = entity_sample_masks.view(-1).float()
        # eg: entity_logits: [212,5],  entity_types: [212,], entity_loss [212,]
        entity_loss = self._entity_criterion(entity_logits, entity_types)
        # entity_loss: tensor(1.9920, grad_fn=<DivBackward0>)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()
        # 关系的损失， rel_sample_masks， 【batch_size,rel_num] --> 【batch_size*rel_num], eg: [60,]
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()
        # 如果存在关系
        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])
            # rel_types是ground_truth
            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count
            # eg: rel_loss: tensor(0.7671, grad_fn=<DivBackward0>)
            #总的损失等于实体损失+关系损失
            train_loss = entity_loss + rel_loss
        else:
            # 当不存在positive/negative relation samples时，只有实体损失
            train_loss = entity_loss

        train_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()

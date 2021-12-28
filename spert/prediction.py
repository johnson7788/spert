import json
from typing import Tuple

import torch

from spert import util
from spert.input_reader import BaseInputReader


def convert_predictions(batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                        batch_rels: torch.tensor, batch: dict, rel_filter_threshold: float,
                        input_reader: BaseInputReader, no_overlapping: bool = False):
    """

    :param batch_entity_clf: 【batch_size, 枚举的实体数量，实体的label总数] 预测每个枚举的实体的logtis
    :type batch_entity_clf:
    :param batch_rel_clf: [batch_size, 关系数量，关系的label总数] 对每个可能的实体，进行两两配对后，预测出可能的关系，对关系进行判断后的logits
    :type batch_rel_clf:
    :param batch_rels:   [batch_size, 关系的数量，2】 实体的位置信息
    :type batch_rels:
    :param batch: 一个batch的数据信息
    'encodings' = {Tensor: (1, 26)} tensor([[  101,  1130, 12439,   117,  1103,  4186,  2084,  1104,  1103,  1244,\n          1311,   117, 25427,   156,   119,  4468,   117,  1108,  1255,  1107,\n          4221, 16836,   117,  3197,   119,   102]])
'context_masks' = {Tensor: (1, 26)} tensor([[True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True]])
'entity_masks' = {Tensor: (1, 185, 26)} tensor([[[False,  True, False,  ..., False, False, False],\n         [False, False,  True,  ..., False, False, False],\n         [False, False, False,  ..., False, False, False],\n         ...,\n         [False, False, False,  ..., False, False, False],\n
'entity_sizes' = {Tensor: (1, 185)} tensor([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,\n          1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,\n          2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,\n          3,
'entity_spans' = {Tensor: (1, 185, 2)} tensor([[[ 1,  2],\n         [ 2,  3],\n         [ 3,  4],\n         [ 4,  5],\n         [ 5,  6],\n         [ 6,  7],\n         [ 7,  8],\n         [ 8,  9],\n         [ 9, 10],\n         [10, 11],\n         [11, 12],\n         [12, 13],\n         [13, 15],\n
'entity_sample_masks' = {Tensor: (1, 185)} tensor([[True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True, True, True, True, True, True, True, True, True, True, True,\n         True
    :type batch:
    :param rel_filter_threshold: 0.4
    :type rel_filter_threshold:  关系的可能性，大于这个可能性的才保留
    :param input_reader: 数据读取器
    :type input_reader:
    :param no_overlapping: eg:False
    :type no_overlapping: bool
    :return:
    :rtype:
    """
    # 获取最大激活（预测实体类型的索引）
    batch_entity_types = batch_entity_clf.argmax(dim=-1)
    # apply entity sample mask
    batch_entity_types *= batch['entity_sample_masks'].long()

    # 过滤关系的logits,  batch_rel_clf: [batch_size, 关系数量，关系的label总数]
    batch_rel_clf[batch_rel_clf < rel_filter_threshold] = 0

    batch_pred_entities = []
    batch_pred_relations = []

    for i in range(batch_rel_clf.shape[0]):
        # 获取每个关系的预测结果
        entity_types = batch_entity_types[i]
        entity_spans = batch['entity_spans'][i]
        entity_clf = batch_entity_clf[i]
        rel_clf = batch_rel_clf[i]
        rels = batch_rels[i]

        # convert predicted entities
        sample_pred_entities = _convert_pred_entities(entity_types, entity_spans,
                                                      entity_clf, input_reader)

        # convert predicted relations
        sample_pred_relations = _convert_pred_relations(rel_clf, rels,
                                                        entity_types, entity_spans, input_reader)

        if no_overlapping:
            sample_pred_entities, sample_pred_relations = remove_overlapping(sample_pred_entities,
                                                                             sample_pred_relations)

        batch_pred_entities.append(sample_pred_entities)
        batch_pred_relations.append(sample_pred_relations)
    # batch_pred_entities是预测实体 list, 是预测的关系，list
    # eg: batch_pred_entities = {list: 1} [[(9, 11, <spert.entities.EntityType object at 0x144e2d1f0>, 0.9992102384567261), (12, 16, <spert.entities.EntityType object at 0x144e2d250>, 0.9991939663887024), (20, 24, <spert.entities.EntityType object at 0x144e2d1f0>, 0.9984266757965088)]]
    #  0 = {list: 3} [(9, 11, <spert.entities.EntityType object at 0x144e2d1f0>, 0.9992102384567261), (12, 16, <spert.entities.EntityType object at 0x144e2d250>, 0.9991939663887024), (20, 24, <spert.entities.EntityType object at 0x144e2d1f0>, 0.9984266757965088)]
    #   0 = {tuple: 4} (9, 11, <spert.entities.EntityType object at 0x144e2d1f0>, 0.9992102384567261)
    #   1 = {tuple: 4} (12, 16, <spert.entities.EntityType object at 0x144e2d250>, 0.9991939663887024)
    #   2 = {tuple: 4} (20, 24, <spert.entities.EntityType object at 0x144e2d1f0>, 0.9984266757965088)
    return batch_pred_entities, batch_pred_relations


def _convert_pred_entities(entity_types: torch.tensor, entity_spans: torch.tensor,
                           entity_scores: torch.tensor, input_reader: BaseInputReader):
    # get entities that are not classified as 'None'
    valid_entity_indices = entity_types.nonzero().view(-1)
    pred_entity_types = entity_types[valid_entity_indices]
    pred_entity_spans = entity_spans[valid_entity_indices]
    pred_entity_scores = torch.gather(entity_scores[valid_entity_indices], 1,
                                      pred_entity_types.unsqueeze(1)).view(-1)

    # convert to tuples (start, end, type, score)
    converted_preds = []
    for i in range(pred_entity_types.shape[0]):
        label_idx = pred_entity_types[i].item()
        entity_type = input_reader.get_entity_type(label_idx)

        start, end = pred_entity_spans[i].tolist()
        score = pred_entity_scores[i].item()

        converted_pred = (start, end, entity_type, score)
        converted_preds.append(converted_pred)

    return converted_preds


def _convert_pred_relations(rel_clf: torch.tensor, rels: torch.tensor,
                            entity_types: torch.tensor, entity_spans: torch.tensor, input_reader: BaseInputReader):
    rel_class_count = rel_clf.shape[1]
    rel_clf = rel_clf.view(-1)

    # get predicted relation labels and corresponding entity pairs
    rel_nonzero = rel_clf.nonzero().view(-1)
    pred_rel_scores = rel_clf[rel_nonzero]

    pred_rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1)
    valid_rel_indices = rel_nonzero // rel_class_count
    valid_rels = rels[valid_rel_indices]

    # get masks of entities in relation
    pred_rel_entity_spans = entity_spans[valid_rels].long()

    # get predicted entity types
    pred_rel_entity_types = torch.zeros([valid_rels.shape[0], 2])
    if valid_rels.shape[0] != 0:
        pred_rel_entity_types = torch.stack([entity_types[valid_rels[j]] for j in range(valid_rels.shape[0])])

    # convert to tuples ((head start, head end, head type), (tail start, tail end, tail type), rel type, score))
    converted_rels = []
    check = set()

    for i in range(pred_rel_types.shape[0]):
        label_idx = pred_rel_types[i].item()
        pred_rel_type = input_reader.get_relation_type(label_idx)
        pred_head_type_idx, pred_tail_type_idx = pred_rel_entity_types[i][0].item(), pred_rel_entity_types[i][1].item()
        pred_head_type = input_reader.get_entity_type(pred_head_type_idx)
        pred_tail_type = input_reader.get_entity_type(pred_tail_type_idx)
        score = pred_rel_scores[i].item()

        spans = pred_rel_entity_spans[i]
        head_start, head_end = spans[0].tolist()
        tail_start, tail_end = spans[1].tolist()

        converted_rel = ((head_start, head_end, pred_head_type),
                         (tail_start, tail_end, pred_tail_type), pred_rel_type)
        converted_rel = _adjust_rel(converted_rel)

        if converted_rel not in check:
            check.add(converted_rel)
            converted_rels.append(tuple(list(converted_rel) + [score]))

    return converted_rels


def remove_overlapping(entities, relations):
    non_overlapping_entities = []
    non_overlapping_relations = []

    for entity in entities:
        if not _is_overlapping(entity, entities):
            non_overlapping_entities.append(entity)

    for rel in relations:
        e1, e2 = rel[0], rel[1]
        if not _check_overlap(e1, e2):
            non_overlapping_relations.append(rel)

    return non_overlapping_entities, non_overlapping_relations


def _is_overlapping(e1, entities):
    for e2 in entities:
        if _check_overlap(e1, e2):
            return True

    return False


def _check_overlap(e1, e2):
    if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
        return False
    else:
        return True


def _adjust_rel(rel: Tuple):
    adjusted_rel = rel
    if rel[-1].symmetric:
        head, tail = rel[:2]
        if tail[0] < head[0]:
            adjusted_rel = tail, head, rel[-1]

    return adjusted_rel


def store_predictions(documents, pred_entities, pred_relations, store_path):
    predictions = []

    for i, doc in enumerate(documents):
        tokens = doc.tokens
        sample_pred_entities = pred_entities[i]
        sample_pred_relations = pred_relations[i]

        # convert entities
        converted_entities = []
        for entity in sample_pred_entities:
            entity_span = entity[:2]
            span_tokens = util.get_span_tokens(tokens, entity_span)
            entity_type = entity[2].identifier
            converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
            converted_entities.append(converted_entity)
        converted_entities = sorted(converted_entities, key=lambda e: e['start'])

        # convert relations
        converted_relations = []
        for relation in sample_pred_relations:
            head, tail = relation[:2]
            head_span, head_type = head[:2], head[2].identifier
            tail_span, tail_type = tail[:2], tail[2].identifier
            head_span_tokens = util.get_span_tokens(tokens, head_span)
            tail_span_tokens = util.get_span_tokens(tokens, tail_span)
            relation_type = relation[2].identifier

            converted_head = dict(type=head_type, start=head_span_tokens[0].index,
                                  end=head_span_tokens[-1].index + 1)
            converted_tail = dict(type=tail_type, start=tail_span_tokens[0].index,
                                  end=tail_span_tokens[-1].index + 1)

            head_idx = converted_entities.index(converted_head)
            tail_idx = converted_entities.index(converted_tail)

            converted_relation = dict(type=relation_type, head=head_idx, tail=tail_idx)
            converted_relations.append(converted_relation)
        converted_relations = sorted(converted_relations, key=lambda r: r['head'])

        doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=converted_entities,
                               relations=converted_relations)
        predictions.append(doc_predictions)

    # store as json
    with open(store_path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)

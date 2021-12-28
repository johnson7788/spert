import random

import torch

from spert import util


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    """
    创建训练样本, 被多线程调用了，创建训练集，在
    :param doc: 一条数据集合
    :type doc:
    :param neg_entity_count: 100
    :type neg_entity_count:
    :param neg_rel_count: 100
    :type neg_rel_count:
    :param max_span_size:  10
    :type max_span_size: int
    :param rel_type_count:  关系的类型的数量， 6
    :type rel_type_count:
    :return:
    :rtype:
    """
    # encodings: [101, 1249, 1111, 1103, 107, 4765, 1111, 1898, 118, 1729, 27597, 117, 107, 1103, 140, 15681, 2101, 18029, 1116, 1115, 107, 3239, 1105, 1927, 5172, 1619, 1103, 4765, 4762, 1106, 1412, 3268, 1105, 5401, 1103, 2666, 1104, 1160, 4765, 1121, 1103, 14885, 142, 18791, 1186, 6257, 131, 22796, 2825, 1186, 1107, 6627, 6090, 129, 119, 127, 117, 1105, 7847, 1186, 3902, 1107, 6090, 129, 119, 130, 117, 2292, 1103, 3239, 5594, 1112, 107, 1632, 7705, 119, 107, 113, 23690, 2116, 6216, 1112, 1502, 114, 1109, 140, 15681, 2101, 5115, 1115, 22796, 2825, 1186, 1110, 170, 107, 5530, 2301, 1114, 6047...
    encodings = doc.encoding
    # 原始token的数量， eg: token_count: 32
    token_count = len(doc.tokens)
    # token encoding到id之后的数量, eg: 38
    context_size = len(encodings)

    # 正样本实体，pos_entity_spans: [(1, 4), (11, 13), (30, 33)]  每个实体的开始和结束位置
    # pos_entity_types, [3, 2, 1], 对应的实体类型
    # pos_entity_sizes， [2, 1, 2]， 对应的实体的长度
    # pos_entity_masks，对应的实体在句子中的位置 [tensor([False,  True,  True,  True, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False]), tensor([False, False, False, False, False, False, False, False, False, False,
    #         False,  True,  True, False, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False]), tensor([False, False, False, False, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False, False, False,
    #         False, False, False, False, False, False, False, False, False, False,
    #          True,  True,  True, False, False, False, False, False])]
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))

    # 正样本关系
    # 收集实体对之间的关系， entity_pair_relations: 这个样本中所有的实体对和关系的实例
    # eg: entity_pair_relations = {dict: 1} {(<spert.entities.Entity object at 0x7f6686e66070>, <spert.entities.Entity object at 0x7f6686e660d0>): [<spert.entities.Relation object at 0x7f6686e661c0>]}
    #  (<spert.entities.Entity object at 0x7f6686e66070>, <spert.entities.Entity object at 0x7f6686e660d0>) = {list: 1} [<spert.entities.Relation object at 0x7f6686e661c0>]
    #   0 = {Relation} <spert.entities.Relation object at 0x7f6686e661c0>
    #    first_entity = {Entity} U.S. Coast Guard Training Center
    #    head_entity = {Entity} U.S. Coast Guard Training Center
    #    relation_type = {RelationType} <spert.entities.RelationType object at 0x7f66a0144880>
    #    reverse = {bool} False
    #    second_entity = {Entity} Petaluma
    #    tail_entity = {Entity} Petaluma
    entity_pair_relations = dict()
    for rel in doc.relations:
        pair = (rel.head_entity, rel.tail_entity)
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # 构建关系的正样本, pos_rels: [(0, 1)], 实体的位置索引，对应着pos_entity_spans中的位置
    # pos_rel_spans: [((10, 17), (20, 21))],pos_rel_types: [[0, 1, 0, 0, 0]]
    # pos_rel_masks 头实体和尾实体中间的词的mask
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair
        s1, s2 = head_entity.span, tail_entity.span    # eg: s1: (23, 25), s2:(31, 34)
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))  #eg: [(0, 2)]
        pos_rel_spans.append((s1, s2))  #eg: [((23, 25), (31, 34))]
        #eg: eg: [5], 关系类型
        pair_rel_types = [r.relation_type.index for r in rels]   #
        # 关系类型变成one-hot, [0, 1, 0, 0, 0]
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]  #eg: [0, 0, 0, 0, 1]
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))

    #实体的负样本构造， 迭代最大实体跨度，迭代token，生成负样本跨度和负样本的长度
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # 负样本实体抽样，例如neg_entity_samples是抽样的结果，neg_entity_samples长度是100， 每个元素是((54, 62), 6)
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))
    #解包，取出跨度和实体长度， 负样本的位置mask和负样本的实体类型设为0
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])
    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    #关系负样本
    # 仅使用强负关系，即不相关的实际（标注）实体对
    neg_rel_spans = []
    # 遍历实体2次，生成负样本关系
    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # 不要给实体自己和自己添加负样本关系
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    #关系的负样本抽样， eg: neg_rel_count: 100, eg: neg_rel_spans: [((42, 45), (30, 38))], min(len(neg_rel_spans), neg_rel_count)：如果设定的负样本的数量和实际产生的负样本的数量不一致，那么用最少的
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))
    # 关系负样本的 eg: [(1, 0)]
    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    # 关系的负样本的2个实体的之间的距离mask
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    # eg: [(0, 0, 0, 0, 0)]，关系生成one-hot表示
    neg_rel_types = [(0,) * (rel_type_count-1)] * len(neg_rel_spans)

    # 合并实体正样本和负样本，
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)
    # 合并关系正样本和负样本
    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks
    # entity_masks的形状是[样本数量，句子长度],  entity_types和entity_sizes: List, 【样本数量,]
    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    #rels：维度[关系数量，2】，2代表着这个关系的实体对在实体列表的位置, eg: [(0, 1), (1, 0)], 第0个实体和第1个实体有一对关系
    # rel_masks： 关系的mask【关系数量，实体1和实体2之间的单词的mask的长度】
    #rel_types: 关系的类型，即ground_truth, [关系数量，关系的标签的总数量]
    assert len(rels) == len(rel_masks) == len(rel_types)
    # create tensors  token id变成tensor
    encodings = torch.tensor(encodings, dtype=torch.long)
    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    # 创建样本masking：
    # 一个批次的实体和关系的样本mask
    # 由于样本的batch stack，可能必须创建“padding”实体/关系
    # 在后面计算损失的时候使用， 如果存在实体mask，那么
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)   #entity_types: [样本数量，]
        entity_masks = torch.stack(entity_masks)   #entity_masks: [样本数量，句子长度]
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)  #entity_sizes :  [样本数量，]
        #TODO: 做什么用
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)   # bool [样本数量，]，
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
    if rels:
        rels = torch.tensor(rels, dtype=torch.long)   # [关系数量，关系的头实体和尾实体的位置，在实体列表中的位置]
        rel_masks = torch.stack(rel_masks)          # 【关系数量，句子长度]
        rel_types = torch.tensor(rel_types, dtype=torch.float32)  #【关系数量，关系标签总数】
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool) #【关系数量，】
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count-1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(doc, max_span_size: int):
    """

    :param doc:
    :type doc:
    :param max_span_size:
    :type max_span_size:
    :return:
    :rtype:
    """
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks)


def create_entity_mask(start, end, context_size):
    """
    mask出这个实体的位置信息
    :param start:  1
    :type start:
    :param end:  4
    :type end:
    :param context_size:  38
    :type context_size:
    :return:
    :rtype:
    """
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    """
    两个关系的实体中间的位置的mask的信息， 头实体和尾实体的中间位置
    :param s1:(23, 25)
    :type s1:
    :param s2:(31, 34)
    :type s2:
    :param context_size: 49
    :type context_size:
    :return:
    :rtype:
    """
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    """
    collate 函数
    :param batch: 一个批次的数据，list格式， 一个元素的内容如下
            0 = {dict: 10} {'encodings': tensor([  101,  2189,  1540,  2442,  1814,  1205,  1118,  1344,  7390,  2416,\n         1149, 12062,  1115,  4634,  1164,   127,   117,  1288,  4481,  1105,\n         5028,  1462,  1118,  2685,  1756, 18221,  1107,  2238,  2460,  1105,\n
         'encodings' = {Tensor: (40,)} tensor([  101,  2189,  1540,  2442,  1814,  1205,  1118,  1344,  7390,  2416,\n         1149, 12062,  1115,  4634,  1164,   127,   117,  1288,  4481,  1105,\n         5028,  1462,  1118,  2685,  1756, 18221,  1107,  2238,  2460,  1105,\n         6309,  6292,   117,  1163, 18221, 15465,  4101,  4368,   119,   102])
         'context_masks' = {Tensor: (40,)} tensor([True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True])
         'entity_masks' = {Tensor: (105, 40)} tensor([[False, False, False,  ..., False, False, False],\n        [False, False, False,  ..., False, False, False],\n        [False, False, False,  ..., False, False, False],\n        ...,\n        [False, False, False,  ..., False, False, False],\n        [False, False, False,  ..., False, False, False],\n        [False, False, False,  ..., False, False, False]])
         'entity_sizes' = {Tensor: (105,)} tensor([ 3,  2,  2,  1,  2,  1, 10,  3, 10,  7,  1,  2,  7,  8,  1,  4,  2,  5,\n         3,  9,  4,  1,  9,  1,  6,  1,  2,  3,  6,  1,  4,  1,  3,  8,  8,  2,\n         6,  7,  9,  5,  4,  4,  4,  2,  4,  9,  2,  2, 10,  5, 10,  9,  9,  2,\n         5,  6,  6,  4, 10,  7,  6,  5,  4,  3,  8,  1,  1,  9,  1,  5,  5, 10,\n         8,  5, 10,  4,  8,  6,  5,  3,  9,  5,  1,  7,  3,  4,  9, 10, 10,  5,\n         8,  2,  2,  4,  8,  3,  8,  7,  8,  3,  1,  4,  2,  3, 10])
         'entity_types' = {Tensor: (105,)} tensor([2, 1, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n        0, 0, 0, 0, 0, 0, 0, 0, 0])
         'rels' = {Tensor: (20, 2)} tensor([[4, 0],\n        [4, 3],\n        [0, 4],\n        [0, 2],\n        [3, 4],\n        [4, 1],\n        [3, 1],\n        [1, 0],\n        [1, 3],\n        [4, 2],\n        [2, 1],\n        [2, 0],\n        [3, 2],\n        [1, 2],\n        [1, 4],\n        [0, 1],\n        [2, 3],\n        [0, 3],\n        [3, 0],\n        [2, 4]])
         'rel_masks' = {Tensor: (20, 40)} tensor([[False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True, False, False, False, False],\n        [False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False,  True, False, False, False, False],\n        [False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True, False, False, False, False],\n        [False, False, False, False, False, False...
         'rel_types' = {Tensor: (20, 5)} tensor([[1., 0., 0., 0., 0.],\n        [1., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.],\n        [0., 0., 0., 0., 0.]])
         'entity_sample_masks' = {Tensor: (105,)} tensor([True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True, True])
         'rel_sample_masks' = {Tensor: (20,)} tensor([True, True, True, True, True, True, True, True, True, True, True, True,\n        True, True, True, True, True, True, True, True])
    :type batch:
    :return:
         padded_batch = {dict: 10} {'encodings': tensor([[  101,   113, 18430,   114,  8254,   117,  1345,   126,   113,   157,\n         14962,  4538,  2591,  2349,   114,   118,   118,  1109,  2341,  1104,\n          4308,  9018,  1116,  1104,  1103,  3047,  1113,  4354,  1105, 17612,\n
         'encodings' = {Tensor: (2, 66)} tensor([[  101,   113, 18430,   114,  8254,   117,  1345,   126,   113,   157,\n         14962,  4538,  2591,  2349,   114,   118,   118,  1109,  2341,  1104,\n          4308,  9018,  1116,  1104,  1103,  3047,  1113,  4354,  1105, 17612,\n          1107,  19
         'context_masks' = {Tensor: (2, 66)} tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  Tr
         'entity_masks' = {Tensor: (2, 109, 66)} tensor([[[False, False, False,  ..., False, False, False],\n         [False, False, False,  ..., False, False, False],\n         [False, False, False,  ..., False, False, False],\n         ...,\n         [False, False, False,  ..., False, False, False],\n
         'entity_sizes' = {Tensor: (2, 109)} tensor([[ 1,  2, 13,  1,  1,  1,  1,  1,  4, 10,  3,  1,  8,  7,  4,  9,  1,  1,\n          8,  7,  4,  3,  6,  8,  5,  1,  1,  6,  7,  7,  8,  4,  3,  2,  7,  2,\n          8, 10,  3,  5,  4,  5,  1,  5,  9,  9,  4,  1, 10,  3,  6,  5,  6,  4,\n          2,
         'entity_types' = {Tensor: (2, 109)} tensor([[1, 4, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n         0, 0
         'rels' = {Tensor: (2, 72, 2)} tensor([[[5, 8],\n         [6, 8],\n         [7, 8],\n         [3, 0],\n         [0, 3],\n         [0, 4],\n         [6, 7],\n         [4, 3],\n         [3, 8],\n         [6, 3],\n         [7, 2],\n         [8, 7],\n         [1, 4],\n         [6, 4],\n         [8, 3],\n
         'rel_masks' = {Tensor: (2, 72, 66)} tensor([[[False, False, False,  ..., False, False, False],\n         [False, False, False,  ..., False, False, False],\n         [False, False, False,  ..., False, False, False],\n         ...,\n         [False, False, False,  ..., False, False, False],\n
         'rel_types' = {Tensor: (2, 72, 5)} tensor([[[0., 0., 0., 0., 1.],\n         [0., 0., 0., 0., 1.],\n         [0., 0., 0., 0., 1.],\n         [0., 0., 0., 0., 0.],\n         [0., 0., 0., 0., 0.],\n         [0., 0., 0., 0., 0.],\n         [0., 0., 0., 0., 0.],\n         [0., 0., 0., 0., 0.],\n
         'entity_sample_masks' = {Tensor: (2, 109)} tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True],\n        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True, ...
         'rel_sample_masks' = {Tensor: (2, 72)} tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,\n          True,  True],\n        [ True,  True,  True,  True,  True,  True, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False, False, False, False, False, False, False, False,\n         False, False, False...
         __len__ = {int} 10
    :rtype:
    """
    padded_batch = dict()
    # dict_keys(['encodings', 'context_masks', 'entity_masks', 'entity_sizes', 'entity_types', 'rels', 'rel_masks', 'rel_types', 'entity_sample_masks', 'rel_sample_masks'])
    keys = batch[0].keys()

    for key in keys:
        # samples 遍历出每个相同类型的向量，list格式，长度是一个批次的样本的个数
        samples = [s[key] for s in batch]
        #batch[0][key].shape： eg： 39， 是token 变成id后的个数
        if not batch[0][key].shape:
            # 如果不存在 token 变成id后的个数，那么stack样本
            padded_batch[key] = torch.stack(samples)
        else:
            # 否则，对样本进行padding，然后stack
            padded_batch[key] = util.padded_stack(samples)
    # padded_batch是一个批次的样本，进行了padding后的
    return padded_batch

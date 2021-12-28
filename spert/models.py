import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """
    获得特定的token嵌入（例如[CLS]）。
    :param h: 句子经过bert后的隐藏层向量【batch_size, seq_length, emb_size] eg: [2,49,768]
    :type h:
    :param x: encodings后的id，[batch_size, seq_length]
    :type x: eg: tensor([[  101,  1130,  2668,   117,   170,  7141,  1107,  5043,  1276,  2132,
         11374,  5425,  1104, 23637,  2499,  7206, 18638,   117,  1103,  4806,
         15467,  1104,  1697,  5107,   119,   102],
        [  101,  1130, 12439,   117,  1103,  2835,  2084,  1104,  1103,  1244,
          1311,   117, 19936,   139,   119, 10011,   117,  1108,  1255,  1107,
          8056,   117,  3197,   119,   102,     0]], device='cuda:0')
    :param token:  eg: 101, cls的id
    :type token:int
    :return:
    :rtype:
    """
    # emb_size： 768
    emb_size = h.shape[-1]
    # 合并前2个维度
    token_h = h.view(-1, emb_size)
    # 拉平x
    flat = x.contiguous().view(-1)

    # 获取给定token的上下文嵌入。获取token_h中是和flat中的tokenid等于给定token的向量
    token_h = token_h[flat == token, :]
    # token_h： 【batch_size, embedding_dim]
    return token_h


class SpERT(BertPreTrainedModel):
    """ 基于跨度的模型来联合提取实体和关系"""
    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # 首先加载Bert模型
        self.bert = BertModel(config)

        # 新建分类层, 关系分类， relation_types是关系label的数量，这里是5，5个关系
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        # 实体的大小进行embedding
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        # 考虑的最大的关系数量
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("设置冻结transformer权重")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        """
            # encodings: [batch_size, padding后的seq_length], token变成id后的内容
            # context_masks: [batch_size, padding后的seq_length】， 样本的实际的长度的mask
            # entity_masks:[batch_size,实体数量，padding后的seq_length] 实体的在句子位置mask
            # entity_sizes: [batch_size,实体数量】 每个实体的长度
            # 'entity_types',[batch_size,,实体数量】
            # 'rels',[batch_size, 关系数量，头实体和尾实体的位置索引】， eg[2,6,2]， 6是包含了正负样本
            # 'rel_masks',[batch_size,关系数量,padding后的seq_length] 关系在样本中的位置，即2个实体之间的词
            # 'rel_types',[batch_size,关系数量，关系的标签总数]  one-hot了的关系
            # 'entity_sample_masks',[batch_size,实体数量]， padding后
            # 'rel_sample_masks'[batch_size,关系数量]，padding后，样本1可能有10个关系，样本2有3个关系，那么样本2就有7个FALSE
        从最后一个transformer层获得上下文的token嵌入。 训练
        :param encodings: 【batch_size,seq_length]
        :type encodings:
        :param context_masks:【batch_size,seq_length
        :type context_masks:
        :param entity_masks:【batch_size,实体数量, seq_length]
        :type entity_masks:
        :param entity_sizes:【batch_size,实体数量]
        :type entity_sizes:
        :param relations:【batch_size,关系数量，头实体和尾实体的位置索引]
        :type relations:
        :param rel_masks:【batch_size,关系数量, seq_length]， 关系在样本中的位置，即2个实体之间的词
        :type rel_masks:
        :return:
        :rtype:
        """
        context_masks = context_masks.float() #context_masks转换成float格式
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        # h是取最后一层隐藏层参数, h的shape, [batch_size, seq_length, hidden_size]
        batch_size = encodings.shape[0]

        # 实体分类，实体的大小的embedding, [batch_size,实体数量】-->[batch_size,实体数量,size_hidden_size】,eg: [2,104,25]
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        # entity_clf [ batch_size, entity_num, entity_label_num], eg: [2,102,5]  实体分类
        # entity_spans_pool [ batch_size, entity_num, embedding_size], eg: [2,102,768] 实体跨度中经过了最大池化， 如果一个实体中有3个词，找出最大的那个词代表这个实体
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # 关系分类， 可能的self._max_pairs最大的关系数量， h_large 【batch_size, 关系数量，seq_length, hidden_size]
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        # 初始一个分类的tensor, [batch_size, 关系数量，关系的label的总数], eg:[2,42, 5]
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # 获取关系的logits
        # 分块处理以减少内存使用
        for i in range(0, relations.shape[1], self._max_pairs):
            # 对关系候选者进行分类, entity_spans_pool[batch_size,实体数量，hidden_size], size_embeddings实体的长度的embedding [batch_size,实体数量，size_hidden_size]
            # relations:【batch_size,关系数量，头实体和尾实体的位置索引] 关系的头实体和尾实体在实体向量中的位置索引
            # 'rel_masks',[batch_size,关系数量,padding后的seq_length] 关系在样本中的位置，即2个实体之间的词
            # h_large 【batch_size, 关系数量，seq_length, hidden_size]
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # 分块后的关系的预测结果
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
        # 返回实体和关系预测的logits
        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        """
        从最后一个transformer层获取上下文的token嵌入
            context_masks = {Tensor: (1, 26)} tensor([[True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True]])
            encodings = {Tensor: (1, 26)} tensor([[  101,  1130, 12439,   117,  1103,  4186,  2084,  1104,  1103,  1244,\n          1311,   117, 25427,   156,   119,  4468,   117,  1108,  1255,  1107,\n          4221, 16836,   117,  3197,   119,   102]])
            entity_masks = {Tensor: (1, 185, 26)} tensor([[[False,  True, False,  ..., False, False, False],\n         [False, False,  True,  ..., False, False, False],\n         [False, False, False,  ..., False, False, False],\n         ...,\n         [False, False, False,  ..., False, False, False],\n
            entity_sample_masks = {Tensor: (1, 185)} tensor([[True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True, True, True, True, True, True, True, True, True, True, True,\n         True, True, True, True, True, True, True, True, True, True, True, True,\n         True
            entity_sizes = {Tensor: (1, 185)} tensor([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,\n          1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,\n          2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,\n          3,
            entity_spans = {Tensor: (1, 185, 2)} tensor([[[ 1,  2],\n         [ 2,  3],\n         [ 3,  4],\n         [ 4,  5],\n         [ 5,  6],\n         [ 6,  7],\n         [ 7,  8],\n         [ 8,  9],\n         [ 9, 10],\n         [10, 11],\n         [11, 12],\n         [12, 13],\n         [13, 15],\n
            self = {SpERT} SpERT(\n  (bert): BertModel(\n    (embeddings): BertEmbeddings(\n      (word_embeddings): Embedding(28996, 768, padding_idx=0)\n      (position_embeddings): Embedding(512, 768)\n      (token_type_embeddings): Embedding(2, 768)\n      (LayerNorm): LayerNorm((768,
        :param encodings:  token转换成id, [batch_size, seq_length]
        :type encodings:
        :param context_masks:  样本的实际的长度的mask, [batch_size, seq_length]
        :type context_masks:
        :param entity_masks:  [batch_size,实体数量，padding后的seq_length] 实体的在句子位置mask, 实体数量是枚举的所有的可能
        :type entity_masks:  [batch_size,实体数量], 枚举的实体从最小到最大的跨度
        :param entity_sizes:
        tensor([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
          2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
          4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
          5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,
          6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
          7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,
          8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
          9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10]])
        :type entity_sizes:
        :param entity_spans:  [batch_size,实体数量， 2]， 2表示头实体和尾实体的位置
        :type entity_spans:
        :param entity_sample_masks: [batch_size,实体数量]， 样本1可能有10个实体，样本2有3个实体，那么样本2就有7个FALSE
        :type entity_sample_masks:
        :return:
        :rtype:
        """
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        # ctx_size：序列长度
        ctx_size = context_masks.shape[-1]
        #实体分类, size_embeddings, [batch_size,实体数量，25】
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        # entity_clf: [batch_size, 实体数量，实体的labels总数]
        # entity_spans_pool:  [batch_size,实体数量，hidden_size]
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # 忽略不构成关系的实际实体的实体候选对象（基于分类器）, entity_spans:  [batch_size,实体数量， 2]， 2表示头实体和尾实体的位置
        # relations： 【batch_size,关系的数量，2】 2是头实体尾实体的位置
        # rel_masks: 【batch_size,关系的数量,seq_length] 2个实体之间的词的位置的mask
        # rel_sample_masks:  [batch_size,关系数量]，padding后，样本1可能有10个关系，样本2有3个关系，那么样本2就有7个FALSE
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)
        # # 关系分类， 可能的self._max_pairs最大的关系数量， h_large 【batch_size, 关系数量，seq_length, hidden_size]
        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        # 初始一个分类的tensor, [batch_size, 关系数量，关系的label的总数], eg:[1,6, 5]
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # 对关系进行分批次预测。这里叫分块预测
        for i in range(0, relations.shape[1], self._max_pairs):
            #预测关系
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            #对关系logits进行二分类
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            # 预测结果加入
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf
        # 只要mask的关系部分
        rel_clf = rel_clf * rel_sample_masks  # mask

        #对实体进行分类结果进行softmax
        entity_clf = torch.softmax(entity_clf, dim=2)
        # 实体分类的logits，rel_clf关系分类的logits, relations带有每个实体的关系的位置信息
        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        """
        最大池化实体候选跨度, 然后进行对实体分类
        :param encodings: [batch_size, seq_length] eg: [2,26]
        :type encodings:
        :param h: 句子经过bert后的隐藏层向量， [batch_size, seq_length, embedding_size], eg: [2,26,768]
        :type h:
        :param entity_masks: [batch_size, entity_num, seq_length], eg: [2,104,26], 例如entity_num是实体的数量, 包括正样本和负样本
        :type entity_masks:
        :param size_embeddings:  # 对实体的长度的embedding, [batch_size, entity_num实体个数, embedding_size], eg: [2,106,25]
        :type size_embeddings:
        :return:
        :rtype:
        """
        # eg: entity_masks增加一个维度，然盘是否为0， m维度 eg: [2,104,26,1]
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        # 获取实体的向量， [batch_size, entity_num, seq_length, hidden_size], [2,104,26,768]
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        # 对实体跨度进行最大池化, entity_spans_pool: [batch_size, entity_num,hidden_size], eg: [2,104,768]
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # 获得作为候选上下文表示的cls token, entity_ctx: [batch_size, hidden_size]
        entity_ctx = get_token(h, encodings, self._cls_token)

        # 创建候选表示，包括背景、最大集合跨度和尺寸嵌入, 拼接这些表示，
        # entity_ctx 是cls的表示
        # entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1) 表示给每个实体都是复制一个cls
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)
        # 对候选实体进行分类, entity_repr: 【batch_size, entity_num, embedding_size(拼接的向量)]
        # entity_clf, shape: [batch_size, entity_num, entity_labels_num]
        entity_clf = self.entity_classifier(entity_repr)
        # entity_spans_pool: 对实体跨度进行最大池化, entity_spans_pool: [batch_size, entity_num,hidden_size], eg: [2,104,768]
        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        """
         # 对关系候选者进行分类
        :param entity_spans:  entity_spans_pool [batch_size,实体数量，hidden_size]， 代表这个实体的向量
        :type entity_spans:
        :param size_embeddings: size_embeddings实体的长度的embedding [batch_size,实体数量，size_hidden_size]
        :type size_embeddings:
        :param relations: 【batch_size,关系数量，头实体和尾实体的位置索引] 关系的头实体和尾实体在实体向量中的位置索引
        :type relations:
        :param rel_masks: [batch_size,关系数量,padding后的seq_length] 关系在样本中的位置，即2个实体之间的词
        :type rel_masks:
        :param h: 【batch_size, 关系数量，seq_length, hidden_size]
        :type h:
        :param chunk_start: ？？？, eg:0,
        :type chunk_start: int
        :return:
        :rtype:
        """
        batch_size = relations.shape[0]

        # 如果可能关系的数量大于我们预设的最大关系的对数，那么对关系进行拆分
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # 获取实体候选表示对， entity_spans： [batch_size,实体数量，hidden_size]，
        # relations： [batch_size, 关系数量，头实体和尾实体的位置在实体列表中信息]
        # entity_pairs： [batch_size, 关系数量，头实体和尾实体的位置，hidden_size]
        # entity_pairs：每个关系对应的头尾实体的向量
        entity_pairs = util.batch_index(entity_spans, relations)
        # 合并后2个维度, [batch_size, 关系数量，头实体和尾实体的位置 * hidden_size]
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # 每个实体的长度的位置，对应成关系
        # size_embeddings 【batch_size, 实体数量, size_hidden_size】
        # relations： [batch_size, 关系数量，头实体和尾实体的位置在实体列表中信息]
        #  size_pair_embeddings 维度 [batch_size, 关系数量，头实体和尾实体的位置, size_hidden_size], eg: [2,6,2,25]
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        # 合并后2个维度, [batch_size, 关系数量，头实体和尾实体的位置, size_hidden_size]
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # 关系上下文（实体候选对之间的上下文）
        # mask非实体候选token，   rel_masks： [batch_size,关系数量,padding后的seq_length] 关系在样本中的位置，即2个实体之间的词
        # m： [batch_size,关系数量,padding后的seq_length， 1]
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        # h和rel_ctx的形状都是【batch_size, 关系数量，seq_length, hidden_size]，m+h的形状还是m的形状
        rel_ctx = m + h
        #最大池化，rel_ctx: 维度【batch_size,关系数量, hidden_size】
        rel_ctx = rel_ctx.max(dim=2)[0]
        # 将相邻或相邻候选实体的上下文向量设置为零, rel_ctx: 维度【batch_size,关系数量, hidden_size】
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # 创建关系候选表示，包括上下文、最大池化实体候选对和对应的实体size表示的嵌入向量
        # rel_ctx： eg: [2,20,768], entity_pairs: [2,20,1539],  size_pair_embeddings: [2,20,50]
        # rel_repr 是3个向量的拼接， [2,20,2354]， 2354是3个隐藏向量维度+ 2个实体的size的维度
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        #关系分类，
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        """

        :param entity_clf: [batch_size, 实体数量,实体的label总数]
        :type entity_clf:
        :param entity_spans: [batch_size, 实体数量,2】 头尾实体的位置
        :type entity_spans:
        :param entity_sample_masks: [batch_size, 实体数量]
        :type entity_sample_masks:
        :param ctx_size: 序列长度
        :type ctx_size: int
        :return:
        :rtype:
        """
        batch_size = entity_clf.shape[0]
        # entity_logits_max： [batch_size, 实体数量], 找出分类出的实体
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # 将跨度分类为实体， shape为1， eg： tensor([31, 56, 84])， 获取最可能为实体的位置的索引
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            # 找出实体的span，即实体的头尾的位置
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            # 实体的索引也变成列表
            non_zero_indices = non_zero_indices.tolist()

            #创建关系和mask，枚举实体对，把所有实体两两配对
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                #情况1：分类为实体的跨度不超过两个，那么就没有关系，随便创建一个
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # 情况2：分类为实体的两个以上跨度，那么创建关系的tensor
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # 获取设备
        device = self.rel_classifier.weight.device
        # 关系进行padding
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        """
        args的内容，字典格式
            # encodings: [batch_size, padding后的seq_length], token变成id后的内容
            # context_masks: [batch_size, padding后的seq_length】， 样本的实际的长度的mask
            # entity_masks:[batch_size,实体数量，padding后的seq_length] 实体的在句子位置mask
            # entity_sizes: [batch_size,实体数量】 每个实体的长度
            # 'entity_types',[batch_size,,实体数量】
            # 'rels',[batch_size, 关系数量，头实体和尾实体的位置索引】， eg[2,6,2]， 6是包含了正负样本
            # 'rel_masks',[batch_size,关系数量,padding后的seq_length] 关系在样本中的位置，即2个实体之间的词
            # 'rel_types',[batch_size,关系数量，关系的标签总数]  one-hot了的关系
            # 'entity_sample_masks',[batch_size,实体数量]， padding后
            # 'rel_sample_masks'[batch_size,关系数量]，padding后，样本1可能有10个关系，样本2有3个关系，那么样本2就有7个FALSE
        :param args:
        :type args:
        :param inference:
        :type inference:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        if not inference:
            # 训练模式
            return self._forward_train(*args, **kwargs)
        else:
            #推理模式
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]

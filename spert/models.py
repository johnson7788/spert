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

        # 新建分类层
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
        :param encodings: 【batch_size,
        :type encodings:
        :param context_masks:【batch_size,
        :type context_masks:
        :param entity_masks:【batch_size,
        :type entity_masks:
        :param entity_sizes:【batch_size,
        :type entity_sizes:
        :param relations:【batch_size,
        :type relations:
        :param rel_masks:【batch_size,
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
        # entity_spans_pool [ batch_size, entity_num, embedding_size], eg: [2,102,768] 实体跨度中经过了最大池化
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # 关系分类， 可能的self._max_pairs最大的关系数量
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # 获取关系的logits
        # 分块处理以减少内存使用
        for i in range(0, relations.shape[1], self._max_pairs):
            # 对关系候选者进行分类
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        """
        从最后一个transformer层获取上下文的token嵌入
        :param encodings:
        :type encodings:
        :param context_masks:
        :type context_masks:
        :param entity_masks:
        :type entity_masks:
        :param entity_sizes:
        :type entity_sizes:
        :param entity_spans:
        :type entity_spans:
        :param entity_sample_masks:
        :type entity_sample_masks:
        :return:
        :rtype:
        """
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

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

        :param entity_spans:
        :type entity_spans:
        :param size_embeddings:
        :type size_embeddings:
        :param relations:
        :type relations:
        :param rel_masks:
        :type rel_masks:
        :param h:
        :type h:
        :param chunk_start:
        :type chunk_start:
        :return:
        :rtype:
        """
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        """

        :param entity_clf:
        :type entity_clf:
        :param entity_spans:
        :type entity_spans:
        :param entity_sample_masks:
        :type entity_sample_masks:
        :param ctx_size:
        :type ctx_size:
        :return:
        :rtype:
        """
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
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

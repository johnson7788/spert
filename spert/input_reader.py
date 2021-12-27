import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer

from spert import util
from spert.entities import Dataset, EntityType, RelationType, Entity, Relation, Document
from spert.opt import spacy


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None, **kwargs):
        """

        :param types_path:
        :type types_path:
        :param tokenizer:
        :type tokenizer:
        :param neg_entity_count:  实体负样本数量
        :type neg_entity_count: 100
        :param neg_rel_count:  关系负样本数量
        :type neg_rel_count:  100
        :param max_span_size: 最大实体跨度
        :type max_span_size: 10
        :param logger:  日志记录器函数
        :type logger:
        :param kwargs: 其它参数
        :type kwargs:
        """
        # 加载实体和关系的类型文件
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types
        self._entity_types = OrderedDict()
        # 实体id到名称的映射
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # 实体， 给实体加入"无实体类型"， 实体的identifier， 实体的索引的位置，short_name和全名，'No Entity'是全名
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        # 名字到实体实体别名列表的映射
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        #指定实体类型, 都赋值给_entity_types和_idx2entity_type，方便后期使用
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # 关系，和实体一样，加入无关系类型
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # 关系类型的存储
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type
        #负样本
        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger
        # 单词表大小
        self._vocabulary_size = tokenizer.vocab_size

    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        """
        只生成一个读取数据集的实例，未开始真正读取
        :param types_path:
        :type types_path:
        :param tokenizer:
        :type tokenizer:
        :param neg_entity_count:
        :type neg_entity_count:
        :param neg_rel_count:
        :type neg_rel_count:
        :param max_span_size:
        :type max_span_size:
        :param logger:
        :type logger:
        """
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, logger)

    def read(self, dataset_path, dataset_label):
        """
        真正开始读取数据
        :param dataset_path:  'data/datasets/conll04/conll04_train.json'
        :type dataset_path:
        :param dataset_label: 'train'
        :type dataset_label:
        :return:
        :rtype:
        """
        # 初始化一个dataset
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        # dataset赋值， 解析数据集，需要耗时较长
        self._parse_dataset(dataset_path, dataset)
        # daset放到self中，备用， dataset_label: 'train'
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        """
        读取数据集
        :param dataset_path: 'data/datasets/conll04/conll04_train.json'
        :type dataset_path:
        :param dataset: 初始化的dataset格式
        :type dataset:
        :return:
        :rtype:
        """
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="开始读取数据集 '%s'" % dataset.label):
            # 解析每条数据，document代表每个句子
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        """
        解析数据
        :param doc:  从数据集读取的内容
        doc = {dict: 4} {'tokens': ['Newspaper', '`', 'Explains', "'", 'U.S.', 'Interests', 'Section', 'Events', 'FL1402001894', 'Havana', 'Radio', 'Reloj', 'Network', 'in', 'Spanish', '2100', 'GMT', '13', 'Feb', '94'], 'entities': [{'type': 'Loc', 'start': 4, 'end': 5}, {'type':
             'tokens' = {list: 20} ['Newspaper', '`', 'Explains', "'", 'U.S.', 'Interests', 'Section', 'Events', 'FL1402001894', 'Havana', 'Radio', 'Reloj', 'Network', 'in', 'Spanish', '2100', 'GMT', '13', 'Feb', '94']
             'entities' = {list: 5} [{'type': 'Loc', 'start': 4, 'end': 5}, {'type': 'Loc', 'start': 9, 'end': 10}, {'type': 'Org', 'start': 10, 'end': 13}, {'type': 'Other', 'start': 15, 'end': 17}, {'type': 'Other', 'start': 17, 'end': 20}]
             'relations' = {list: 1} [{'type': 'OrgBased_In', 'head': 2, 'tail': 1}]
        :type doc: dict, 包含tokens，entities，relations
        :param dataset:
        :type dataset:
        :return:
        :rtype:
        """
        # 原始数据的token，实体，和关系
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']
        # eg: doc_encoding: [101, 21158, 169, 16409, 18220, 1116, 112, 158, 119, 156, 119, 17067, 1116, 6177, 17437, 23485, 17175, 1568, 10973, 24400, 1604, 1580, 1527, 16092, 2664, 11336, 2858, 3361, 3998, 1107, 2124, 13075, 1568, 14748, 1942, 1492, 13650, 5706, 102]
        # 解析token，和编码，doc_tokens是解析原始数据后，我们需要的token
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)
        # 解析实体提及
        # eg: jentities: [{'type': 'Loc', 'start': 4, 'end': 5}, {'type': 'Loc', 'start': 9, 'end': 10}, {'type': 'Org', 'start': 10, 'end': 13}, {'type': 'Other', 'start': 15, 'end': 17}, {'type': 'Other', 'start': 17, 'end': 20}]
        # eg: doc_tokens: [Newspaper, `, Explains, ', U.S., Interests, Section, Events, FL1402001894, Havana, Radio, Reloj, Network, in, Spanish, 2100, GMT, 13, Feb, 94]
        # eg: entities: 返回文档的实体， 解析后的实体
        entities = self._parse_entities(jentities, doc_tokens, dataset)
        #解析关系， 通过解析出来的实体和原书记讲的关系
        # eg: relations = {list: 1} [<spert.entities.Relation object at 0x113103100>]
        #  0 = {Relation} <spert.entities.Relation object at 0x113103100>
        #   first_entity = {Entity} Havana
        #   head_entity = {Entity} Radio Reloj Network
        #   relation_type = {RelationType} <spert.entities.RelationType object at 0x14b4dae80>
        #   reverse = {bool} True
        #   second_entity = {Entity} Radio Reloj Network
        #   tail_entity = {Entity} Havana
        # 解析后的关系
        relations = self._parse_relations(jrelations, entities, dataset)
        #document : document = {Document} <spert.entities.Document object at 0x118fc3e80>
             # doc_id = {int} 0
             # encoding = {list: 39} [101, 21158, 169, 16409, 18220, 1116, 112, 158, 119, 156, 119, 17067, 1116, 6177, 17437, 23485, 17175, 1568, 10973, 24400, 1604, 1580, 1527, 16092, 2664, 11336, 2858, 3361, 3998, 1107, 2124, 13075, 1568, 14748, 1942, 1492, 13650, 5706, 102]
             # entities = {list: 5} [<spert.entities.Entity object at 0x17586c190>, <spert.entities.Entity object at 0x17586c070>, <spert.entities.Entity object at 0x17586c1c0>, <spert.entities.Entity object at 0x17586c220>, <spert.entities.Entity object at 0x17586c280>]
             # relations = {list: 1} [<spert.entities.Relation object at 0x17586c2e0>]
             # tokens = {TokenSpan: 20} <spert.entities.TokenSpan object at 0x177a60550>
        #加入token，实体，关系到数据集中
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)

        return document

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        """
        把原始的json格式的实体，解析成我们需要的格式
        :param jentities:  原始的json格式的实体格式的列表
        :type jentities:
        :param doc_tokens: 原始的tokens
        :type doc_tokens:
        :param dataset: 数据集实例
        :type dataset:
        :return:
        :rtype:
        """
        entities = []
        for entity_idx, jentity in enumerate(jentities):
            # eg: entity_type = {EntityType} <spert.entities.EntityType object at 0x165180220>
            #  identifier = {str} 'Loc'
            #  index = {int} 1
            #  short_name = {str} 'Loc'
            #  verbose_name = {str} 'Location'

            # eg: jentity: {'type': 'Loc', 'start': 4, 'end': 5}
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            #创建实体提及: eg: tokens: [U.S.]
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            # entity: U.S.
            entities.append(entity)
        # eg: entities: list
        #      0 = {Entity} U.S.
        #      1 = {Entity} Havana
        #      2 = {Entity} Radio Reloj Network
        #      3 = {Entity} 2100 GMT
        #      4 = {Entity} 13 Feb 94
        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        """

        :param jrelations:
        :type jrelations:
        :param entities:
        :type entities:
        :param dataset:
        :type dataset:
        :return:
        :rtype:
        """
        relations = []
        for jrelation in jrelations:
            # eg: jrelation: {'type': 'OrgBased_In', 'head': 2, 'tail': 1}
            # eg: relation_type = {RelationType} <spert.entities.RelationType object at 0x13f9ace80>
            #  identifier = {str} 'OrgBased_In'
            #  index = {int} 3
            #  short_name = {str} 'OrgBI'
            #  symmetric = {bool} False
            #  verbose_name = {str} 'Organization based in'
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            #创建关系
            head = entities[head_idx]
            tail = entities[tail_idx]
            # 判断头实体和尾实体的位置，谁在前面，谁作为头实体
            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # 对称关系：头在句子中出现在尾之前
            if relation_type.symmetric and reverse:
                # 交换头尾实体
                head, tail = util.swap(head, tail)
            # 创建一个关系
            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)
        # 所有关系
        return relations


class JsonPredictionInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, spacy_model: str = None,
                 max_span_size: int = None, logger: Logger = None):
        """
        解析源数据
        :param types_path:
        :type types_path:
        :param tokenizer:
        :type tokenizer:
        :param spacy_model:  en_core_web_sm
        :type spacy_model:
        :param max_span_size:
        :type max_span_size:
        :param logger:
        :type logger:
        """
        super().__init__(types_path, tokenizer, max_span_size=max_span_size, logger=logger)
        self._spacy_model = spacy_model
        self._nlp = spacy.load(spacy_model) if spacy is not None and spacy_model is not None else None

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset) -> Document:
        """

        :param document:
        :type document:
        :param dataset:
        :type dataset:
        :return:
        :rtype:
        """
        if type(document) == list:
            jtokens = document
        elif type(document) == dict:
            jtokens = document['tokens']
        else:
            jtokens = [t.text for t in self._nlp(document)]

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # create document
        document = dataset.create_document(doc_tokens, [], [], doc_encoding)

        return document


def _parse_tokens(jtokens, dataset, tokenizer):
    """

    :param jtokens: 每个tokens, eg: ['Newspaper', '`', 'Explains', "'", 'U.S.', 'Interests', 'Section', 'Events', 'FL1402001894', 'Havana', 'Radio', 'Reloj', 'Network', 'in', 'Spanish', '2100', 'GMT', '13', 'Feb', '94']
    :type jtokens:
    :param dataset:
    :type dataset:
    :param tokenizer: tokenizer
    :type tokenizer:
    :return:
    :rtype:
    """
    doc_tokens = []

    # 全文档编码，包括特殊token（[CLS]和[SEP]）和原始token的字节对编码 ，doc_encoding是CLS
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

    # parse tokens
    for i, token_phrase in enumerate(jtokens):
        # token变成id, token_encoding: [21158]
        token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
        if not token_encoding:
            # 如果发现token_encoding返回为空，那么这个token为UNK
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
        # span的开始和结束, (CLS, CLS_ID, tokenid), 每个token可能被tokenizer多个，例如一个英文单词encode后变成多个id
        span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
        # 创建一个token的实例
        token = dataset.create_token(i, span_start, span_end, token_phrase)
        # 完成一个token， 放到doc_tokens中保存
        doc_tokens.append(token)
        doc_encoding += token_encoding
    #解析完成后，加上一个SEP结束
    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]
    # eg: doc_encoding: [101, 21158, 169, 16409, 18220, 1116, 112, 158, 119, 156, 119, 17067, 1116, 6177, 17437, 23485, 17175, 1568, 10973, 24400, 1604, 1580, 1527, 16092, 2664, 11336, 2858, 3361, 3998, 1107, 2124, 13075, 1568, 14748, 1942, 1492, 13650, 5706, 102]
    return doc_tokens, doc_encoding

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/12/29 9:15 下午
# @File  : generate_data.py
# @Author: johnson
# @Desc  :  生成数据集
import sys
import random
import os
import json
sys.path.append('/Users/admin/git/label-studio/myexample')  # 获取label原数据的路径
from convert_labeled_data import get_shenji_entity_relation  # 获取label处理好的数据

def get_data():
    data = []
    relations_names = []
    entity_names = []
    source_data = get_shenji_entity_relation(dirpath="/opt/ai/kaiyuan")
    for one_data in source_data:
        text = one_data["text"]
        tokens = list(text)
        entities = []
        entities_ids = []
        source_entities = one_data["entities"]
        for entity in source_entities:
            entity_type = entity["labels"]
            if entity_type not in entity_names:
                entity_names.append(entity_type)
            start = entity["start"]
            end = entity["end"]
            id = entity["id"]
            one_entity = {
                "type": entity_type,
                "start": start,
                "end": end
            }
            entities.append(one_entity)
            entities_ids.append(id)
        relations = []
        for one_rel in one_data["relations"]:
            head_id = one_rel["head"]["id"]
            head_location = entities_ids.index(head_id)
            tail_id = one_rel["tail"]["id"]
            tail_location = entities_ids.index(tail_id)
            rel_type = one_rel["label"]
            if rel_type not in relations_names:
                relations_names.append(rel_type)
            relation = {
                "type": rel_type,
                "head": head_location,  # 头实体在实体列表中的位置
                "tail": tail_location  # 尾实体在实体列表中的位置
            }
            relations.append(relation)
        one = {
            "tokens": tokens,
            "entities": entities,
            "relations": relations
        }
        data.append(one)
    print(f"共收集到数据: {len(data)}条")
    return data, entity_names, relations_names

def save_to(s_data, path):
    with open(path, 'w') as f:
        json.dump(s_data, f, ensure_ascii=False, indent=2)

def split_save_data(random_seed=30, train_rate=0.8, dev_rate=0.1, test_rate=0.1):
    """
    :param data:
    :type data:
    :param random_seed:
    :type random_seed:
    :param train_rate:
    :type train_rate:
    :param dev_rate:
    :type dev_rate:
    :param test_rate:
    :type test_rate:
    :return:
    :rtype:
    """
    data, entity_names, relations_names = get_data()
    save_dir = "data/datasets/kaiyuan"
    type_file = os.path.join(save_dir, "kaiyuan_types.json")
    type_data = {}
    entities = {}
    for entity in entity_names:
        one = {
            "short": entity,
            "verbose": entity
        }
        entities[entity] = one
    type_data["entities"] = entities
    relations = {}
    for relation in relations_names:
        one = {
            "short": relation,
            "verbose": relation,
            "symmetric": False,
        }
        relations[relation] = one
    type_data["relations"] = relations
    save_to(type_data, type_file)
    random.seed(random_seed)
    # 可以通过id找到对应的源数据
    data_id = list(range(len(data)))
    random.shuffle(data_id)
    total = len(data)
    train_num = int(total * train_rate)
    dev_num = int(total * dev_rate)
    test_num = int(total * test_rate)
    train_data_id = data_id[:train_num]
    dev_data_id = data_id[train_num:train_num + dev_num]
    test_data_id = data_id[train_num + dev_num:]
    train_data = [data[id] for id in train_data_id]
    dev_data = [data[id] for id in dev_data_id]
    test_data = [data[id] for id in test_data_id]
    print(f"训练集数量{len(train_data)}, 开发集数量{len(dev_data)}, 测试集数量{len(test_data)}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_path = os.path.join(save_dir, "kaiyuan_train.json")
    dev_path = os.path.join(save_dir, "kaiyuan_dev.json")
    test_path = os.path.join(save_dir, "kaiyuan_test.json")
    save_to(s_data=train_data, path=train_path)
    save_to(s_data=dev_data, path=dev_path)
    save_to(s_data=test_data, path=test_path)
    return train_data, dev_data, test_data, train_data_id, dev_data_id, test_data_id

if __name__ == '__main__':
    split_save_data()

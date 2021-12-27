# 配置文件
```angular2html
example_eval.conf  配置文件的第一行是迭代的次数
example_predict.conf
example_train.conf
```
# 训练模型
```angular2html
spert.py train --config configs/example_train.conf --debug

2021-12-27 14:15:29,664 [MainThread  ] [INFO ]  开始训练第: 19个 epoch
训练 epoch 19: 100%|██████████████████████████| 461/461 [00:30<00:00, 15.21it/s]
2021-12-27 14:15:59,981 [MainThread  ] [INFO ]  开始评估: valid
Evaluate epoch 20: 100%|██████████████████████| 231/231 [00:12<00:00, 18.06it/s]
Evaluation

--- Entities (named entity recognition (NER)) ---
An entity is considered correct if the entity type and span is predicted correctly

                type    precision       recall     f1-score      support
                Peop        94.43        95.76        95.09          283
                 Org        83.13        81.18        82.14          170
                 Loc        88.75        90.68        89.71          322
               Other        82.29        66.95        73.83          118

               micro        88.84        87.35        88.09          893
               macro        87.15        83.64        85.19          893

--- Relations ---

Without named entity classification (NEC)
A relation is considered correct if the relation type and the spans of the two related entities are predicted correctly (entity type is not considered)

                type    precision       recall     f1-score      support
               LocIn        79.66        72.31        75.81           65
                Live        68.48        69.23        68.85           91
                Work        77.97        66.67        71.88           69
               OrgBI        64.62        55.26        59.57           76
                Kill        77.55        90.48        83.52           42

               micro        72.84        68.80        70.76          343
               macro        73.65        70.79        71.92          343

With named entity classification (NEC)
A relation is considered correct if the relation type and the two related entities are predicted correctly (in span and entity type)

                type    precision       recall     f1-score      support
               LocIn        79.66        72.31        75.81           65
                Live        68.48        69.23        68.85           91
                Work        77.97        66.67        71.88           69
               OrgBI        64.62        55.26        59.57           76
                Kill        77.55        90.48        83.52           42

               micro        72.84        68.80        70.76          343
               macro        73.65        70.79        71.92          343
2021-12-27 14:16:14,645 [MainThread  ] [INFO ]  日志保存至: data/log/conll04_train/2021-12-27_14:05:39.241011
2021-12-27 14:16:14,645 [MainThread  ] [INFO ]  模型保存至: data/save/conll04_train/2021-12-27_14:05:39.241011

Process finished with exit code 0

```

# 评估
```
spert.py eval --config configs/example_eval.conf
2021-12-27 14:26:12,778 [MainThread  ] [INFO ]  开始评估数据集: test
Evaluate epoch 0: 100%|███████████████████████| 288/288 [00:35<00:00,  8.02it/s]
评估结果:
--- 实体 (named entity recognition (NER)) ---
如果正确预测了实体类型和跨度，则认为该实体是正确的。

                type    precision       recall     f1-score      support
                 Loc        92.16        90.87        91.51          427
                Peop        95.09        96.57        95.83          321
               Other        79.20        74.44        76.74          133
                 Org        82.18        83.84        83.00          198

               micro        89.66        89.25        89.46         1079
               macro        87.16        86.43        86.77         1079

--- 关系评估1 Without named entity classification (NEC)---
如果正确预测了两个相关实体的关系类型和跨度，则认为关系是正确的（不考虑实体类型）

                type    precision       recall     f1-score      support
               OrgBI        73.47        68.57        70.94          105
               LocIn        74.36        61.70        67.44           94
                Work        67.95        69.74        68.83           76
                Live        76.24        77.00        76.62          100
                Kill        87.23        87.23        87.23           47

               micro        74.88        71.33        73.06          422
               macro        75.85        72.85        74.21          422

--- 关系评估2 With named entity classification (NEC)---
如果正确预测了关系类型和两个相关实体（在跨度和实体类型中），则认为关系是正确的

                type    precision       recall     f1-score      support
               OrgBI        73.47        68.57        70.94          105
               LocIn        73.08        60.64        66.28           94
                Work        67.95        69.74        68.83           76
                Live        76.24        77.00        76.62          100
                Kill        87.23        87.23        87.23           47

               micro        74.63        71.09        72.82          422
               macro        75.59        72.64        73.98          422
2021-12-27 14:26:54,536 [MainThread  ] [INFO ]  Logged in: data/log/conll04_eval/2021-12-27_14:26:07.430781
```

# 预测, 注意，需要下载spacy模型: python -m spacy download en_core_web_sm
```
spert.py predict --config configs/example_predict.conf
--------------------------------------------------
配置如下:
需要迭代 1 次
--------------------------------------------------
迭代 0
--------------------------------------------------
Parse dataset 'dataset': 100%|██████████| 3/3 [00:00<00:00, 211.76it/s]
Predict: 100%|██████████| 3/3 [00:18<00:00,  6.11s/it]
```

# 目录结构
```angular2html
├── args.py    所有的参数
├── config_reader.py    参数整理
├── configs   预设的配置文件
│   ├── example_eval.conf     评估的配置
│   ├── example_predict.conf
│   └── example_train.conf  训练的配置
├── data
│   ├── datasets
│   │   ├── ade  数据集
│   │   ├── conll04
│   │   │   ├── conll04_dev.json
│   │   │   ├── conll04_prediction_example.json
│   │   │   ├── conll04_test.json
│   │   │   ├── conll04_train.json
│   │   │   ├── conll04_train_dev.json
│   │   │   └── conll04_types.json  保存实体和关系的类型信息
│   │   └── scierc
│   ├── log 保存日志
│   ├── models  预训练模型
│   │   ├── ade
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   └── vocab.txt
│   │   ├── conll04
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   └── vocab.txt
│   │   └── scierc
│   │       ├── config.json
│   │       ├── pytorch_model.bin
│   │       └── vocab.txt
│   ├── predictions.json
│   └── save  训练完成保存的模型
├── scripts
│   ├── conversion
│   │   ├── convert_ade.py
│   │   ├── convert_conll04.py
│   │   └── convert_scierc.py
│   ├── fetch_datasets.sh  下载数据集
│   └── fetch_models.sh  下载模型
├── spert
│   ├── entities.py 实体，关系类型的定义
│   ├── evaluator.py  评估函数
│   ├── input_reader.py  读取数据集
│   ├── loss.py   损失
│   ├── models.py  模型
│   ├── opt.py  没啥用
│   ├── prediction.py  预测处理程序
│   ├── sampling.py  样本操作程序
│   ├── spert_trainer.py 训练模型
│   ├── templates
│   │   ├── entity_examples.html
│   │   └── relation_examples.html
│   ├── trainer.py
│   └── util.py
└── spert.py  主程序
```

# 运行流程
spert.py：__train: 初始化模型类SpERTTrainer，初始化模型保存路径，日志路径，日志级别，加载self._tokenizer等


# 中文的数据集测试

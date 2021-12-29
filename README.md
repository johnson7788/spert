# SpERT: Span-based Entity and Relation Transformer
PyTorch code for SpERT: "Span-based Entity and Relation Transformer". For a description of the model and experiments, see our paper: https://arxiv.org/abs/1909.07755 (published at ECAI 2020).

![alt text](http://deepca.cs.hs-rm.de/img/deepca/spert.png)

## 安装
### 依赖
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

### 获取数据
获取转换后的（特定的JSON格式）CoNLL04\[1\]（我们使用与\[4\]相同的分割）、SciERC\[2\]和ADE\[3\]数据集（原始数据集见参考文件）。
```
bash ./scripts/fetch_datasets.sh
一条ade数据
{
  "tokens": [
    "A",
    "case",
    "of",
    "a",
    "53-year",
    "-",
    "old",
    "man",
    "who",
    "developed",
    "acute",
    "pneumonitis",
    "after",
    "bleomycin",
    "and",
    "moderate",
    "oxygen",
    "administration",
    "is",
    "presented",
    "."
  ],
  "entities": [
    {
      "type": "Adverse-Effect",
      "start": 10,
      "end": 12
    },
    {
      "type": "Drug",
      "start": 13,
      "end": 14
    },
    {
      "type": "Drug",
      "start": 16,
      "end": 17
    }
  ],
  "relations": [
    {
      "type": "Adverse-Effect",
      "head": 0,   #头实体在实体列表中的位置
      "tail": 1    #尾实体在实体列表中的位置
    },
    {
      "type": "Adverse-Effect",
      "head": 0,
      "tail": 2
    }
  ],
  "orig_id": 2598
}
```

取出模型checkpoint（每个数据集的5次运行中的最好结果）:
```
bash ./scripts/fetch_models.sh
```
所附的ADE模型是在 "data/datasets/ade "下的split "1"（"ade_split_1_train.json" / "ade_split_1_test.json"）上训练的。

## 示例
(1) 在训练数据集上训练CoNLL04，在开发数据集上评估。
```
python ./spert.py train --config configs/example_train.conf
```

(2) 在测试数据集上评估CoNLL04模型。
```
python ./spert.py eval --config configs/example_eval.conf
```

(3) 使用CoNLL04模型进行预测。支持的数据格式见文件'data/datasets/conll04/conll04_prediction_example.json'。你有三个选项来指定输入的句子，请选择适合你的需求的选项。如果数据集包含原始句子，必须安装'spacy'来进行tokenization。通过'python -m spacy download model_label'下载一个spacy模型，并在配置文件中设置为spacy_model（见'configs/example_predict.conf'）。
```
python ./spert.py predict --config configs/example_predict.conf
```
## 实验结果的复现
- 最终的模型是在train+dev的组合数据集上训练的（例如'conll04_train_dev.json'）。
- 复制SciERC的结果。为了增加一个特征，需要在提交[7b27b7d](https://github.com/lavis-nlp/spert/commit/7b27b7d258d0b4bb44103b9d0f9e19f2ce08611f)中改变负对称关系的采样。这导致了SciERC的实验结果略有改善。如果你想完全复制ECAI 2020的论文结果，请使用提交[3f4ab22](https://github.com/lavis-nlp/spert/commit/3f4ab22857f9ca0d96b582084a2a0ceb3e9826f9)。


## 额外说明
- 为了用SciBERT训练SpERT，[5\]从https://github.com/allenai/scibert（在 "PyTorch HuggingFace Models "下）下载SciBERT，并在配置文件中设置 "model_path "和 "tokenizer_path "以指向SciBERT目录。
- 如果模型预测了许多错误的正面实体提及，可以尝试增加负面实体样本的数量（配置文件中的'neg_entity_count'）。
- 你可以调用 "python ./spert.py train --help" / "python ./spert.py eval --help" "python ./spert.py predict --help "来了解训练/评估/预测的参数描述。
- Please cite our paper when you use SpERT: <br/>
```
Markus Eberts, Adrian Ulges. Span-based Joint Entity and Relation Extraction with Transformer Pre-training. 24th European Conference on Artificial Intelligence, 2020.
```

## References
```
[1] Dan Roth and Wen-tau Yih, ‘A Linear Programming Formulation forGlobal Inference in Natural Language Tasks’, in Proc. of CoNLL 2004 at HLT-NAACL 2004, pp. 1–8, Boston, Massachusetts, USA, (May 6 -May 7 2004). ACL.
[2] Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi, ‘Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction’, in Proc. of EMNLP 2018, pp. 3219–3232, Brussels, Belgium, (October-November 2018). ACL.
[3] Harsha Gurulingappa, Abdul Mateen Rajput, Angus Roberts, JulianeFluck,  Martin  Hofmann-Apitius,  and  Luca  Toldo,  ‘Development  of a  Benchmark  Corpus  to  Support  the  Automatic  Extraction  of  Drug-related Adverse Effects from Medical Case Reports’, J. of BiomedicalInformatics,45(5), 885–892, (October 2012).
[4] Pankaj Gupta,  Hinrich Schütze, and Bernt Andrassy, ‘Table Filling Multi-Task Recurrent  Neural  Network  for  Joint  Entity  and  Relation Extraction’, in Proc. of COLING 2016, pp. 2537–2547, Osaka, Japan, (December 2016). The COLING 2016 Organizing Committee.
[5] Iz Beltagy, Kyle Lo, and Arman Cohan, ‘SciBERT: A Pretrained Language Model for Scientific Text’, in EMNLP, (2019).
```



{
  "tokens": [
    "John",
    "Wilkes",
    "Booth",
    ",",
    "who",
    "assassinated",
    "President",
    "Lincoln",
    ",",
    "was",
    "an",
    "actor",
    "."
  ],
  "entities": [
    {
      "type": "Peop",
      "start": 0,
      "end": 3
    },
    {
      "type": "Peop",
      "start": 6,
      "end": 8
    }
  ],
  "relations": [
    {
      "type": "Kill",
      "head": 0,
      "tail": 1
    }
  ],
  "orig_id": 5178
}
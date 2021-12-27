import argparse

def _add_common_args(arg_parser):
    """
    常规参数
    :param arg_parser:
    :type arg_parser:
    :return:
    :rtype:
    """
    arg_parser.add_argument('--config', type=str, help="指定配置文件")
    # Input
    arg_parser.add_argument('--types_path', type=str, help="通往类型规格的路径")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="tokenizer的路径")
    arg_parser.add_argument('--max_span_size', type=int, default=10, help="span跨度的最大大小")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="如果为真，在预处理过程中，输入被小写")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="采样进程的数量。=0表示没有取样的多进程")

    # Model / Training / Evaluation
    arg_parser.add_argument('--model_path', type=str, help="包含模型checkpoint的目录的路径")
    arg_parser.add_argument('--model_type', type=str, default="spert", help="模型的类型")
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="如果为真，即使有CUDA设备，也在CPU上进行训练/评估。")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="评估/预测批次大小")
    arg_parser.add_argument('--max_pairs', type=int, default=1000,
                            help="训练/评估期间要处理的最大实体对")
    arg_parser.add_argument('--rel_filter_threshold', type=float, default=0.4, help="关系的过滤阈值")
    arg_parser.add_argument('--size_embedding', type=int, default=25, help="维度嵌入的维度")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="SpERT中使用的dropout概率")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="是否冻结 BERT 权重")
    arg_parser.add_argument('--no_overlapping', action='store_true', default=False,
                            help="如果为真，则不对重叠的实体和有重叠的实体的关系进行评估")

    # Misc
    arg_parser.add_argument('--seed', type=int, default=None, help="随机数种子")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="缓存transformer模型的路径（用于HuggingFacetransformer库）。")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="debugging模式开/关")


def _add_logging_args(arg_parser):
    arg_parser.add_argument('--label', type=str, help="运行的标签。用作日志/模型的目录名称")
    arg_parser.add_argument('--log_path', type=str, help="储存训练/评估日志的目录的路径")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,
                            help="如果为真，将预测结果存储在磁盘上（在日志目录中）。")
    arg_parser.add_argument('--store_examples', action='store_true', default=False,
                            help="如果为真，将评估实例存储在磁盘上（在日志目录中）。")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="要存储的评估实例的数量（如果store_examples == True）。")


def train_argparser():
    """
    训练的参数
    :return:
    :rtype:
    """
    arg_parser = argparse.ArgumentParser()
    # Input
    arg_parser.add_argument('--train_path', type=str, help="训练数据的路径")
    arg_parser.add_argument('--valid_path', type=str, help="验证集路径")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="存储模型checkpoint的目录的路径")
    arg_parser.add_argument('--init_eval', action='store_true', default=False,
                            help="如果为真，在训练前评估验证集")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,
                            help="将优化器与模型一起保存")
    arg_parser.add_argument('--train_log_iter', type=int, default=100, help="每x次迭代记录训练过程")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,
                            help="只在训练后评估模型，而不是在每个epoch都评估")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=2, help="训练批次大小")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--neg_entity_count', type=int, default=100,
                            help="每份文件（句子）的负面实体样本数。")
    arg_parser.add_argument('--neg_relation_count', type=int, default=100,
                            help="每份文件（句子）的负面关系样本数。")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="在线性增加/减少计划中，warmup训练占总训练迭代的比例")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")

    _add_common_args(arg_parser)
    _add_logging_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()
    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="评估的数据集")
    _add_common_args(arg_parser)
    _add_logging_args(arg_parser)

    return arg_parser


def predict_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")
    arg_parser.add_argument('--predictions_path', type=str, help="Path to store predictions")
    arg_parser.add_argument('--spacy_model', type=str, help="Label of SpaCy model (used for tokenization)")

    _add_common_args(arg_parser)

    return arg_parser

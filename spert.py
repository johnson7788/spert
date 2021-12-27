import argparse
from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.spert_trainer import SpERTTrainer


def _train():
    arg_parser = train_argparser()
    #解析配置
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    """
    真正进行训练的配置
    :param run_args: 命令行和配置文件的参数 Namespace(cache_path=None, config='configs/example_train.conf', cpu=False, debug=True, epochs=20, eval_batch_size=1, example_count=None, final_eval=True, freeze_transformer=False, init_eval=False, label='conll04_train', log_path='data/log/', lowercase=False, lr=5e-05, lr_warmup=0.1, max_grad_norm=1.0, max_pairs=1000, max_span_size=10, model_path='bert-base-cased', model_type='spert', neg_entity_count=100, neg_relation_count=100, no_overlapping=False, prop_drop=0.1, rel_filter_threshold=0.4, sampling_processes=4, save_optimizer=False, save_path='data/save/', seed=None, size_embedding=25, store_examples=True, store_predictions=True, tokenizer_path='bert-base-cased', train_batch_size=2, train_log_iter=100, train_path='data/datasets/conll04/conll04_train.json', types_path='data/datasets/conll04/conll04_types.json', valid_path='data/datasets/conll04/conll04_dev.json', weight_decay=0.01)
    :type run_args:
    :return:
    :rtype:
    """
    # 初始化
    trainer = SpERTTrainer(run_args)
    # 训练
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _predict():
    arg_parser = predict_argparser()
    process_configs(target=__predict, arg_parser=arg_parser)


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="实体关系联合抽取")
    arg_parser.add_argument('mode', type=str, help="选择train, eval,还是predict")
    args, _ = arg_parser.parse_known_args()
    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'predict':
        _predict()
    else:
        raise Exception("必须三选一 ['train', 'eval', 'predict'], e.g. 'python spert.py train ...'")

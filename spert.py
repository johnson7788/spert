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
    :param run_args:
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
        raise Exception("Mode not in ['train', 'eval', 'predict'], e.g. 'python spert.py train ...'")

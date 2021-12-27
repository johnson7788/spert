import copy
import multiprocessing as mp


def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    # 多进程
    ctx = mp.get_context('spawn')

    for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
        p = ctx.Process(target=target, args=(run_args,))
        p.start()
        p.join()


def _read_config(path):
    """
    解析配置文件
    :param path:  'configs/example_train.conf'
    :type path:
    :return: list， 配置列表
    :rtype:
    """
    lines = open(path).readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    """
    变成命令格式的参数
    :param config:
    :type config:
    :return:
    :rtype:
    """
    config_list = []
    for k, v in config.items():
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))

    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    """
    返回配置
    :param arg_parser:
    :type arg_parser:
    :param args:
    :type args:
    :param verbose:
    :type verbose:
    :return:
    :rtype:
    """
    _print = (lambda x: print(x)) if verbose else lambda x: x
    if args.config:
        config = _read_config(args.config)

        for run_repeat, run_config in config:
            print("-" * 50)
            print("配置如下:")
            (print(f"{k}: {v}") for k, v in run_config.items())

            args_copy = copy.deepcopy(args)
            config_list = _convert_config(run_config)
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)
            run_args_dict = vars(run_args)

            # set boolean values
            for k, v in run_config.items():
                if v.lower() == 'false':
                    run_args_dict[k] = False

            print("需要迭代 %s 次" % run_repeat)
            print("-" * 50)

            for iteration in range(run_repeat):
                _print("迭代 %s" % iteration)
                _print("-" * 50)

                yield run_args, run_config, run_repeat

    else:
        yield args, None, None

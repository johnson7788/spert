import csv
import json
import os
import random
import shutil

import numpy as np
import torch

from spert.entities import TokenSpan

CSV_DELIMETER = ';'


def create_directories_file(f):
    d = os.path.dirname(f)

    if d and not os.path.exists(d):
        os.makedirs(d)

    return f


def create_directories_dir(d):
    """
    创建一个目录
    :param d:
    :type d:
    :return:
    :rtype:
    """
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d


def create_csv(file_path, *column_names):
    """
    创建csv文件, 写入表头
    :param file_path:  'data/log/conll04_train/2021-12-27_16:11:32.297274/lr_train.csv'
    :type file_path:
    :param column_names:   ('lr', 'epoch', 'iteration', 'global_iteration')
    :type column_names:
    :return:
    :rtype:
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if column_names:
                writer.writerow(column_names)


def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def append_csv_multiple(file_path, *rows):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)


def read_csv(file_path):
    lines = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)

    return lines[0], lines[1:]


def copy_python_directory(source, dest, ignore_dirs=None):
    source = source if source.endswith('/') else source + '/'
    for (dir_path, dir_names, file_names) in os.walk(source):
        tail = '/'.join(dir_path.split(source)[1:])
        new_dir = os.path.join(dest, tail)

        if ignore_dirs and True in [(ignore_dir in tail) for ignore_dir in ignore_dirs]:
            continue

        create_directories_dir(new_dir)

        for file_name in file_names:
            if file_name.endswith('.py'):
                file_path = os.path.join(dir_path, file_name)
                shutil.copy2(file_path, new_dir)


def save_dict(log_path, dic, name):
    # save arguments
    # 1. as json
    path = os.path.join(log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f)
    f.close()

    # 2. as string
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()


def summarize_dict(summary_writer, dic, name):
    table = 'Argument|Value\n-|-'

    for k, v in vars(dic).items():
        row = '\n%s|%s' % (k, v)
        table += row
    summary_writer.add_text(name, table)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)


def flatten(l):
    return [i for p in l for i in p]


def get_as_list(dic, key):
    if key in dic:
        return [dic[key]]
    else:
        return []


def extend_tensor(tensor, extended_shape, fill=0):
    """
    对tensor向量进行扩充，即padding
    :param tensor:
    :type tensor:
    :param extended_shape:  [30]， 填充的最大维度
    :type extended_shape: list
    :param fill: 使用0进行padding
    :type fill:
    :return:
    :rtype:
    """
    # eg: tensor_shape: 34
    tensor_shape = tensor.shape
    # 创建一个相同的shape的tensor，用padding的内容填充，这里是0填充
    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)
    # 根据维度的不同，把原tensor填充进来
    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    """
    对样本进行padding， padding后进行stack
    :param tensors: list 格式，里面是长度不等的tensor
    :type tensors:
    :param padding: 使用0进行padding
    :type padding:
    :return:
    :rtype:
    """
    # 确定数据维度个数
    dim_count = len(tensors[0].shape)
    # 确定最后一维的维度的最大数量
    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        # 对每个数据进行padding
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)
    # padding后的数据进行stack
    stacked = torch.stack(padded_tensors)
    return stacked


def batch_index(tensor, index, pad=False):
    """
    根据给定的index检索给定的tensor
    :param tensor:
    :type tensor:
    :param index:
    :type index:
    :param pad:
    :type pad:
    :return:
    :rtype:
    """
    if tensor.shape[0] != index.shape[0]:
        raise Exception(f"第一个维度不相同，请检查")

    if not pad:
        # 返回的维度[batch_size, 关系数量，头实体和尾实体的位置，hidden_size]
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def padded_nonzero(tensor, padding=0):
    indices = padded_stack([tensor[i].nonzero().view(-1) for i in range(tensor.shape[0])], padding)
    return indices


def swap(v1, v2):
    return v2, v1


def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        if t.span[0] == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.span[1] == span[1]:
            return TokenSpan(span_tokens)

    return None


def to_device(batch, device):
    """
    batch中的每个元素都放到device上
    :param batch:
    :type batch:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def check_version(config, model_class, model_path):
    """

    :param config:
    :type config:
    :param model_class:
    :type model_class:
    :param model_path: 如果是本地模型，检查本地模型的配置， 'bert-base-cased'
    :type model_path:
    :return:
    :rtype:
    """
    if os.path.exists(model_path):
        model_path = model_path if model_path.endswith('.bin') else os.path.join(model_path, 'pytorch_model.bin')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        config_dict = config.to_dict()

        # version check
        loaded_version = config_dict.get('spert_version', '1.0')
        if 'rel_classifier.weight' in state_dict and loaded_version != model_class.VERSION:
            msg = ("Current SpERT version (%s) does not match the version of the loaded model (%s). "
                   % (model_class.VERSION, loaded_version))
            msg += "Use the code matching your version or train a new model."
            raise Exception(msg)

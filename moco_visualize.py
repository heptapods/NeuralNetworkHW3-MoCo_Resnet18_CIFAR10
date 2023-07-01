import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

moco_record_dir = './moco-record'
moco_pretrain_txt = os.path.join(moco_record_dir, 'moco-pretrain.txt')
moco_lincls_txt = os.path.join(moco_record_dir, 'moco-lincls.txt')


def parse_pretrain_line(line):
    line = line.strip()
    line_parts = line.split(", ")
    info = {}
    info["epoch"] = line_parts[0].split()
    for part in line_parts[1:]:
        key, value = part.split(" ", 1)
        value = value.split(" (")[0]
        value = float(value) if '.' in value else int(value)
        info[key] = value
    return info

def parse_lincls_line(line):
    line = line.strip()
    line_parts = line.split("\t")
    info = {}
    info["epoch"] = line_parts[0].split()
    for part in line_parts[1:]:
        key, value = part.split(" ", 1)
        value = value.split(" (")[0]
        value = float(value) if '.' in value else int(value)
        info[key] = value
    return info

def parse_test_acc_line(line):
    line = line.strip().split(" ")
    info = {}
    for i in range(1, len(line), 2):
        key = line[i].strip()
        value = float(line[i + 1])
        info[key] = value
    return info


def get_pretrain_record(pretrain_record_file, pretrain_num_round):
    pretrain_dicts = []
    with open(pretrain_record_file, 'r') as f:
        line = f.readline()
        epoch = 0
        round = 0
        while line:
            info = parse_pretrain_line(line)
            info['epoch'] = epoch
            info['round'] = round
            pretrain_dicts.append(info)
            line = f.readline()
            round += 10
            if round > pretrain_num_round:
                round = 0
                epoch += 1
        return pd.DataFrame.from_dict(pretrain_dicts)

def get_lincls_record(lincls_record_file, lincls_num_round):
    lincls_dicts = []
    test_acc_dicts = []
    with open(lincls_record_file, 'r') as f:
        line = f.readline()
        epoch = 0
        round = 0
        while line:
            line = line.strip()
            print(line)
            if not line[0] == '*':
                info = parse_lincls_line(line)
                info['epoch'] = epoch
                info['round'] = round
                lincls_dicts.append(info)
                line = f.readline()
                round += 10
                if round > lincls_num_round:
                    round = 0
                    epoch += 1
            else:
                info = parse_test_acc_line(line)
                line = f.readline()
                test_acc_dicts.append(info)
        return (pd.DataFrame.from_dict(lincls_dicts), pd.DataFrame.from_dict(test_acc_dicts))


if __name__ == '__main__':
    pretrain_record_df = get_pretrain_record(moco_pretrain_txt, pretrain_num_round=195)
    pretrain_loss = pretrain_record_df.loc[1:, 'Loss']
    iter = (1+np.arange(len(pretrain_loss))) * 10
    plt.plot(iter, pretrain_loss)
    plt.title("moco pretrain loss vs iteration")
    plt.xlabel('iteration')
    plt.ylabel('moco pretrain loss')
    plt.show()

    pretrain_acc_1 = pretrain_record_df.loc[1:, 'Acc@1']
    pretrain_acc_5 = pretrain_record_df.loc[1:, 'Acc@5']
    iter = (1 + np.arange(len(pretrain_acc_1))) * 10
    plt.plot(iter, pretrain_acc_1)
    plt.plot(iter, pretrain_acc_5)
    plt.title("moco pretrain accuracy vs iteration")
    plt.legend(['top-1 acc', 'top-5 acc'])
    plt.xlabel('iteration')
    plt.ylabel('accuracy(%)')
    plt.show()

    lincls_record_df, test_acc_df = get_lincls_record(moco_lincls_txt, lincls_num_round=196)
    lincls_loss = lincls_record_df.loc[1:, 'Loss']
    iter = (1+np.arange(len(lincls_loss))) * 10
    plt.plot(iter, lincls_loss)
    plt.title("linear classification loss vs iteration")
    plt.xlabel('iteration')
    plt.ylabel('linear classification loss')
    plt.show()

    train_acc_1 = lincls_record_df.groupby('epoch')['Acc@1'].last()
    train_acc_5 = lincls_record_df.groupby('epoch')['Acc@5'].last()
    test_acc_1 = test_acc_df['Acc@1']
    test_acc_5 = test_acc_df['Acc@5']
    epoch = np.arange(len(train_acc_1))
    plt.plot(epoch, train_acc_1)
    plt.plot(epoch, train_acc_5)
    plt.plot(epoch, test_acc_1, '.-')
    plt.plot(epoch, test_acc_5,'.-')
    plt.title("linear classification accuracy vs epoch")
    plt.legend(['top-1 acc on training set', 'top-5 acc on training set', 'top-1 acc on test set', 'top-5 acc on test set'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.show()


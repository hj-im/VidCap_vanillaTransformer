import warnings

warnings.filterwarnings("ignore")

import sys
import torch.optim as optim
import torch
from torch.autograd import Variable
import string
import math
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import random
import numpy as np
import json
import collections
import coco_eval
import argparse
from cnn_back_vanilla_tr import Transformer
from torch.utils.data import DataLoader, Dataset
from makeVocab import Vocab


def parsing():
    parser = argparse.ArgumentParser(description='Video Captioning Vanilla model')
    parser.add_argument('--dataset', dest='dataset',
                        help='target dataset',
                        default='msvd', type=str)
    parser.add_argument('--d_model', dest='d_model',
                        help='hidden_dimension_model',
                        default=512, type=int)
    parser.add_argument('--num_layers', dest='num_layers',
                        help='the_number_of_layers',
                        default=4, type=int)
    parser.add_argument('--head_num', dest='head_num',
                        help='the_number_of_heads',
                        default=8, type=int)
    parser.add_argument('--dropout', dest='dropout',
                        help='dropout_rate',
                        default=0.3, type=float)
    parser.add_argument('--lr', dest='lr',
                        help='learning_rate',
                        default=1e-4, type=float)
    parser.add_argument('--bs', dest='bs',
                        help='batch_size',
                        default=64, type=int)
    parser.add_argument('--word_threshold', dest='word_threshold',
                        help='filtering_threshold_for_vocab dataset',
                        default=3, type=int)
    parser.add_argument('--max_length', dest='max_length',
                        help='sentence_max_length',
                        default=20, type=int)
    parser.add_argument('--seed', dest='seed',
                        help='control_seed',
                        default=1234, type=int)
    parser.add_argument('--model_name', dest='model_name',
                        help='model_name_for_save',
                        default='vanilla_tr', type=str)
    parser.add_argument('--epochs', dest='epochs',
                        help='epochs',
                        default=40, type=int)
    parser.add_argument('--SOS', dest='SOS',
                        help='SOS token',
                        default=1, type=int)
    parser.add_argument('--EOS', dest='EOS',
                        help='EOS token',
                        default=2, type=int)
    parser.add_argument('--UNK', dest='UNK',
                        help='UNK token',
                        default=3, type=int)
    parser.add_argument('--PAD', dest='PAD',
                        help='PAD token',
                        default=0, type=int)
    parser.add_argument('--beam', dest='beam',
                        help='beam_size',
                        default=5, type=int)
    args = parser.parse_args()

    return args


def inter_preprocess_func():
    with open('dataset/train_val_path_caption_msvd.json', 'r') as f:
        train_val_json = json.load(f)
    video_train = collections.defaultdict(list)
    train_c_lst = []
    train_p_lst = []
    for tv1 in train_val_json['train']:
        captions_tr = f"SOS {tv1['caption'].translate(str.maketrans('', '', string.punctuation))} EOS"
        video_train[tv1['file_path']].append(captions_tr)
    for vid_path_tr in video_train:
        cap_list_train = video_train[vid_path_tr]
        train_c_lst.extend(cap_list_train)
        train_p_lst.extend([vid_path_tr] * len(cap_list_train))
    video_val = collections.defaultdict(list)
    val_c_lst = []
    val_p_lst = []
    for tv2 in train_val_json['valid']:
        captions_va = f"SOS {tv2['caption'].translate(str.maketrans('', '', string.punctuation))} EOS"
        video_val[tv2['file_path']].append(captions_va)
    for vid_path_val in video_val:
        cap_list_val = video_val[vid_path_val]
        val_c_lst.extend(cap_list_val)
        val_p_lst.extend([vid_path_val] * len(cap_list_val))

    return train_p_lst, train_c_lst, val_p_lst, val_c_lst


def inter_preprocess_func_test():
    with open('dataset/test_path_caption_msvd.json', 'r') as f:
        test_val_json = json.load(f)
    video_test = collections.defaultdict(list)
    test_c_lst = []
    test_p_lst = []
    for tv3 in test_val_json['test']:
        captions_t = f"SOS {tv3['caption'].translate(str.maketrans('', '', string.punctuation))} EOS"
        video_test[tv3['file_path']].append(captions_t)

    for vid_path_te in video_test:
        cap_list_test = video_test[vid_path_te]
        test_c_lst.extend(cap_list_test)
        test_p_lst.extend([vid_path_te] * len(cap_list_test))

    return test_p_lst, test_c_lst


def test_preprocess_func():
    with open('dataset/gen_caption.json', 'r') as f:
        test_gen = json.load(f)
    lst = []
    for i in test_gen['test']:
        lst.append(i["file_path"])
    return lst


class CustomDataset(Dataset):
    def __init__(self, feature_dict, caption_dict, max_length=20):
        self.feature_dict = feature_dict
        self.caption_dict = caption_dict
        self.pad_token_idx = 0
        self.max_length = max_length

    def __len__(self):
        return len(self.feature_dict)

    def __getitem__(self, idx):
        key = np.load(self.feature_dict[idx][:-8] + 'video' + self.feature_dict[idx][-8:-3] + 'npy')
        vis = torch.Tensor(key)
        vis = torch.squeeze(vis)
        tar = self.caption_dict[idx]
        if len(tar) >= self.max_length:
            caption = torch.tensor(tar[:self.max_length])
        else:
            rest = self.max_length - len(tar)
            caption = torch.tensor(tar + [self.pad_token_idx] * rest)
        caption_mask = make_std_mask(caption, pad_token_idx=self.pad_token_idx)
        return vis, caption, caption_mask, self.feature_dict[idx]


def make_std_mask(tg, pad_token_idx):
    target_mask = (tg != pad_token_idx).unsqueeze(-2)
    target_mask = target_mask & Variable(subsequent_mask(tg.size(-1)).type_as(target_mask.data))
    return target_mask.squeeze()


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tg, pad_token_idx):
    target_mask = (tg != pad_token_idx).unsqueeze(-2)
    target_mask = target_mask & Variable(subsequent_mask(tg.size(-1)).type_as(target_mask.data))
    return target_mask.squeeze()


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def valid_preprocess_func():
    with open('dataset/gen_caption_val.json', 'r') as f:
        test_gen = json.load(f)
    lst = []
    for i in test_gen['valid']:
        lst.append(i["file_path"])

    return lst


def make_batch(samples):
    inputs = [sample[0] for sample in samples]
    captions = [sample[1] for sample in samples]
    caption_masks = [sample[2] for sample in samples]
    path_ = [sample[3] for sample in samples]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return [padded_inputs.contiguous(),
            torch.stack(captions).contiguous(), torch.stack(caption_masks).contiguous(), path_]


def data_pre(val_caption_lst_temp, voc, max_length=20):
    temp_lst_ = []

    for caption_ in val_caption_lst_temp:
        temp_lst = []
        ct = 0
        for word in caption_.split(' '):
            ct += 1
            if ct == max_length:
                break
            try:
                temp_lst.append(voc.word2idx[word])
            except:
                temp_lst.append(voc.word2idx["UNK"])
        temp_lst_.append(temp_lst)
    return temp_lst_


def evaluate(model, dataloader, epoch, nb_epochs, isTest=True):
    print('test_Start')
    model.eval()
    output = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            vis = data[0].to(device)
            pred_seq = model.predict(vis, None)
            pred_seq = pred_seq['outputs'].clone().detach()
            hypo = []
            for token in pred_seq[0]:
                hypo.append(voc.index2word[int(token)])
                if int(token) == 2:
                    break
            cap = hypo[:-1]
            caption_is = {}
            caption_is['file_path'] = data[3]
            caption_is['caption'] = ' '.join(cap)
            output.append(caption_is)
        gts, caps, ids = coco_eval.make_file_for_metric(output, isTest=isTest)
        print('===================')
        new_scorer = coco_eval.COCOScorer()

        test_scores = new_scorer.score(gts, caps, ids)
        print('-' * 89)
        print('--------- Epoch {:4d}/{} Best Epoch : {} ---------'.format(
            epoch, nb_epochs, best_epoch))
        print('Scores : {}'.format(test_scores))
        print('-' * 89)
        with open(f'results/best_epoch{epoch}.json', 'w') as f:
            json.dump(output, f)


if __name__ == '__main__':
    args = parsing()
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    voc = Vocab(args)
    voc.load()

    voc.filtering_func(args.word_threshold)

    print("==== Video Captioning Training Set ====")
    train_path_lst, train_caption_lst_temp, val_path_lst, val_caption_lst_temp = inter_preprocess_func()
    test_p_lst, test_c_lst = inter_preprocess_func_test()

    train_cap_lst = data_pre(train_caption_lst_temp, voc)
    valid_cap_lst = data_pre(val_caption_lst_temp, voc)
    test_cap_lst = data_pre(test_c_lst, voc)

    val_path_lst = valid_preprocess_func()
    test_p_lst = test_preprocess_func()

    train_dataset = CustomDataset(train_path_lst, train_cap_lst, max_length=args.max_length)
    valid_dataset = CustomDataset(val_path_lst, valid_cap_lst, max_length=args.max_length)
    test_dataset = CustomDataset(test_p_lst, test_cap_lst, max_length=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=make_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)

    checkpoint_path = f'{os.getcwd()}/checkpoints'
    save_path = f'{checkpoint_path}/{args.model_name}'
    print('model_name : ', args.model_name)

    model = Transformer(vocab_size=voc.words,
                        num_layers=args.num_layers,
                        d_model=args.d_model,
                        max_seq_len=args.max_length,
                        head_num=args.head_num,
                        dropout=args.dropout,
                        beam_size=args.beam,
                        src_pad_idx=voc.word2idx['PAD'],
                        trg_pad_idx=voc.word2idx['PAD'], trg_bos_idx=voc.word2idx['SOS'],
                        trg_eos_idx=voc.word2idx['EOS']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    print('Trainable Parameters: %.3fM' % parameters)

    nb_epochs = args.epochs
    current_cider = 0
    start_epoch = 0
    best_epoch = 0
    load_file = False
    print("Training Start")
    for epoch in range(start_epoch, nb_epochs + 1):
        total_loss = 0.0
        batch_loss = 0.0
        if os.path.isfile(f'{save_path}/{args.model_name}_{epoch - 1}.pth') and load_file == True:
            print(f'==load {epoch - 1}, {args.model_name} tr enc-dec checkpotnt==')
            checkpoint = torch.load(f'{save_path}/{args.model_name}_{epoch - 1}.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            vis = data[0].to(device)
            cap = data[1].to(device)
            cap_mask = data[2].to(device)
            vis_mask = None
            optimizer.zero_grad()
            prediction, loss = model.forward(vis, cap, cap_mask, cap, epoch=epoch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f} lr: {}'.format(
                    epoch, nb_epochs, batch_idx + 1, len(train_dataloader),
                                      batch_loss / 100
                    , optimizer.param_groups[0]['lr']
                ))
                batch_loss = 0.0

        # evaluate valid
        total_loss = 0
        model.eval()
        output = []
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_dataloader):
                vis = data[0].to(device)
                pred_seq = model.predict(vis, None)  # vit_pretrained_temp_fix
                pred_seq = pred_seq['outputs'].clone().detach()
                hypo = []
                for token in pred_seq[0]:
                    hypo.append(voc.index2word[int(token)])
                    if int(token) == 2:
                        break

                cap = hypo[:-1]
                caption_is = {}
                caption_is['file_path'] = data[3]
                caption_is['caption'] = ' '.join(cap)

                output.append(caption_is)

            gts, caps, ids = coco_eval.make_file_for_metric(output)
            scorer = coco_eval.COCOScorer()

            val_scores = scorer.score(gts, caps, ids)

            new_cider = float(val_scores['CIDEr'])
            if new_cider > current_cider:
                best_epoch = epoch
                current_cider = new_cider
                if os.path.isdir(f'{save_path}') == False:
                    os.mkdir(f'{save_path}')
                if epoch <= 20:
                    torch.save({
                        'epoch': epoch,  # 현재 학습 epoch
                        'model_state_dict': model.state_dict(),  # 모델 저장
                        'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                        'losses': total_loss / len(train_dataloader)  # Loss 저장

                    }, f'{save_path}/{args.model_name}_ep{epoch}.pth')
            val_loss = total_loss / (len(valid_dataloader) - 1)
            print('-' * 89)
            print('--------- Epoch {:4d}/{} Best Epoch : {} ---------'.format(
                epoch, nb_epochs, best_epoch))
            print('Scores : {}'.format(val_scores))
            print('-' * 89)
        print('val_end')

        # test evalutate
        if epoch == best_epoch:
            evaluate(model, test_dataloader, epoch, nb_epochs)
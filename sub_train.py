import warnings
warnings.filterwarnings("ignore")
import sys
from torch.autograd import Variable
import torch

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import os
import time
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import TransformerCfg
from dictionary import Vocabulary
import json
import collections
import torch.optim as optim
import coco_eval

# from vit_pretrained_temp_fix_pre_gate_v13_lnfix import Transformer
from vit_311_sum_extraction_vanillaTr_test4 import Transformer

from pytorch_pretrained_vit_ import ViT
import string
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
        #print(vid_path_tr,cap_list_train[0])
        train_c_lst.extend(cap_list_train)
        train_p_lst.extend([vid_path_tr]*len(cap_list_train))

    video_val = collections.defaultdict(list)
    val_c_lst = []
    val_p_lst = []

    for tv2 in train_val_json['valid']:
        captions_va = f"SOS {tv2['caption'].translate(str.maketrans('', '', string.punctuation))} EOS"
        video_val[tv2['file_path']].append(captions_va)

    for vid_path_val in video_val:
        cap_list_val = video_val[vid_path_val]
        val_c_lst.extend(cap_list_val)
        val_p_lst.extend([vid_path_val]*len(cap_list_val))

    return train_p_lst, train_c_lst, val_p_lst, val_c_lst

def test_preprocess_func():
    with open('dataset/gen_caption.json','r') as f:
        test_gen = json.load(f)
    lst = []
    for i in test_gen['test']:
        lst.append(i["file_path"])

    return lst

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
        test_p_lst.extend([vid_path_te]*len(cap_list_test))

    return test_p_lst, test_c_lst



class ImageDataset(Dataset):
    def __init__(self, feature_dict, caption_dict, max_length = 20):
        self.feature_dict = feature_dict
        self.caption_dict = caption_dict
        self.pad_token_idx = 0
        self.max_length = max_length

    def __len__(self):

        return len(self.feature_dict)

    def __getitem__(self, idx):

        v_path='/workspace/video/videofiles/msvd_frame_folder_25/frame_folder_25'
        number_ = int(self.feature_dict[idx][-8:-4])


        video_number = f'video{str(number_).zfill(4)}'
        list1 = os.listdir(os.path.join(v_path,video_number))
        list1.sort(key = lambda x:int(x[5:-4].zfill(4)))
        pre_lst = []
        size = 224
        for l1 in [list1[i] for i in range(2, 25, 3)]:
            img_ = Image.open(os.path.join(v_path, video_number,l1))
            trans_img = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            pre_img = trans_img(img_)
            pre_lst.append(pre_img)

        vis = torch.stack(pre_lst,dim = 0)

        tar = self.caption_dict[idx]

        if len(tar)>=20:
            caption = torch.tensor(tar[:20])
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


def set_seed(seed):
    '''
      For reproducibility
    '''
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def valid_preprocess_func():
    with open('dataset/gen_caption_val.json','r') as f:
        test_gen = json.load(f)
    lst = []
    for i in test_gen['valid']:
        lst.append(i["file_path"])

    return lst


def func(epoch):
    k = 3
    return 0.5 ** (epoch//k)


if __name__ == '__main__':
  seed = 1234
  set_seed(seed)

  cfg = TransformerCfg()
  cfg.dataset = 'msvd'
  voc = Vocabulary(cfg)

  ##
  d_model = cfg.d_model
  max_length = 20
  num_layer = cfg.n_layers
  head_num = cfg.head_num
  dropout = cfg.dropout
  learning_rate = cfg.learning_rate

  ##
  batch_size = 8
  min_count = 3
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  voc.load2()
  voc.trim(min_count)

  print("==== Video Captioning Training Set ====")
  train_path_lst, train_caption_lst_temp, val_path_lst, val_caption_lst_temp = inter_preprocess_func()

  train_cap_lst = []

  for caption_ in train_caption_lst_temp:
      temp_lst = []
      ct = 0
      for word in caption_.split(' '):
          ct+=1
          if ct == max_length:
              break
          try:
              temp_lst.append(voc.word2index[word])
          except:
              #pass
              temp_lst.append(voc.word2index["UNK"])
      train_cap_lst.append(temp_lst)

  valid_cap_lst = []

  for caption_ in val_caption_lst_temp:
      temp_lst = []
      ct = 0
      for word in caption_.split(' '):
          ct+=1
          if ct == max_length:
              break
          try:
              temp_lst.append(voc.word2index[word])
          except:
              #pass
              temp_lst.append(voc.word2index["UNK"])
      valid_cap_lst.append(temp_lst)

  val_path_lst = valid_preprocess_func()

  train_dataset = ImageDataset(train_path_lst, train_cap_lst)
  valid_dataset = ImageDataset(val_path_lst, valid_cap_lst)


  # test_p_lst = test_preprocess_func()
  test_p_lst, test_c_lst = inter_preprocess_func_test()

  text_cap_lst = []

  for caption_ in test_c_lst:
      temp_lst = []
      ct = 0
      for word in caption_.split(' '):
          ct+=1
          if ct == max_length:
              break
          try:
              temp_lst.append(voc.word2index[word])
          except:
              #pass
              temp_lst.append(voc.word2index["UNK"])
      text_cap_lst.append(temp_lst)

  test_p_lst = test_preprocess_func()
  # print(test_p_lst,text_cap_lst)

  test_dataset = ImageDataset(test_p_lst, text_cap_lst)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 0)
  valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False,num_workers = 0)
  test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 0)

  model_enc = ViT('B_16', pretrained=True).cuda()

  model = Transformer(vocab_size=voc.num_words,
                      num_layers=num_layer,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      src_pad_idx=voc.word2index['PAD'],
                      trg_pad_idx=voc.word2index['PAD'], trg_bos_idx=voc.word2index['SOS'],
                      trg_eos_idx=voc.word2index['EOS'], pre_train_value = 0,
                      vit_encoder = model_enc).cuda()

  dir_path = os.getcwd()
  checkpoint_path = f'{dir_path}/checkpoints'
  model_name = f'test4_s{seed}_1stage_lr{learning_rate}_layer{num_layer}'
  print('model_name : ',model_name)

  parameters = filter(lambda p: p.requires_grad, model.parameters())
  parameters = sum([np.prod(p.size()) for p in parameters])/1000000
  print('Trainable Parameters: %.3fM' % parameters)

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  nb_epochs = 40
  current_cider = 0.0
  start_epoch = 0
  best_epoch = 0
  load_file = False
  print("Training Start")
  if batch_size != 8: print("Batch Size : ",batch_size)
  for epoch in range(start_epoch ,nb_epochs + 1):
          total_loss = 0.0
          batch_loss = 0.0
          # if epoch > 2:
          #     print(f'==model enc finetuning==')
          # if start_epoch != 0 and epoch == start_epoch:
          #     if os.path.isfile(f'{checkpoint_path}/{model_name}_ep{epoch-1}.pth'):
          #         print(f'==load {epoch - 1}, {model_name} tr enc-dec checkpotnt==')
          #         checkpoint = torch.load(f'{checkpoint_path}/{model_name}_ep{epoch-1}.pth', map_location=device)
          #         model.load_state_dict(checkpoint['model_state_dict'])
          #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

          if os.path.isfile(f'{checkpoint_path}/{model_name}_ep{epoch-1}.pth') and load_file == True:
              print(f'==load {epoch - 1}, {model_name} tr enc-dec checkpotnt==')
              checkpoint = torch.load(f'{checkpoint_path}/{model_name}_ep{epoch-1}.pth', map_location=device)
              model.load_state_dict(checkpoint['model_state_dict'])
              optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

          model.train()

          for batch_idx, data in enumerate(train_dataloader):
              vis = data[0].to(device)
              cap = data[1].to(device)
              cap_mask = data[2].to(device)
              vis_mask = None
              optimizer.zero_grad()
              prediction, loss = model.forward(vis, cap, cap_mask, cap, epoch = epoch)
              loss.backward()
              optimizer.step()

              total_loss +=loss.item()
              batch_loss+=loss.item()
              if (batch_idx+1) % 100 == 0:
                  print('Epoch {:4d}/{} Batch {}/{} Loss: {:.6f} lr: {}'.format(
                      epoch, nb_epochs, batch_idx + 1, len(train_dataloader),
                      batch_loss/100
                      , optimizer.param_groups[0]['lr']
                  ))
                  batch_loss = 0.0

          if start_epoch == 0 and epoch <3:
              if epoch <= 25:
                  torch.save({
                      'epoch': epoch,  # 현재 학습 epoch
                      'model_state_dict': model.state_dict(),  # 모델 저장
                      'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                      'losses': total_loss/len(train_dataloader)  # Loss 저장
                  }, f'{checkpoint_path}/{model_name}_ep{epoch}.pth')

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
                if os.path.isdir(f'{checkpoint_path}/best/{model_name}') == False:
                    os.mkdir(f'{checkpoint_path}/best/{model_name}')
                if epoch <= 20:
                    torch.save({
                        'epoch': epoch,  # 현재 학습 epoch
                        'model_state_dict': model.state_dict(),  # 모델 저장
                        'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                        'losses': total_loss / len(train_dataloader)  # Loss 저장

                    }, f'{checkpoint_path}/best/{model_name}/{model_name}_best_ep{epoch}.pth')
            val_loss = total_loss / (len(valid_dataloader) - 1)
            print('-' * 89)
            print('--------- Epoch {:4d}/{} Best Epoch : {} ---------'.format(
                epoch, nb_epochs, best_epoch))
            print('Scores : {}'.format(val_scores))
            print('-' * 89)
          print('val_end')

          if epoch == best_epoch:
              print('test_Start')
              total_loss = 0
              model.eval()
              output = []
              with torch.no_grad():
                for batch_idx, data in enumerate(test_dataloader):
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
                print('test_Ing')
                gts, caps, ids = coco_eval.make_file_for_metric(output, isTest=True)
                print('===================')
                new_scorer = coco_eval.COCOScorer()

                test_scores = new_scorer.score(gts, caps, ids)

                print('-' * 89)
                print('--------- Epoch {:4d}/{} Best Epoch : {} ---------'.format(
                    epoch, nb_epochs, best_epoch))
                print('Scores : {}'.format(test_scores))
                print('-' * 89)
                with open(f'{model_name}_best_epoch{epoch}.json', 'w') as f:
                    json.dump(output, f)
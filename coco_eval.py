import json
import os
import argparse
import sys

sys.path.append('coco-caption')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


# Define a context manager to suppress stdout and stderr.


class suppress_stdout_stderr:
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed0
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class COCOScorer(object):
    def __init__(self):
        self._ = 0
        # print('init COCO-EVAL scorer')

    def score(self, GT, RES, IDs, tokenizers = None):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            #            print ID
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        scorers = [
            #(Bleu(), "Bleu_4"),
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:

            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if m == 'Bleu_4':
                        self.setEval(sc, m)
                        self.setImgToEvalImgs(scs, IDs, m)
                        #print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)

        return self.eval

    def setEval(self, score, method):
        # print(score)
        self.eval[method] = "%0.4f" % score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score


def make_file_for_metric(cap_dic_predict,read_path=None, gt_json_exist=True, isTest=False,length_ = 20):
    if isTest:
        gt_file_name = 'dataset/gt_msvd.json'
        gts_path = 'dataset/msvd_test_cap_json.json'
    else:
        gt_file_name = 'dataset/gt_msvd_val.json'
        gts_path = 'dataset/msvd_cap_json.json'
    if gt_json_exist == False:
        with open(gts_path, 'r') as gt:
            gt_k = json.load(gt)

        gts_dic_k = gt_k['sentences']
        gts_dic = {}

        for i in range(len(gts_dic_k)):
            if i % 10000 == 0:
                print(i)
            if gts_dic_k[i]['video_id'][3:] not in gts_dic.keys():
                gts_dic[gts_dic_k[i]['video_id'][-4:]] = []

            gts_dic_cap = {}
            gts_dic_cap['image_id'] = gts_dic_k[i]['video_id'][3:]
            gts_dic_cap['cap_id'] = len(gts_dic[gts_dic_k[i]['video_id'][-4:]])
            if len(gts_dic_k[i]['caption'].split(' ')) > length_:
                a = gts_dic_k[i]['caption'].split(' ')[:length_]
                captions_tr = ' '.join(a)
                gts_dic_cap['caption'] = captions_tr
            else:
                gts_dic_cap['caption'] = gts_dic_k[i]['caption']
            gts_dic_cap['tokenized'] = gts_dic_k[i]['caption']

            gts_dic[gts_dic_k[i]['video_id'][3:]].append(gts_dic_cap)

        with open(gt_file_name, 'w') as gt_js:
            json.dump(gts_dic, gt_js)

        print('Making gts json Success')

    elif gt_json_exist == True:
        with open(gt_file_name, 'r') as gt_js:
            gts_dic = json.load(gt_js)
    if gts_dic == None:
        print("Building gts_dic is failed!")
        exit()
    if read_path is not None:
        with open(read_path, 'r') as cap:
            cap_k = json.load(cap)
    else:
        cap_k = cap_dic_predict
    cap_dic = {}

    return_id = []
    for i in range(len(cap_k)):
        cap_dic_cap = {}
        cap_dic_cap['image_id'] = cap_k[i]['file_path'][-8:-4]
        if len(cap_k[i]['caption'].split(' ')) > length_:
            a = cap_k[i]['caption'].split(' ')[:length_]
            captions_tr = ' '.join(a)
            cap_dic_cap['caption'] = captions_tr
        else:
            cap_dic_cap['caption'] = cap_k[i]['caption']
        cap_dic[cap_k[i]['file_path'][0][-8:-4]] = []
        cap_dic[cap_k[i]['file_path'][0][-8:-4]].append(cap_dic_cap)
        return_id.append(str(cap_k[i]['file_path'][0][-8:-4]))

    return gts_dic, cap_dic, return_id

def score(ref, sample):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def cocoscorer(gts_path, read_path):
    gts, caps, IDs = make_json_metric(gts_path, read_path, True)
    scorer = COCOScorer()
    scorer.score(gts, caps, IDs)


def get_args():
    parser = argparse.ArgumentParser('Use cocosocorer metric')
    parser.add_argument('--read_path', type=str, default='json.json')
    parser.add_argument('--gts_path', type=str, default='dataset/gt_msvd.json.json')
    parser.add_argument('--gts_exist', type=bool, default=True)

    args = parser.parse_args()
    return args

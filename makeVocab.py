import os
import pickle as pickle
from collections import Counter


class Vocab:
    def __init__(self, args):
        self.dataset = args.dataset
        self.word2idx = {"PAD": args.PAD, "EOS": args.EOS, "SOS": args.SOS, "UNK": args.UNK}
        self.idx2word = {args.PAD: "PAD", args.EOS: "EOS", args.SOS: "SOS", args.UNK: "UNK"}
        self.word2count = {}
        self.word_threshold = 5
        self.filtering = False
        self.words = 0

    def word2vocab(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            if not self.filtering:
                self.word2count[word] = 1
            self.idx2word[self.num_words] = word
            self.num_words += 1
        else:
            if not self.filtering:
                self.word2count[word] += 1

    def sen2vocab(self, sentence):
        for word in sentence.split(' '):
            self.word2vocab(word)

    def save(self, word2idx='w2i.pkl', idx2word='i2w.pkl', word2cnt='w2c.pkl'):
        w2i = os.path.join('vocab', self.dataset + '_l_' + word2idx)
        i2w = os.path.join('vocab', self.dataset + '_l_' + idx2word)
        w2c = os.path.join('vocab', self.dataset + '_l_' + word2cnt)
        with open(w2i, 'wb') as f:
            pickle.dump(self.word2idx, f)
        with open(i2w, 'wb') as f:
            pickle.dump(self.idx2word, f)
        with open(w2c, 'wb') as f:
            pickle.dump(self.word2cnt, f)

    def load(self, word2index_dic='w2i.pkl', index2word_dic='i2w.pkl', word2count_dic='w2c.pkl'):
        print('vocab', self.name + '_' + word2index_dic)
        w2i = os.path.join('vocab', self.dataset + '_l_' + word2index_dic)
        i2w = os.path.join('vocab', self.dataset + '_l_' + index2word_dic)
        w2c = os.path.join('vocab', self.dataset + '_l_' + word2count_dic)
        with open(w2i, 'rb') as fp:
            self.word2index = pickle.load(fp)

        with open(i2w, 'rb') as fp:
            self.index2word = pickle.load(fp)

        with open(w2c, 'rb') as fp:
            self.word2count = pickle.load(fp)
        self.num_words = len(self.word2idx)

    def filtering(self, threshold):
        if not self.filtering:
            self.filtering = True
            temp_vocab = [word for word, cnt in self.word2count.items() if cnt >= threshold]
            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
            ))
            self.word2idx = {"PAD": self.args.PAD, "EOS": self.args.EOS, "SOS": self.args.SOS,
                               "UNK": self.args.UNK}
            self.idx2word = {self.args.PAD: "PAD", self.args.EOS: "EOS", self.args.SOS: "SOS",
                               self.args.UNK: "UNK"}
            for word in temp_vocab:
                self.word2vocab(word)
                if word not in self.word2count:
                    del self.word2count[word]
        else:
            print("No filtering Vocab. Please Check Argument or Hyper Parameter")


if __name__ =="__main__":
    print("Make Vocab file")
    Vocab.save()
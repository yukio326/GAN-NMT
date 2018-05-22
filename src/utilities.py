import numpy
from chainer import *
from itertools import zip_longest
from collections import defaultdict
import sys
import datetime
from gensim.models import word2vec
import random

class Configuration:
    def __init__(self, mode, path):
        self.mode = mode
        self.path = path
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    exec("self.{}".format(line))
        try:
            if self.mode not in ['train', 'test', 'dev']:
                raise ValueError('you must set mode = \'train\' or \'test\' or \'dev\'')
            if self.use_gpu not in [True, False]:
                raise ValueError('you must set use_gpu = True or False')
            if self.gpu_device < 0:
                raise ValueError('you must set gpu_device >= 0')
            if self.source_vocabulary_size < 1:
                raise ValueError('you must set source_vocabulary_size >= 1')
            if self.target_vocabulary_size < 1:
                raise ValueError('you must set target_vocabulary_size >= 1')
            if self.embed_size < 1:
                raise ValueError('you must set embed_size >= 1')
            if self.hidden_size < 1:
                raise ValueError('you must set hidden_size >= 1')
            if self.epoch < 1:
                raise ValueError('you must set epoch >= 1')
            if self.use_dropout not in [True, False]:
                raise ValueError('you must set use_dropout = True or False')
            if self.batch_size < 1:
                raise ValueError('you must set batch_size >= 1')
            if self.pooling < 1:
                raise ValueError('you must set pooling >= 1')
            if self.generation_limit < 1:
                raise ValueError('you must set generation_limit >= 1')
            if self.use_beamsearch not in [True, False]:
                raise ValueError('you must set use_beamsearch = True or False')
        except Exception as ex:
            print(ex)
            sys.exit()

        if self.use_gpu:
            import cupy
            self.library = cupy
        else:
            self.library = numpy
       
        if hasattr(self, "optimizer"):
            self.optimizer = self.set_optimizer(self.optimizer)
        
        if hasattr(self, "generator_optimizer"):
            self.generator_optimizer = self.set_optimizer(self.generator_optimizer)
        
        if hasattr(self, "discriminator_optimizer"):
            self.discriminator_optimizer = self.set_optimizer(self.discriminator_optimizer)
        
        if not self.use_dropout:
            self.dropout_rate = 0.0

        if not self.use_beamsearch:
            self.beam_size = 1

        if self.mode == "dev":
            self.use_beamsearch = False
            self.use_reconstructor_beamsearch = False

    def set_optimizer(self, opt):
        if opt == "AdaGrad":
            opt = optimizers.AdaGrad(lr = self.learning_rate)
        elif opt == "AdaDelta":
            opt = optimizers.AdaDelta()
        elif opt == "Adam":
            opt = optimizers.Adam()
        elif opt == "SGD":
            opt = optimizers.SGD(lr = self.learning_rate)
        return opt


class Vocabulary:
    def make(path, vocabulary_size):
        self = Vocabulary()
        self.word2id = defaultdict(lambda: 0)
        self.id2word = dict()
        with open(path, "r") as f:
            word_frequency = defaultdict(lambda: 0)
            for words in f:
                for word in words.strip("\n").split(" "):
                    word_frequency[word] += 1
            self.word2id["<unk>"] = 0
            self.word2id["</s>"] = 1
            self.word2id[""] = -1 #for padding
            self.id2word[0] = "<unk>"
            self.id2word[1] = "</s>"
            self.id2word[-1] = "" #for padding
            for i, (word, frequency) in zip(range(vocabulary_size - 2), sorted(sorted(word_frequency.items(), key = lambda x: x[0]), key = lambda x: x[1], reverse = True)):
                self.word2id[word] = i + 2
                self.id2word[i + 2] = word
        self.size = len(self.word2id) - 1
        return self

    def save(self, path):
        with open(path, "w") as f:
            for i in range(self.size):
                f.write(self.id2word[i] + "\n")

    def load(path):
        self = Vocabulary()
        self.word2id = defaultdict(lambda: 0)
        self.id2word = dict()
        with open(path, "r") as f:
            for i, word in enumerate(f):
                word = word.strip("\n")
                self.word2id[word] = i
                self.id2word[i] = word
        self.size = i + 1
        self.word2id[""] = -1 #for padding
        self.id2word[-1] = "" #for padding
        return self

def make_word2vec(path, embed_size):
    word2vec_model = word2vec.Word2Vec(word2vec.LineSentence(path), size = embed_size, min_count = 1)
    return word2vec_model

def save_word2vec(word2vec_model, path):
    word2vec_model.save(path)

def load_word2vec(path):
    return word2vec.Word2Vec.load(path)

def convert_wordlist(batch, vocabulary):
    for sentence in list(cuda.to_cpu(functions.transpose(functions.vstack(batch)).data)):
        word_list = list()
        for i in list(sentence):
            word_list.append(vocabulary.id2word[i])
        if "</s>" in word_list:
            yield " ".join(word_list[:word_list.index("</s>")])
        else:
            yield " ".join(word_list)

def mono_batch(path, vocabulary, batch_size, lib):
    with open(path, "r") as f:
        batch = list()
        for line in f:
            wordid_list = list()
            for word in line.strip("\n").split():
                wordid_list.append(vocabulary.word2id[word])
            wordid_list.append(vocabulary.word2id["</s>"])
            batch.append(wordid_list)
            if len(batch) == batch_size:
                yield [Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch, fillvalue = -1)]
                batch = list()
        if len(batch) > 0:
            yield [Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch, fillvalue = -1)]

def random_sorted_parallel_batch(source_path, target_path, source_vocabulary, target_vocabulary, batch_size, pooling, lib):
    batch_list = list()
    batch = list()
    for n_pairs in generate_n_pairs(source_path, target_path, source_vocabulary, target_vocabulary, batch_size * pooling):
        for st_pair in sorted(n_pairs, key = lambda x: len(x[0]), reverse = True):
            batch.append(st_pair)
            if len(batch) == batch_size:
                batch_list.append(batch)
                batch = list()
    if len(batch) > 0:
        batch_list.append(batch)
    random.shuffle(batch_list)
    for batch in batch_list:
        batch_source = [batch[i][0] for i in range(len(batch))]
        batch_target = [batch[i][1] for i in range(len(batch))]
        yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_target, fillvalue = -1)])

def random_sorted_3parallel_batch(source_path, target_path, output_path, source_vocabulary, target_vocabulary, batch_size, pooling, lib):
    batch_list = list()
    batch = list()
    for n_pairs in generate_n_3pairs(source_path, target_path, output_path, source_vocabulary, target_vocabulary, batch_size * pooling):
        for sto_pair in sorted(n_pairs, key = lambda x: len(x[0]), reverse = True):
            batch.append(sto_pair)
            if len(batch) == batch_size:
                batch_list.append(batch)
                batch = list()
    if len(batch) > 0:
        batch_list.append(batch)
    random.shuffle(batch_list)
    for batch in batch_list:
        batch_source = [batch[i][0] for i in range(len(batch))]
        batch_target = [batch[i][1] for i in range(len(batch))]
        batch_output = [batch[i][2] for i in range(len(batch))]
        yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_target, fillvalue = -1)], [Variable(lib.array(list(z), dtype = lib.int32)) for z in zip_longest(*batch_output, fillvalue = -1)])

def generate_n_pairs(source_path, target_path, source_vocabulary, target_vocabulary, n):
    with open(source_path, "r") as fs, open(target_path, "r") as ft:
        n_pairs = list()
        for line_source, line_target in zip(fs, ft):
            wordid_source = list()
            wordid_target = list()
            for word in line_source.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(source_vocabulary.word2id["</s>"])
            for word in line_target.strip("\n").split():
                wordid_target.append(target_vocabulary.word2id[word])
            wordid_target.append(target_vocabulary.word2id["</s>"])
            n_pairs.append([wordid_source, wordid_target])
            if len(n_pairs) == n:
                yield n_pairs
                n_pairs = list()
        if len(n_pairs) > 0:
            yield n_pairs

def generate_n_3pairs(source_path, target_path, output_path, source_vocabulary, target_vocabulary, n):
    with open(source_path, "r") as fs, open(target_path, "r") as ft, open(output_path, "r") as fo:
        n_pairs = list()
        for line_source, line_target, line_output in zip(fs, ft, fo):
            wordid_source = list()
            wordid_target = list()
            wordid_output = list()
            for word in line_source.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(source_vocabulary.word2id["</s>"])
            for word in line_target.strip("\n").split():
                wordid_target.append(target_vocabulary.word2id[word])
            wordid_target.append(target_vocabulary.word2id["</s>"])
            for word in line_output.strip("\n").split():
                wordid_output.append(target_vocabulary.word2id[word])
            wordid_output.append(target_vocabulary.word2id["</s>"])
            n_pairs.append([wordid_source, wordid_target, wordid_output])
            if len(n_pairs) == n:
                yield n_pairs
                n_pairs = list()
        if len(n_pairs) > 0:
            yield n_pairs

def copy_model(pre_model, model):
    assert isinstance(pre_model, link.Chain)
    assert isinstance(model, link.Chain)
    for child in pre_model.children():
        if child.name not in model.__dict__: continue
        model_child = model[child.name]
        if type(child) != type(model_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, model_child)
        if isinstance(child, link.Link):
            match = True
            for p, m in zip(child.namedparams(), model_child.namedparams()):
                if p[0] != m[0]:
                    match = False
                    break
                if p[1].data.shape != m[1].data.shape:
                    match = False
                    break
            if not match:
                continue
            for p, m in zip(child.namedparams(), model_child.namedparams()):
                p[1].data = m[1].data

def trace(*args):
	print(datetime.datetime.now(), '...', *args, file=sys.stderr)
	sys.stderr.flush()

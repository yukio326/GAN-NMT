#python rnn_csgan_pretrain.py [mode (train or dev or test)] [config_path] [best_epoch (only testing)]

import sys
from chainer import *
from utilities import *
from nmt import Encoder, Decoder, AttentionalNMT
import random
import copy
from rnn_csgan import Discriminator

class CSGANNMT(Chain):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, embed_size, hidden_size, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, beam_size, library, pre_nmt):
        super(CSGANNMT, self).__init__(
            generator = AttentionalNMT(source_vocabulary_size, target_vocabulary_size, embed_size, hidden_size, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, beam_size, library),
            discriminator = Discriminator(embed_size, hidden_size, use_dropout, dropout_rate, library),
		)
        if pre_nmt is not None:
            copy_model(pre_nmt, self.generator)

    def __call__(self, batch_source, batch_target, batch_output):
        self.reset_states()
        if batch_target is not None:
            source_embed_states = list()
            target_embed_states = list()
            predict_embed_states = list()
            for word in batch_source:
                source_embed_states.append(functions.tanh(self.generator.encoder.word2embed(word)))
            for word in batch_target:
                target_embed_states.append(functions.tanh(self.generator.decoder.word2embed(word)))
            for word in batch_output:
                predict_embed_states.append(functions.tanh(self.generator.decoder.word2embed(word)))
            loss_generator2, loss_discriminator, predicts_discriminator_true, predicts_discriminator_generate = self.discriminator(source_embed_states, target_embed_states, predict_embed_states)
            loss_generator = None
            predicts_generator = None
        else:
            loss_generator = None
            loss_discriminator = None
            predicts_discriminator_true = None
            predicts_discriminator_generate = None
        return loss_generator, loss_discriminator, predicts_generator, predicts_discriminator_true, predicts_discriminator_generate

    def reset_states(self):
        self.generator.reset_states()

    def get_score(self, source, target):
        source_embed = list()
        target_embed = list()
        for word in source:
            source_embed.append(functions.tanh(self.generator.encoder.word2embed(word)))
        for word in target:
            target_embed.append(functions.tanh(self.generator.decoder.word2embed(word)))

        _, _, score, _ = self.discriminator(source_embed, target_embed, target_embed)
        return score

def train(config):
    trace("Start Pre-Training ...")
    trace("Loading Vocabulary ...")
    source_vocabulary = Vocabulary.load("{}.source_vocabulary".format(config.pre_model))
    target_vocabulary = Vocabulary.load("{}.target_vocabulary".format(config.pre_model))
    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size

    trace("Making Model ...")
    pre_nmt = AttentionalNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, None, None, config.use_dropout, config.dropout_rate, None, False, None, config.library)
    serializers.load_hdf5("{}.weights".format(config.pre_model), pre_nmt)
    
    nmt = CSGANNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, None, None, config.use_dropout, config.dropout_rate, None, False, None, config.library, pre_nmt)
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()

    discriminator_opt = config.discriminator_optimizer
    discriminator_opt.setup(nmt.discriminator)
    discriminator_opt.add_hook(optimizer.WeightDecay(0.0001))

    for epoch in range(config.epoch):
        trace("Epoch {}/{}".format(epoch + 1, config.epoch))
        accum_loss_discriminator = 0.0
        finished = 0
        random.seed(epoch)
        for batch_source, batch_target, batch_output in random_sorted_3parallel_batch(config.source_train, config.target_train, config.output_train, source_vocabulary, target_vocabulary, config.batch_size, config.pooling, config.library):
            loss_generator, loss_discriminator, batch_predict_generator, batch_predict_discriminator_true, batch_predict_discriminator_generate = nmt(batch_source, batch_target, batch_output)
            accum_loss_discriminator += loss_discriminator.data
            nmt.zerograds()
            loss_discriminator.backward()
            discriminator_opt.update()

            for source, target, predict_generator, predict_discriminator_true, predict_discriminator_generate in zip(convert_wordlist(batch_source, source_vocabulary), convert_wordlist(batch_target, target_vocabulary), convert_wordlist(batch_output, target_vocabulary), batch_predict_discriminator_true.data, batch_predict_discriminator_generate.data):
                trace("Epoch {}/{}, Sample {}".format(epoch + 1, config.epoch, finished + 1))
                trace("Source                = {}".format(source))
                trace("Target                = {}".format(target))
                trace("Predict_Generator     = {}".format(predict_generator))
                trace("Predict_Discriminator = True:{} Generate:{}".format(predict_discriminator_true, predict_discriminator_generate))
                finished += 1

        trace("accum_loss_discriminator = {}".format(accum_loss_discriminator))
        trace("Saving Model ...")
        model = "{}.pretrain.{:03d}".format(config.model, epoch + 1)
        source_vocabulary.save("{}.source_vocabulary".format(model))
        target_vocabulary.save("{}.target_vocabulary".format(model))
        serializers.save_hdf5("{}.weights".format(model), nmt)
        serializers.save_hdf5("{}.optimizer_discriminator".format(model), discriminator_opt)

    trace("Finished.")

def test(config):
    trace("Loading Vocabulary ...")
    source_vocabulary = Vocabulary.load("{}.source_vocabulary".format(config.model))
    target_vocabulary = Vocabulary.load("{}.target_vocabulary".format(config.model))
    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size

    trace("Loading Model ...")
    nmt = CSGANNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, None, None, False, 0.0, config.generation_limit, config.use_beamsearch, config.beam_size, config.library, None)
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()
    serializers.load_hdf5("{}.weights".format(config.model), nmt)

    trace("Generating Translation ...")
    finished = 0
    
    with open(config.predict_file, 'w') as ft:
        for batch_source in mono_batch(config.source_file, source_vocabulary, 1, config.library):
            trace("Sample {} ...".format(finished + 1))
            _, _, batch_predict, _, _ = nmt(batch_source, None)
            for predict in convert_wordlist(batch_predict, target_vocabulary):
                ft.write("{}\n".format(predict))
                finished += 1

if __name__ == "__main__":
    config = Configuration(sys.argv[1], sys.argv[2])
    if config.mode == "train":
        config.pre_model = "{}.{:03d}".format(config.pre_model, config.pre_best_epoch)
        train(config)
    elif config.mode == "test":
        trace("Start Testing ...")
        config.source_file = config.source_test
        config.predict_file = "{}.test_result.beam{}".format(config.model, config.beam_size)
        config.model = "{}.{:03d}".format(config.model, int(sys.argv[3]))
        test(config)
        trace("Finished.")
    elif config.mode == "dev":
        trace("Start Developing ...")
        config.source_file = config.source_dev
        model = config.model
        if len(sys.argv) == 5:
            start = int(sys.argv[3]) - 1
            end = int(sys.argv[4])
        else:
            start = 0
            end = config.epoch
        for i in range(start, end):
            config.model = "{}.{:03d}".format(model, i + 1)
            trace("Model {}/{}".format(i + 1, config.epoch))
            config.predict_file = "{}.dev_result".format(config.model)
            test(config)
        trace("Finished.")

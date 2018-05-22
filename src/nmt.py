#python nmt.py [mode (train or dev or test)] [config_path] [best_epoch (only testing)]

import sys
from chainer import *
from utilities import *
import random
import copy

class Encoder(Chain):
    #Bi-directional LSTM (forward + backward)
    def __init__(self, source_vocabulary_size, embed_size, hidden_size, source_vocabulary, source_word2vec, use_dropout, dropout_rate, library):
        super(Encoder, self).__init__(
            word2embed = links.EmbedID(source_vocabulary_size, embed_size, ignore_label = -1),
            embed2hidden_forward = links.LSTM(embed_size, hidden_size),
            embed2hidden_backward = links.LSTM(embed_size, hidden_size),
        )
        if source_word2vec is not None:
            for i in range(source_vocabulary_size):
                word = source_vocabulary.id2word[i]
                if word in source_word2vec:
                    self.word2embed.W.data[i] = source_word2vec[word]
        self.vocabulary_size = source_vocabulary_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.library = library

    def __call__(self, sentence):
        return self.forward(sentence)[0]
    
    def forward(self, sentence):
        embed_states = list()
        hidden_backward_states = list()
        hidden_states = list()
        for word in sentence:
            embed_states.append(functions.dropout(functions.tanh(self.word2embed(word)), ratio = self.dropout_rate))
        for embed in embed_states[::-1]:
            hidden_backward_states.insert(0, functions.dropout(functions.tanh(self.embed2hidden_backward(embed)), ratio = 0.0)) #False
        for embed, hidden_backward in zip(embed_states, hidden_backward_states):
            plus = functions.dropout(functions.tanh(self.embed2hidden_forward(embed)), ratio = 0.0) + hidden_backward #False
            hidden_states.append(plus)
        return hidden_states, embed_states

    def reset_states(self):
        self.embed2hidden_forward.reset_state()
        self.embed2hidden_backward.reset_state()

class Decoder(Chain):
	#Luong Global-Attention (dot)
    def __init__(self, target_vocabulary_size, embed_size, hidden_size, target_vocabulary, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, beam_size, library):
        super(Decoder, self).__init__(
            encoder2decoder_init = links.LSTM(hidden_size, hidden_size),
            word2embed = links.EmbedID(target_vocabulary_size, embed_size, ignore_label = -1),
            embed_hidden_tilde2hidden = links.LSTM(embed_size + hidden_size, hidden_size),
            attention_hidden2hidden_tilde = links.Linear(2 * hidden_size, hidden_size),
            hidden_tilde2predict = links.Linear(hidden_size, target_vocabulary_size),
        )
        if target_word2vec is not None:
            for i in range(target_vocabulary_size):
                word = target_vocabulary.id2word[i]
                if word in target_word2vec:
                    self.word2embed.W.data[i] = target_word2vec[word]
        self.vocabulary_size = target_vocabulary_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.generation_limit = generation_limit
        self.use_beamsearch = use_beamsearch
        self.beam_size = beam_size
        self.library = library

    def __call__(self, encoder_hidden_states, sentence):
        return self.forward(encoder_hidden_states, sentence)[:2]

    def forward(self, encoder_hidden_states, sentence):
        predicts = list()
        target_embed_states = list()
        predict_embed_states = list()
        hidden_states = list()
        attention_weights_matrix = list()
        if sentence is not None:
            loss = Variable(self.library.zeros((), dtype = self.library.float32))
            for i, word in enumerate(sentence):
                if i == 0:
                    hidden = functions.dropout(functions.tanh(self.encoder2decoder_init(encoder_hidden_states[0])), ratio = 0.0) #False
                    encoder_hidden_states = functions.dstack(encoder_hidden_states)
                    self.copy_states()
                else:
                    previous_embed = functions.dropout(functions.tanh(self.word2embed(sentence[i - 1])), ratio = self.dropout_rate)
                    target_embed_states.append(previous_embed)
                    hidden = functions.dropout(functions.tanh(self.embed_hidden_tilde2hidden(functions.concat((previous_embed, hidden_tilde)))), ratio = 0.0) #False
                attention_weights = functions.softmax(functions.batch_matmul(encoder_hidden_states, hidden, transa = True))
                attention = functions.reshape(functions.batch_matmul(encoder_hidden_states, attention_weights), (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]))
                hidden_tilde = functions.dropout(functions.tanh(self.attention_hidden2hidden_tilde(functions.concat((attention, hidden)))), ratio = self.dropout_rate)
                hidden_states.append(hidden_tilde)
                score = self.hidden_tilde2predict(hidden_tilde)
                predict = functions.argmax(score, axis = 1)
                loss += functions.softmax_cross_entropy(score, word, ignore_label = -1)
                predict = functions.where(word.data == -1, word, predict)
                predicts.append(predict.data)
                predict_embed_states.append(functions.dropout(functions.tanh(self.word2embed(predict)), ratio = self.dropout_rate))
            target_embed_states.append(functions.dropout(functions.tanh(self.word2embed(sentence[-1])), ratio = self.dropout_rate))
            return loss, predicts, target_embed_states, predict_embed_states, hidden_states, None

        elif not self.use_beamsearch:
            while len(predicts) < self.generation_limit:
                if len(predicts) == 0:
                    hidden = functions.tanh(self.encoder2decoder_init(encoder_hidden_states[0]))
                    encoder_hidden_states = functions.dstack(encoder_hidden_states)
                    self.copy_states()
                else:
                    previous_embed = functions.tanh(self.word2embed(predict))
                    predict_embed_states.append(previous_embed)
                    hidden = functions.tanh(self.embed_hidden_tilde2hidden(functions.concat((previous_embed, hidden_tilde))))
                attention_weights = functions.softmax(functions.batch_matmul(encoder_hidden_states, hidden, transa = True))
                attention_weights_matrix.append(functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1])))
                attention = functions.reshape(functions.batch_matmul(encoder_hidden_states, attention_weights), (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]))
                hidden_tilde = functions.tanh(self.attention_hidden2hidden_tilde(functions.concat((attention, hidden))))
                hidden_states.append(hidden_tilde)
                score = self.hidden_tilde2predict(hidden_tilde)
                predict = functions.argmax(score, axis = 1)
                predicts.append(predict.data)
                if predict.data[0] == 1:
                    break
            predict_embed_states.append(functions.tanh(self.word2embed(predict)))
            attention_weights_matrix = functions.stack(attention_weights_matrix, axis = 1)
            return None, predicts, None, predict_embed_states, hidden_states, attention_weights_matrix

        else:
            initial_beam = [(0, None, list(), encoder_hidden_states, list(), list(), list())]
            for _, _, sentence, _, predict_embed_states, hidden_states, attention_weights_matrix in sorted(self.n_forwards(initial_beam), key = lambda x: x[0].data / len(x[2]), reverse = True):
                for word in sentence:
                    predicts.append(word.data)
                attention_weights_matrix = functions.stack(attention_weights_matrix, axis = 1)
                break
            return None, predicts, None, predict_embed_states, hidden_states, attention_weights_matrix

    def n_forwards(self, initial_beam):
        beam = [0] * self.generation_limit
        for i in range(self.generation_limit):
            beam[i] = list()
            if i == 0:
                new_beam = list()
                for logprob, states, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix in initial_beam:
                    hidden = functions.tanh(self.encoder2decoder_init(encoder_hidden_states[0]))
                    encoder_hidden_states = functions.dstack(encoder_hidden_states)
                    self.copy_states()
                    attention_weights = functions.softmax(functions.batch_matmul(encoder_hidden_states, hidden, transa = True))
                    attention = functions.reshape(functions.batch_matmul(encoder_hidden_states, attention_weights), (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]))
                    hidden_tilde = functions.tanh(self.attention_hidden2hidden_tilde(functions.concat((attention, hidden))))
                    prob = functions.softmax(self.hidden_tilde2predict(hidden_tilde))
                    cell, hidden = self.get_states()
                    for predict in numpy.argsort(cuda.to_cpu(prob.data)[0])[-1:-self.beam_size-1:-1]:
                        predict_variable = Variable(self.library.array([predict], dtype = self.library.int32))
                        if predict == 1:
                            new_beam.append((logprob + functions.log(prob[0][predict]), None, sentence + [predict_variable], encoder_hidden_states, embed_states + [functions.tanh(self.word2embed(predict_variable))], hidden_states + [hidden_tilde], attention_weights_matrix + [functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1]))])) 
                        else:
                            new_beam.append((logprob + functions.log(prob[0][predict]), (cell, hidden, hidden_tilde), sentence + [predict_variable], encoder_hidden_states, embed_states, hidden_states + [hidden_tilde], attention_weights_matrix + [functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1]))]))
                for _, (logprob, states, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix) in zip(range(self.beam_size), sorted(new_beam, key = lambda x: x[0].data / len(x[2]), reverse = True)):
                    beam[i].append((logprob, states, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix))
            else:
                new_beam = list()
                for logprob, states, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix in beam[i - 1]:
                    if states is not None:
                        previous_cell, previous_hidden, previous_hidden_tilde = states
                        self.embed_hidden_tilde2hidden.set_state(previous_cell, previous_hidden)
                        previous_embed = functions.tanh(self.word2embed(sentence[-1]))
                        hidden = functions.tanh(self.embed_hidden_tilde2hidden(functions.concat((previous_embed, previous_hidden_tilde))))
                        attention_weights = functions.softmax(functions.batch_matmul(encoder_hidden_states, hidden, transa = True))
                        attention = functions.reshape(functions.batch_matmul(encoder_hidden_states, attention_weights), (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]))
                        hidden_tilde = functions.tanh(self.attention_hidden2hidden_tilde(functions.concat((attention, hidden))))
                        prob = functions.softmax(self.hidden_tilde2predict(hidden_tilde))
                        cell, hidden = self.get_states()
                        for predict in numpy.argsort(cuda.to_cpu(prob.data)[0])[-1:-self.beam_size-1:-1]:
                            predict_variable = Variable(self.library.array([predict], dtype = self.library.int32))
                            if predict == 1:
                                new_beam.append((logprob + functions.log(prob[0][predict]), None, sentence + [predict_variable], encoder_hidden_states, embed_states + [previous_embed, functions.tanh(self.word2embed(predict_variable))], hidden_states + [hidden_tilde], attention_weights_matrix + [functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1]))])) 
                            else:
                                new_beam.append((logprob + functions.log(prob[0][predict]), (cell, hidden, hidden_tilde), sentence + [predict_variable], encoder_hidden_states, embed_states + [previous_embed], hidden_states + [hidden_tilde], attention_weights_matrix + [functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1]))]))
                    else:
                        new_beam.append((logprob, None, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix))
                for _, (logprob, states, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix) in zip(range(self.beam_size), sorted(new_beam, key = lambda x: x[0].data / len(x[2]), reverse = True)):
                    beam[i].append((logprob, states, sentence, encoder_hidden_states, embed_states, hidden_states, attention_weights_matrix))
        return beam[-1]


    def get_states(self):
        cell = copy.deepcopy(self.embed_hidden_tilde2hidden.c)
        hidden = copy.deepcopy(self.embed_hidden_tilde2hidden.h)
        return cell, hidden

    def copy_states(self):
        cell = self.encoder2decoder_init.c
        hidden = self.encoder2decoder_init.h
        self.embed_hidden_tilde2hidden.set_state(cell, hidden)

    def reset_states(self):
        self.encoder2decoder_init.reset_state()
        self.embed_hidden_tilde2hidden.reset_state()

class AttentionalNMT(Chain):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, embed_size, hidden_size, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, beam_size, library):
        super(AttentionalNMT, self).__init__(
            encoder = Encoder(source_vocabulary_size, embed_size, hidden_size, source_vocabulary, source_word2vec, use_dropout, dropout_rate, library),
            decoder = Decoder(target_vocabulary_size, embed_size, hidden_size, target_vocabulary, target_word2vec, use_dropout, dropout_rate, generation_limit, use_beamsearch, beam_size, library),
		)

    def __call__(self, batch_source, batch_target):
        self.reset_states()
        encoder_hidden_states = self.encoder(batch_source)
        loss, predicts = self.decoder(encoder_hidden_states, batch_target)
        return loss, predicts

    def forward(self, batch_source, batch_target):
        self.reset_states()
        encoder_hidden_states, source_embed_states = self.encoder.forward(batch_source)
        loss, predicts, target_embed_states, predict_embed_states, decoder_hidden_states, attention_matrix = self.decoder.forward(encoder_hidden_states, batch_target)
        return loss, predicts, source_embed_states, target_embed_states, predict_embed_states, encoder_hidden_states, decoder_hidden_states, attention_matrix
    
    def reset_states(self):
        self.encoder.reset_states()
        self.decoder.reset_states()

def train(config):
    if len(sys.argv) == 4:
        start = int(sys.argv[3]) - 1
        trace("Start Re-Training ...")
        trace("Loading Vocabulary ...")
        source_vocabulary = Vocabulary.load("{}.{:03d}.source_vocabulary".format(config.model, start))
        target_vocabulary = Vocabulary.load("{}.{:03d}.target_vocabulary".format(config.model, start))
        source_word2vec = None
        target_word2vec = None
    else:
        start = 0
        trace("Start Training ...")
        trace("Making Vocabulary ...")
        source_vocabulary = Vocabulary.make(config.source_train, config.source_vocabulary_size)
        target_vocabulary = Vocabulary.make(config.target_train, config.target_vocabulary_size)

        if config.use_word2vec == "Load":
            trace("Loading Word2vec ...")
            source_word2vec = load_word2vec(config.source_word2vec_file)
            target_word2vec = load_word2vec(config.target_word2vec_file)
            save_word2vec(source_word2vec, "{}.source_word2vec".format(config.model))
            save_word2vec(target_word2vec, "{}.target_word2vec".format(config.model))
        elif config.use_word2vec == "Make":
            trace("Making Word2vec ...")
            source_word2vec = make_word2vec(config.source_train, config.embed_size)
            target_word2vec = make_word2vec(config.target_train, config.embed_size)
            save_word2vec(source_word2vec, "{}.source_word2vec".format(config.model))
            save_word2vec(target_word2vec, "{}.target_word2vec".format(config.model))
        else:
            source_word2vec = None
            target_word2vec = None

    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size
    
    trace("Making Model ...")
    nmt = AttentionalNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, config.use_dropout, config.dropout_rate, None, False, None, config.library)
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()

    opt = config.optimizer
    opt.setup(nmt)
    opt.add_hook(optimizer.GradientClipping(5))


    if start != 0:
        serializers.load_hdf5("{}.{:03d}.weights".format(config.model, start), nmt)
        serializers.load_hdf5("{}.{:03d}.optimizer".format(config.model, start), opt)

    for epoch in range(start, config.epoch):
        trace("Epoch {}/{}".format(epoch + 1, config.epoch))
        accum_loss = 0.0
        finished = 0
        random.seed(epoch)
        for batch_source, batch_target in random_sorted_parallel_batch(config.source_train, config.target_train, source_vocabulary, target_vocabulary, config.batch_size, config.pooling, config.library):
            nmt.zerograds()
            loss, batch_predict = nmt(batch_source, batch_target)
            accum_loss += loss.data
            loss.backward()
            opt.update()

            for source, target, predict in zip(convert_wordlist(batch_source, source_vocabulary), convert_wordlist(batch_target, target_vocabulary), convert_wordlist(batch_predict, target_vocabulary)):
                trace("Epoch {}/{}, Sample {}".format(epoch + 1, config.epoch, finished + 1))
                trace("Source  = {}".format(source))
                trace("Target  = {}".format(target))
                trace("Predict = {}".format(predict))
                finished += 1

        trace("accum_loss = {}".format(accum_loss))
        trace("Saving Model ...")
        model = "{}.{:03d}".format(config.model, epoch + 1)
        source_vocabulary.save("{}.source_vocabulary".format(model))
        target_vocabulary.save("{}.target_vocabulary".format(model))
        serializers.save_hdf5("{}.weights".format(model), nmt)
        serializers.save_hdf5("{}.optimizer".format(model), opt)

    trace("Finished.")

def test(config):
    trace("Loading Vocabulary ...")
    source_vocabulary = Vocabulary.load("{}.source_vocabulary".format(config.model))
    target_vocabulary = Vocabulary.load("{}.target_vocabulary".format(config.model))
    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size

    trace("Loading Model ...")
    nmt = AttentionalNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.embed_size, config.hidden_size, source_vocabulary, target_vocabulary, None, None, False, 0.0, config.generation_limit, config.use_beamsearch, config.beam_size, config.library)
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()
    serializers.load_hdf5("{}.weights".format(config.model), nmt)

    trace("Generating Translation ...")
    finished = 0
    
    with open(config.predict_file, 'w') as ft:
        for batch_source in mono_batch(config.source_file, source_vocabulary, 1, config.library):
            trace("Sample {} ...".format(finished + 1))
            _, batch_predict, _, _, _, _, _, batch_attention = nmt.forward(batch_source, None)
            for source, predict, attention in zip(convert_wordlist(batch_source, source_vocabulary), convert_wordlist(batch_predict, target_vocabulary), batch_attention.data):
                ft.write("{}\n".format(predict))
                finished += 1

if __name__ == "__main__":
    config = Configuration(sys.argv[1], sys.argv[2])
    if config.mode == "train":
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

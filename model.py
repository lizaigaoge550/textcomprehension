import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
class Model():
    def __init__(self, config, vocab):
        self._config = config
        self._vocab = vocab
        if self._config.gpu_num != "":
            self.gpu_num_array = self._config.gpu.split(',')
            self.device_num = 0

    def _get_device(self):
        if self._config.gpu_num != "":
            num = self.device_num % len(self.gpu_num_array)
            res_num = self.gpu_num_array[num]
            num += 1
            return '/gpu:{0}'.format(res_num)
        else:
            return '/cpu:0'

    def build_graph(self):
        tf.logging.info('Building graph...')
        self._add_placeholders()
        #article 和 question都先经过一个双相lstm
        article_output, question_output = self._contextual_layer()
        attention_output = self._attention_layer(article_output, question_output)
        model_output = self._model_layer(attention_output)
        p1, p2 = self._output_layer(attention_output, model_output)
        self.loss_op_layer(p1,p2)

    def _get_cell(self):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self._config.hidden_dim, initializer=tf.zeros_initializer,
                                          state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self._config.hidden_dim, initializer=tf.zeros_initializer,
                                          state_is_tuple=True)
        return cell_fw, cell_bw

    def _get_bi_lstm_output(self, cell_fw, cell_bw, len , input):
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, dtype=tf.float32, sequence_length=len, swap_memory=True)
        return output

    def _add_placeholders(self):
        with tf.device(self._get_device()):
            self.article = tf.placeholder(tf.int32, [self._config.batch_size, self._config.max_article_len], name='article')
            self.article_len = tf.placeholder(tf.int32, [self._config.batch_size], name='article_len')
            self.article_padding_mask = tf.placeholder(tf.float32, [self._config.batch_size, self._config.max_article_len], name='article_padding_mask')

            self.question = tf.placeholder(tf.int32, [self._config.batch_size, self._config.max_question_len], name='question')
            self.question_len = tf.placeholder(tf.int32, [self._config.batch_size], name='question_len')
            self.question_padding_mask = tf.placeholder(tf.float32, [self._config.batch_size,self._config.max_question_len], name='question_padding_mask')

            self.start_index = tf.placeholder(tf.int32, [self._config.batch_size], name='start_index')
            self.end_index = tf.placeholder(tf.int32, [self._config.batch_size], name='end_index')

    def _contextual_layer(self):
        vsize = self._vocab.size()
        device = self._get_device()
        with tf.variable_scope('embedding_layer'),tf.device(device):
            embedding = tf.get_variable('embedding', [vsize, self._config.emb_size], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=self._config.trunc_norm_init_std))
            article_embedding = tf.nn.embedding_lookup(embedding, self.article, name='article_embedding')
            question_embedding = tf.nn.embedding_lookup(embedding, self.question, name='question_embedding')

        with tf.variable_scope('contextual_layer'),tf.device(device):
            cell_fw, cell_bw = self._get_cell()
            article_lstm_output = self._get_bi_lstm_output(cell_fw, cell_bw, self._config.article_len, article_embedding)
            question_lstm_output = self._get_bi_lstm_output(cell_fw, cell_bw, self._config.question_len, question_embedding)

        return tf.concat(axis=2,values=article_lstm_output),tf.concat(axis=2, values=question_lstm_output)

    #input : batch, article_len, 8*d
    def _model_layer(self, input):
        with tf.variable_scope("model_layer"),tf.device(self._get_device()):
            cell_fw, cell_bw = self._get_cell()
            model_output = self._get_bi_lstm_output(cell_fw, cell_bw, self._config.article_len, input)
        return tf.concat(values=model_output, axis=2)  #batch article_len 2*d

    def softsel(self, target, matrix):
        with tf.variable_scope('softsel'):
            a = tf.nn.softmax(matrix)
            target_rank = len(target.get_shape().as_list())
            out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
            return out

    def _attention_layer(self, article, question):
        #article [batch, article_len, hidden_size*2], question [batch, question_len, 2*hidden_size]
        with tf.variable_scope('attention_layer'), tf.device(self._get_device()):
            article_aug = tf.tile(tf.expand_dims(article,2),[1,1,self._config.max_question_len,1]) #[batch, article_len, question_len, 2*d]
            question_aug = tf.tile(tf.expand_dims(question,1),[1,self._config.max_article_len,1,1]) #[batch, article_len, question_len, 2*d]
            attention_matrix = tf.reduce_sum(article_aug * question_aug, axis=-1) #[batch, article_len, question_len]
            article_mask = tf.tile(tf.expand_dims(self.article_padding_mask,2),[1,1,self._config.max_question_len])
            question_mask = tf.tile(tf.expand_dims(self.question_padding_mask,1),[1,self._config.max_article_len,1])
            mask = article_mask & question_mask
            attention_matrix *= mask

            u_a = self.softsel(question_aug, attention_matrix) #batch article 2*d
            h_a = self.softsel(article, tf.reduce_max(attention_matrix, axis=-1)) #batch 2*d
            h_a = tf.tile(tf.expand_dims(h_a,1),[1,self._config.max_article_len,1]) #batch article 2*d

            return tf.concat(axis=2, values=[article, u_a, article*u_a, article*h_a]) #[batch, article_len, 8*d]

    def get_predict_value(self, input, w):
        f = tf.reshape(tf.matmul(input, w),[self._config.batch_size,-1]) #batch*articlen
        return tf.nn.softmax(f, axis=1)

    def _output_layer(self, g, m):
        def get_M_square():
            with tf.variable_scope('M_square'):
                cell_fw, cell_bw = self._get_cell()
                m_square = self._get_bi_lstm_output(cell_fw, cell_bw, self.article_len, m)
                return tf.concat(values=m_square, axis=2)

        with tf.variable_scope('output_layer'), tf.device(self._get_device()):
            w1 = tf.get_variable(name='w1', shape=[10*self._config.hidden_dim], initializer=xavier_initializer())
            w2 = tf.get_variable(name='w2', shape=[10*self._config.hidden_dim], initializer=xavier_initializer())
            index_1_input = tf.concat(values=[g, m], axis=2)
            p1 = self.get_predict_value(index_1_input, w1)
            index_2_input = tf.concat(values=[g, get_M_square()], axis=2)
            p2 = self.get_predict_value(index_2_input, w2)
            return p1, p2

    def loss_op_layer(self,p1,p2):
        #p1, p2 batch * article
        with tf.variable_scope('loss'),tf.device(self._get_device()):
            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.start_index, logits=p1, name='start_loss')
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.end_index, logits=p2, name='end_loss')
            self.loss = start_loss + end_loss

            with tf.variable_scope('optimization'):
                tvars = tf.trainable_variables()
                gradients = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                grads, global_norm = tf.clip_by_global_norm(gradients, self._config.max_grad_norm)
                optimizer = tf.train.AdagradOptimizer(self._config.lr,
                                                      initial_accumulator_value=self._config.adagrad_init_acc)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_step')


import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

from rnn import dynamic_rnn

from Nets import prelu,din_fcn_attention, dice
from vecAttGruCell import VecAttGRUCell

class Model(object):
    def __init__(self,nums_name_ids, nums_function_ids, nums_tracks,
                 EMBEDDING_DIM,HIDDEN_SIZE,ATTENTION_SIZE,use_negsampling = False):
        with tf.name_scope("Inputs"):
            self.target_ph = tf.placeholder(tf.float32,[None, None],name='target_ph')
            self.nameid_batch_ph = tf.placeholder(tf.int32,[None,],name='nameid_batch_ph')
            self.funcid_batch_ph = tf.placeholder(tf.int32,[None,None],name='functionid_batch_ph')
            self.nameid_his_batch_ph = tf.placeholder(tf.int32, [None,], name='nameid_his_batch_ph')
            self.funcid_his_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='functionid_his_batch_ph')
            self.trackid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='trackid_his_batch_ph')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.lr = tf.placeholder(tf.float64, [])  #
            self.use_negsampling = use_negsampling

            if use_negsampling:
                self.neg_funcid_his_batch_ph = tf.placeholder(tf.int32, [None, None, None, None], name='neg_functionid_his_batch_ph')
                self.neg_trackid_his_batch_ph = tf.placeholder(tf.int32,[None, None, None])


        with tf.name_scope("Embedding_layer"):

            self.nameid_embedding_var = tf.get_variable("nameid_embeding_var",[nums_name_ids, EMBEDDING_DIM])
            tf.summary.histogram("nameid_embedding_var",self.nameid_embedding_var)
            self.nameid_batch_embedded = tf.nn.embedding_lookup(self.nameid_embedding_var, self.nameid_batch_ph)
            self.nameid_his_batch_embedded = tf.nn.embedding_lookup(self.nameid_embedding_var,self.nameid_his_batch_ph)

            self.funcid_embedding_var = tf.get_variable("funcid_embedding_var",[nums_function_ids, EMBEDDING_DIM])
            tf.summary.histogram("funcid_embedding_var", self.funcid_embedding_var)
            self.funcid_batch_embedded = tf.nn.embedding_lookup(self.funcid_embedding_var, self.funcid_batch_ph)
            self.funcid_his_batch_embedded = tf.nn.embedding_lookup(self.funcid_embedding_var, self.funcid_his_batch_ph)

            self.trackid_embedding_var = tf.get_variable("trackid_embedding_var", [nums_tracks, EMBEDDING_DIM])
            tf.summary.histogram("trackid_embedding_var", self.trackid_embedding_var)
            self.trackid_his_batch_embedded = tf.nn.embedding_lookup(self.trackid_embedding_var, self.trackid_his_batch_ph)

            if use_negsampling:
                self.neg_funcid_his_batch_embedded = tf.nn.embedding_lookup(self.funcid_embedding_var, self.neg_funcid_his_batch_ph)
                self.neg_trackid_his_batch_embedded = tf.nn.embedding_lookup(self.trackid_embedding_var, self.neg_trackid_his_batch_ph)

        self.funcid_batch_embedded = tf.reduce_sum(self.funcid_batch_embedded, 1)
        self.item_eb = tf.concat([self.nameid_batch_embedded, self.nameid_his_batch_embedded,
                                  self.funcid_batch_embedded],1)

        self.funcid_his_batch_embedded = tf.reduce_sum(self.funcid_his_batch_embedded, 2)
        self.item_his_eb = tf.concat([self.funcid_his_batch_embedded,
                                      self.trackid_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

        if self.use_negsampling:
            self.neg_funcid_his_batch_embedded = tf.reduce_sum(self.neg_funcid_his_batch_embedded, 3)
            self.neg_item_his_eb = tf.concat([self.neg_funcid_his_batch_embedded[:,:,0,:],
                                            self.neg_trackid_his_batch_embedded[:,:,0,:]],-1)
            self.neg_item_his_eb = tf.reshape(self.neg_item_his_eb,[-1, tf.shape(self.funcid_his_batch_embedded)[1],EMBEDDING_DIM * 2])

            self.neg_his_eb = tf.concat([self.neg_funcid_his_batch_embedded,self.neg_trackid_his_batch_embedded],-1)
            self.neg_his_eb_sum1 = tf.reduce_sum(self.neg_his_eb, 2)
            self.neg_his_eb_sum = tf.reduce_sum(self.neg_his_eb_sum1, 1)
        self.summary_writer = tf.summary.FileWriter("summary")

    def build_fcn_net(self,inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1,200,activation=None,name="f1")

        if use_dice:
            dnn1 = dice(dnn1,name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1,80,activation=None, name='f2')

        if use_dice:
            dnn2 = dice(dnn2,name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')

        dnn3 = tf.layers.dense(dnn2, 11, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope("Metrics"):
            ctr_loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss

            if self.use_negsampling:
                self.loss = self.loss + self.aux_loss

            tf.summary.scalar("loss",self.loss)

        self.optimizer  = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat),self.target_ph), tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

        self.merged = tf.summary.merge_all()


    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input = tf.concat([h_states, click_seq], -1)
        noclick_input = tf.concat([h_states, noclick_seq], -1)

        click_prop_ = self.auxiliary_net(click_input, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input, stag=stag)[:, :, 0]

        click_loss_ = -tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)

        return loss_





    def auxiliary_net(self, input, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=input, name='bn1'+stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None,name='f1'+ stag, reuse=tf.AUTO_REUSE)
        dnn1  = tf.nn.sigmoid(dnn1)

        dnn2 = tf.layers.dense(dnn1, 50, activation=None,name='f2' + stag,reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)

        dnn3 = tf.layers.dense(dnn2, 2, activation=None,name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001

        return y_hat



    def train(self, sess, inps, iter):
        self.summary_writer.add_graph(sess.graph)
        if self.use_negsampling:
            summary, loss, accuracy, aux_loss, _ = sess.run([self.merged, self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                    feed_dict={
                                                        self.target_ph: inps[0],
                                                        self.nameid_batch_ph: inps[1],
                                                        self.funcid_batch_ph: inps[2],
                                                        self.nameid_his_batch_ph: inps[3],
                                                        self.funcid_his_batch_ph: inps[4],
                                                        self.trackid_his_batch_ph: inps[5],
                                                        self.lr: inps[6],
                                                        self.mask: inps[7],
                                                        self.seq_len_ph: inps[8],
                                                        self.neg_funcid_his_batch_ph: inps[9],
                                                        self.neg_trackid_his_batch_ph: inps[10],

                                                    })
            self.summary_writer.add_summary(summary, iter)
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer],
                                                    feed_dict={
                                                        self.target_ph: inps[0],
                                                        self.nameid_batch_ph: inps[1],
                                                        self.funcid_batch_ph: inps[2],
                                                        self.nameid_his_batch_ph: inps[3],
                                                        self.funcid_his_batch_ph: inps[4],
                                                        self.trackid_his_batch_ph: inps[5],
                                                        self.lr: inps[6],
                                                        self.mask: inps[7],
                                                        self.seq_len_ph: inps[8]
                                                    })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss],
                                                       feed_dict={
                                                           self.target_ph: inps[0],
                                                           self.nameid_batch_ph: inps[1],
                                                           self.funcid_batch_ph: inps[2],
                                                           self.nameid_his_batch_ph: inps[3],
                                                           self.funcid_his_batch_ph: inps[4],
                                                           self.trackid_his_batch_ph: inps[5],
                                                           self.mask: inps[6],
                                                           self.seq_len_ph: inps[7],
                                                           self.neg_funcid_his_batch_ph: inps[8],
                                                           self.neg_trackid_his_batch_ph: inps[9],

                                                       })
            return probs, loss, accuracy, aux_loss

        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy],
                                                       feed_dict={
                                                           self.target_ph: inps[0],
                                                           self.nameid_batch_ph: inps[1],
                                                           self.funcid_batch_ph: inps[2],
                                                           self.nameid_his_batch_ph: inps[3],
                                                           self.funcid_his_batch_ph: inps[4],
                                                           self.trackid_his_batch_ph: inps[5],
                                                           self.mask: inps[6],
                                                           self.seq_len_ph: inps[7]

                                                       })
            return probs, loss, accuracy, 0

    def save(self, sess ,path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restore from %s' % path)



class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self,nums_name_ids, nums_function_ids, nums_tracks,\
                 EMBEDDING_DIM,HIDDEN_SIZE,ATTENTION_SIZE,use_negsampling=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg, self).__init__(nums_name_ids, nums_function_ids, nums_tracks,
                 EMBEDDING_DIM,HIDDEN_SIZE,ATTENTION_SIZE,use_negsampling)
        with tf.name_scope("rnn_1"):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE),inputs=self.item_his_eb,sequence_length=self.seq_len_ph,dtype=tf.float32,scope="gru1")
            tf.summary.histogram("GRU_outputs",rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :], self.neg_item_his_eb[:, 1:, :],mask=self.mask[:,1:],stag="gru")
        self.aux_loss = aux_loss_1

        with tf.name_scope("Attention_layer_1"):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1,stag='1_1',mode='LIST',return_alphas=True)
            tf.summary.histogram("alpha_outputs", alphas)

        with tf.name_scope("rnn_2"):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE),inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas,-1),
                                                     sequence_length=self.seq_len_ph,
                                                     dtype=tf.float32,
                                                     scope='gru2')
            tf.summary.histogram("final_state2",final_state2)

        inp = tf.concat([self.item_eb,self.item_his_eb_sum, final_state2],1)
        self.build_fcn_net(inp,use_dice=True)
















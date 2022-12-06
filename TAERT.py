'''
TAERT: Triple-Attentional Explainable Recommendation with Temporal Convolutional Network
'''

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

class TAERT(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u") 
        self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i") 
        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.drop = tf.placeholder(tf.float32, name="dropout")
        iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W1")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_user = tf.reshape(self.embedded_user,[-1,review_len_u,embedding_size])

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_item = tf.reshape(self.embedded_item,[-1,review_len_i,embedding_size])

        #TCN_layer
        num_filters = 100
        kernel_size = 2
        level_size = 8
        num_channels = [num_filters]*level_size
        for i in range(level_size):
            with tf.name_scope("user_temporal_conv-%d" % i):
                init = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01)
                dilation_rate = 2**i
                conv1=tf.layers.conv1d(
                    inputs=self.embedded_user,
                    filters=num_channels[i],
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    kernel_initializer=init
                )
                batch1=tf.layers.batch_normalization(conv1)
                act1=tf.nn.relu(batch1)
                drop1=tf.layers.dropout(act1,rate=self.drop)
                conv2=tf.layers.conv1d(
                    inputs=drop1,
                    filters=num_channels[i],
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    kernel_initializer=init
                )
                batch2=tf.layers.batch_normalization(conv2)
                act2=tf.nn.relu(batch2)
                drop2=tf.layers.dropout(act2,rate=self.drop)
                if self.embedded_user.shape[-1] != drop2.shape[-1]:
                    self.embedded_user=tf.layers.conv1d(
                        inputs=self.embedded_user,
                        filters=num_channels[i],
                        kernel_size=1,
                        padding="same",
                        kernel_initializer=init
                    )
                self.embedded_user=tf.nn.relu(self.embedded_user+drop2)

        self.embedded_user = tf.transpose(self.embedded_user,[0,2,1])
        self.embedded_user = tf.reshape(self.embedded_user,[-1,review_len_u])
        Wwu = tf.Variable(tf.truncated_normal([review_len_u,review_len_u],-0.1,0.1), name='wwu')
        bwu = tf.Variable(tf.constant(0.1,shape=[review_len_u]),name='bwu')
        awu = tf.matmul(self.embedded_user,Wwu)+bwu
        awu = tf.nn.tanh(awu)
        Wwwu = tf.Variable(tf.truncated_normal([review_len_u,review_len_u],-0.1,0.1), name='wwwu')
        awu = tf.matmul(awu,Wwwu)
        awu = tf.nn.softmax(awu,1)
        self.embedded_user = tf.multiply(self.embedded_user,awu)
        self.embedded_user = tf.reshape(self.embedded_user,[-1,num_filters,review_len_u])
        self.embedded_user = tf.transpose(self.embedded_user, [0, 2, 1])
        self.embedded_user = tf.reduce_sum(self.embedded_user,1)
        self.h_pool_flat_u=tf.reshape(self.embedded_user,[-1,review_num_u,num_filters])

        for i in range(level_size):
            with tf.name_scope("item_temporal_conv-%d" % i):
                init = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01)
                dilation_rate =2**i
                conv1 = tf.layers.conv1d(
                    inputs=self.embedded_item,
                    filters=num_channels[i],
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    kernel_initializer=init
                )
                batch1 = tf.layers.batch_normalization(conv1)
                act1 = tf.nn.relu(batch1)
                drop1 = tf.layers.dropout(act1, self.drop)
                conv2 = tf.layers.conv1d(
                    inputs=drop1,
                    filters=num_channels[i],
                    kernel_size=kernel_size,
                    padding="causal",
                    dilation_rate=dilation_rate,
                    kernel_initializer=init
                )
                batch2 = tf.layers.batch_normalization(conv2)
                act2 = tf.nn.relu(batch2)
                drop2 = tf.layers.dropout(act2, self.drop)
                if self.embedded_item.shape[-1] != drop2.shape[-1]:
                    self.embedded_item = tf.layers.conv1d(
                        inputs=self.embedded_item,
                        filters=num_channels[i],
                        kernel_size=1,
                        padding="same",
                        kernel_initializer=init
                    )
                self.embedded_item = tf.nn.relu(self.embedded_item + drop2)
        self.embedded_item = tf.transpose(self.embedded_item, [0, 2, 1])
        self.embedded_item = tf.reshape(self.embedded_item, [-1, review_len_i])
        Wwi = tf.Variable(tf.truncated_normal([review_len_i, review_len_i], -0.1, 0.1), name='wwi')
        bwi = tf.Variable(tf.constant(0.1, shape=[review_len_i]), name='bwi')
        awi = tf.matmul(self.embedded_item, Wwi) + bwi
        awi = tf.nn.tanh(awi)
        Wwwi = tf.Variable(tf.truncated_normal([review_len_i,review_len_i],-0.1,0.1),name='wwwi')
        awi = tf.matmul(awi,Wwwi)
        awi = tf.nn.softmax(awi,1)
        self.embedded_item = tf.multiply(self.embedded_item, awi)
        self.embedded_item = tf.reshape(self.embedded_item, [-1, num_filters, review_len_i])
        self.embedded_item = tf.transpose(self.embedded_item, [0, 2, 1])
        self.embedded_item = tf.reduce_sum(self.embedded_item, 1)
        self.h_pool_flat_i = tf.reshape(self.embedded_item, [-1, review_num_i, num_filters])

        l2_loss += tf.nn.l2_loss(Wwu)
        l2_loss += tf.nn.l2_loss(Wwi)
        l2_loss += tf.nn.l2_loss(Wwwi)
        l2_loss += tf.nn.l2_loss(Wwwu)
        num_filters_total=num_filters

        self.h_drop_u = self.h_pool_flat_u
        self.h_drop_i = self.h_pool_flat_i
        with tf.name_scope("attention"):
            Wau = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wau')
            Wru = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wru')
            Wpu = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpu')
            bau = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bau")
            bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
            self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(iidW, self.input_reuid))
            self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_u, Wau) + tf.einsum('ajk,kl->ajl', self.iid_a, Wru) + bau),
                                             Wpu)+bbu
            self.u_a = tf.nn.softmax(self.u_j,1)


            Wai = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wai')
            Wri = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wri')
            Wpi = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpi')
            bai = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
            self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(uidW, self.input_reiid))
            self.i_j =tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_i, Wai) + tf.einsum('ajk,kl->ajl', self.uid_a, Wri) + bai),
                                             Wpi)+bbi

            self.i_a = tf.nn.softmax(self.i_j,1)

            l2_loss += tf.nn.l2_loss(Wau)
            l2_loss += tf.nn.l2_loss(Wru)
            l2_loss += tf.nn.l2_loss(Wri)
            l2_loss += tf.nn.l2_loss(Wai)

        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_sum(tf.multiply(self.u_a, self.h_drop_u), 1)
            self.u_feas = tf.nn.dropout(self.u_feas, rate = self.drop)
            self.i_feas = tf.reduce_sum(tf.multiply(self.i_a, self.h_drop_i), 1)
            self.i_feas = tf.nn.dropout(self.i_feas, rate = self.drop)

        with tf.name_scope("get_fea"):
            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf,self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf,self.input_iid)
            self.uid = tf.reshape(self.uid,[-1,embedding_id])
            self.iid = tf.reshape(self.iid,[-1,embedding_id])
            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.pu = tf.matmul(self.u_feas, Wu) + bu
            self.u_feas = self.pu + self.uid 

            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.qi = tf.matmul(self.i_feas, Wi) + bi
            self.i_feas = self.qi +self.iid 

        with tf.name_scope('attentive_interaction'):
            self.cct = tf.concat([self.u_feas, self.i_feas], 1)
            Wc = tf.Variable(tf.random_uniform([n_latent * 2, n_latent], -0.1, 0.1), name='Wc')
            bc = tf.Variable(tf.constant(0.1,shape=[n_latent]),name="bc")
            self.vc = tf.Variable(tf.random_uniform([n_latent,n_latent],-0.1,0.1),name='vc')
            self.asc = tf.matmul(tf.nn.relu(tf.matmul(self.cct,Wc)+bc)
                                    ,self.vc)
            self.asc = tf.nn.softmax(self.asc)
            self.F = tf.multiply(
                self.asc,
                tf.multiply(self.u_feas,self.i_feas)
            )
            l2_loss += tf.nn.l2_loss(Wc)

        with tf.name_scope('ncf'):
            self.FM = self.F
            self.FM = tf.nn.relu(self.FM)

            self.FM=tf.nn.dropout(self.FM,rate=self.drop)

            Wmul=tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

            self.mul=tf.matmul(self.FM,Wmul)
            self.score=tf.reduce_sum(self.mul,1,keepdims=True)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

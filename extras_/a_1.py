
# coding: utf-8

# In[2]:



import tensorflow as tf
import os
from sklearn import metrics
import numpy as np
import pickle as pk
import os

with open('word_embedding','rb') as f: 
    word_embedding = pk.load(f)
        
with open('char_embedding','rb') as f: 
    char_embedding = pk.load(f)
        
with open('train_words2ids','rb') as f: 
    train_words2ids = pk.load(f)
        
        
with open('train_chars2ids','rb') as f: 
    train_chars2ids = pk.load(f)
        
with open('train_labels2ids','rb') as f: 
    train_labels2ids = pk.load(f)
        
        
with open('train_sequence_lengths','rb') as f: 
    train_sequence_lengths = pk.load(f)
        
with open('test_words2ids','rb') as f: 
    test_words2ids = pk.load(f)
        
with open('test_chars2ids','rb') as f: 
    test_chars2ids = pk.load(f)
        
with open('test_labels2ids','rb') as f: 
    test_labels2ids = pk.load(f)
        
with open('test_sequence_lengths','rb') as f: 
    test_sequence_lengths = pk.load(f)
        
with open('label_num','rb') as f: 
    label_num = pk.load(f)
    
with open('ids2labels','rb') as f: 
    ids2labels = pk.load(f)


# In[3]:



#https://arxiv.org/abs/1603.01354#https: 
#@monk \m/


class CNN_BILSTM_CRF_NETWORK (object):
    
    
    
    def __init__(self,batch_size,word_length,sentence_length,word_dim,chr_vocab,char_dim,labels_categories,word_em=None,word_embeda=None):
        
        
        

        

        
        
        
        #placeholders
        self.input_sentences       = tf.placeholder(name='sentence',
                                               dtype=tf.int32,
                                               shape=[batch_size,sentence_length])
        self.input_char_sentence   = tf.placeholder(name='char_sentence',
                                               dtype=tf.int32,
                                               shape=[batch_size,sentence_length,word_length])
        self.labels                = tf.placeholder(name='labels',
                                               dtype=tf.int32,
                                               shape=[batch_size,sentence_length])
        self.sequence_len          = tf.placeholder(name='sequence_lengths',
                                               dtype=tf.int32,
                                               shape=[batch_size])
        self.keep_prob =             tf.placeholder(tf.float32, 
                                               name='keep_prob')
        
        

        
        
        
        


        
        

            
            # if word embedding provided
        embed = np.array(word_embeda)
        shape = embed.shape
        
        word_embedding = tf.get_variable(name="Word_embedding", 
                                             shape=shape, 
                                             initializer=tf.constant_initializer(embed), 
                                             trainable=False)
        
#         else:
            
#             #use random
#             word_embedding = tf.get_variable(name='word_embedding',
#                                              shape=[word_vocab,word_dim],
#                                              initializer=tf.random_uniform_initializer(-0.01,0.01),
#                                              dtype=tf.float32)
            
        #char embedding for cnn
        chr_embedding = tf.get_variable(name='chr_embedding',
                                        shape=[chr_vocab,char_dim],
                                        initializer=tf.random_uniform_initializer(-tf.sqrt(3.0/char_dim),tf.sqrt(3.0/char_dim)),
                                        dtype=tf.float32)
        
        
        
        #embedd input with embedding dim
        word_embedd_input = tf.nn.embedding_lookup(word_embedding,self.input_sentences)
        char_embedd_input = tf.nn.embedding_lookup(chr_embedding,self.input_char_sentence)
        
        


        
        
        
        
        
        with tf.variable_scope('cnn_network'):
            # CNN for Character-level Representation
            # A dropout layer (Srivastava et al., 2014) is applied before character embeddings are input to CNN.
            # CNN window size 3
            # number of filters 30
            cnn_filter = tf.get_variable(name='filter',
                                     shape=[3,char_dim,1,30],
                                     initializer=tf.random_uniform_initializer(-0.01,0.01),
                                     dtype=tf.float32)
            
            cnn_bias   = tf.get_variable(name='cnn_bias',
                                    shape=[30],
                                    initializer=tf.random_uniform_initializer(-0.01,0.01),
                                    dtype=tf.float32)
            
            
            
            #input dim and filter dim must match
            cnn_input = tf.reshape(char_embedd_input,[-1,word_length,char_dim,1])
        
            #dropout_applied
            cnn_input = tf.nn.dropout(cnn_input,keep_prob=self.keep_prob)
            cnn_network = tf.add( tf.nn.conv2d(cnn_input,cnn_filter,
                                               strides=[1,1,1,1],
                                               padding='VALID') , 
                                 cnn_bias )
        
            relu_applied = tf.nn.relu(cnn_network)
        
        
            #2 words for pad and unknown
            max_pool     = tf.nn.max_pool(relu_applied,
                                          ksize=[1,word_length-2,1,1],
                                          strides=[1,1,1,1],
                                          padding='VALID')
        
        
            #reshape for bilstm_network
            cnn_output   = tf.reshape(max_pool,[-1,sentence_length,30])
            
            
            
            
        word_embedding_reshape = tf.reshape(word_embedd_input,[-1,sentence_length,word_dim])
        
        # character-level representation vector is concatenated
        # with the word embedding vector to feed into
        # the BLSTM network.
    
        concat_operation = tf.concat([cnn_output,word_embedding_reshape],2)
        
        
            

        with tf.variable_scope('bi-lstm_network'):
            
              #  dropout layers are applied on both the input
              #  and output vectors of BLSTM.
            lstm_input = tf.nn.dropout(concat_operation,keep_prob=self.keep_prob)
            
            #hidden_units = 200
            forward_cell = tf.contrib.rnn.LSTMCell(num_units=200)
            backward_cell = tf.contrib.rnn.LSTMCell(num_units=200)
            
            model_output , (fs,fw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                                     cell_bw=backward_cell,
                                                                     inputs=lstm_input,
                                                                     sequence_length=self.sequence_len,
                                                                     dtype=tf.float32)
            
            # Instead of taking last output concat and reshape
            rnn_output_con = tf.concat([model_output[0],model_output[1]],2)
            
            
            #hidden_units = 200 since its bidirectional so 2*200
        rnn_output = tf.reshape(rnn_output_con,[-1,2*200])
        rnn_output_drop = tf.nn.dropout(rnn_output,
                                        keep_prob=self.keep_prob)
        
        
        
        
        
        #linear_projection without activation function
        
        projection_weight = tf.get_variable(name='projection_weight',
                                           shape=[2*200,labels_categories],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-0.01,0.01))
        projection_bias   = tf.get_variable(name='bias',
                                           shape=[labels_categories],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        projection        =  tf.add(tf.matmul(rnn_output_drop,projection_weight),projection_bias)
        
        self.projection_output = tf.reshape(projection,[-1,sentence_length,
                                                   labels_categories])
        
        
        
        
        
        # calculate_loss
        
        # for more explanation I've added docs for this function
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.projection_output,
                                                                              self.labels,
                                                                              self.sequence_len)
        
        self.loss    = -tf.reduce_mean(self.log_likelihood)
        
        
        #training
        
        train  = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9)

        grad_s = train.compute_gradients(self.loss)
        grad_clip = [[tf.clip_by_value(grad_,-5,5) , var_] for grad_,var_ in grad_s]
        
        apply_gradient = train.apply_gradients(grad_clip)
        
        self.train_operation = apply_gradient
        

        
    
    def viterbi_algorithm(self,sequence_length_,logits_,labels_,transition_):

        predicted_labels =[]
        real_labels = []
        for seq_len, logits_data , label_ in zip(sequence_length_,logits_,labels_):
            score_ = np.array(logits_data[:seq_len]).astype(np.float32)


            #     print(score)           #seq_len x num_tags    
            #     print(transition_params)#num_tags x num_tags

            viterbi,viterbi_score = tf.contrib.crf.viterbi_decode(score_,transition_)
            predicted_labels.append(viterbi)
            real_labels.append(viterbi_score)

        return {

                'predicted_labels' : predicted_labels  , 
                'real_labels' : real_labels 

               }


    

    def accuracy_calculation(self,real_,predicted_,seq_len_p):
        score_ = []
        assert len(real_)==len(predicted_) , 'length should be same'
        
        for real_h,seq_lenh,predicted_h in zip(real_,predicted_,seq_len_p):
            real_data = real_h[:predicted_h]
            predicted_data = seq_lenh
            
            if list(real_data) == list(predicted_data):
                score_.append(1)
            else:
                score_.append(0)
                
        return score_.count(1)/len(score_)
    
            
        
#         for combined in zip(real_,predicted_):
#             if combined[0]==combined[1]:
#                     score_.append(1)
#             else:
#                     score_.append(0)

#         return score_.count(1)/len(score_)
        
#         self.accuracy_calculation = {'accuracy_calculation': accuracy_calculation}
        
    def train(self,sess):
        

        
        if os.path.exists('Model/model.ckpt.meta'): 
            saver = tf.train.Saver()
            saver.restore(sess, './Model/model.ckpt')
        else:
            tf.global_variables_initializer().run()
        num_batch = len(train_words2ids) // 10
        for epo in range(1):
            for i in range(20):
                batch_words_ids = np.array(train_words2ids[i*10 : (i+1)*10])
                batch_chars_ids = np.array(train_chars2ids[i*10 : (i+1)*10]) 
                batch_labels = np.array(train_labels2ids[i*10 : (i+1)*10])
                batch_sequence_lengths = np.array(train_sequence_lengths[i*10 : (i+1)*10])
                

                
                _, predication, show_loss, trans_matrix = sess.run([self.train_operation, self.projection_output, self.loss, self.transition_params], 
                                                     feed_dict = {self.input_sentences: batch_words_ids, 
                                                                  self.input_char_sentence: batch_chars_ids,
                                                                  self.labels: batch_labels,
                                                                  self.sequence_len: batch_sequence_lengths,self.keep_prob:0.5})
                data_d= self.viterbi_algorithm(batch_sequence_lengths,predication,batch_labels,trans_matrix)
                
#                 p = self.evaluate(
#                         10, data_d['predicted_labels'], batch_labels, batch_sequence_lengths)

                p=self.accuracy_calculation(batch_labels,data_d['predicted_labels'],batch_sequence_lengths)

    
                if i % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d], loss: %.8f, accurate = %.4f"% (epo, i, num_batch, show_loss, p))
        saver = tf.train.Saver()
        saver.save(sess, 'Model/model.ckpt')
        
    def test(self, sess, ids2labels, label_num):        
        predict_label = []
        test_label = []         
        batch_num = len(test_words2ids) // 10
        for i in range(batch_num):
            batch_words_ids = np.array(test_words2ids[i*10 : (i+1)*10])
            batch_chars_ids = np.array(test_chars2ids[i*10 : (i+1)*10]) 
            batch_labels = np.array(test_labels2ids[i*10 : (i+1)*10])
            batch_sequence_lengths = np.array(test_sequence_lengths[i*10 : (i+1)*10])
            
            predication, trans_matrix = sess.run([self.projection_output, self.transition_params], 
                                                 feed_dict = {self.input_sentences: batch_words_ids, 
                                                                  self.input_char_sentence: batch_chars_ids,
                                                                  self.labels: batch_labels,
                                                                  self.sequence_len: batch_sequence_lengths,self.keep_prob:1})
            predicts = self.viterbi_algorithm(batch_sequence_lengths,predication,batch_labels,trans_matrix)
            
            
            
            for j in range(10):
                sentence_len = batch_sequence_lengths[j]
                test_label.extend(batch_labels[j][:sentence_len])
                
                
                
                predict_label.extend(predicts['predicted_labels'][j])
            if i % 50 == 0:
                print('process completed %d %%, please be patient' %int(i/batch_num*100))
        target_names = []
        for i, label in enumerate(test_label):
            test_label[i] = label - 1 
        for i, label in enumerate(predict_label):
            predict_label[i] = label - 1         
        for i in range(label_num):
            if i == 0:
                continue
            target_names.append(ids2labels[i])

        print(metrics.classification_report(test_label, predict_label, target_names=target_names))


# In[4]:


# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle as pk
import os

# flags = tf.app.flags

def main(_):
    
#     with open('word_embedding','rb') as f: 
#         word_embedding = pk.load(f)
        
        
#     with open('char_embedding','rb') as f: 
#         char_embedding = pk.load(f)
        
#     with open('train_words2ids','rb') as f: 
#         train_words2ids = pk.load(f)
        
        
#     with open('train_chars2ids','rb') as f: 
#         train_chars2ids = pk.load(f)
        
        
#     with open('train_labels2ids','rb') as f: 
#         train_labels2ids = pk.load(f)
        
        
#     with open('train_sequence_lengths','rb') as f: 
#         train_sequence_lengths = pk.load(f)
        
#     with open('test_words2ids','rb') as f: 
#         test_words2ids = pk.load(f)
        
#     with open('test_chars2ids','rb') as f: 
#         test_chars2ids = pk.load(f)
        
        
#     with open('test_labels2ids','rb') as f: 
#         test_labels2ids = pk.load(f)
        
        
#     with open('test_sequence_lengths','rb') as f: 
#         test_sequence_lengths = pk.load(f)
        
        
#     with open('label_num','rb') as f: 
#         label_num = pk.load(f)

    print('finish converting')

    with tf.Session() as sess:
        print('begin training')
        
        
        
        

        
        model = CNN_BILSTM_CRF_NETWORK(10,70,150,100,86,30,10,word_em=1,word_embeda=word_embedding)
        
        
        model.train(sess)
 
#         model.train(sess)
#         print('finish training')
#         print('start testing')
        if os.path.exists('Model/model.ckpt.meta'): 
            saver = tf.train.Saver()
            saver.restore(sess, './Model/model.ckpt')

            model.test(sess, ids2labels, label_num)
        else:
            raise Exception("[!] Train a model first")

if __name__ == '__main__':
  tf.app.run()


# In[ ]:


#              precision    recall  f1-score   support

#       B-ORG       0.80      0.80      0.80      1660
#       B-LOC       0.85      0.91      0.88      1663
#       I-ORG       0.80      0.84      0.82       834
#      I-MISC       0.57      0.70      0.63       213
#           O       0.99      0.99      0.99     38468
#      B-MISC       0.73      0.80      0.76       698
#       B-PER       0.94      0.84      0.89      1615
#       I-LOC       0.77      0.83      0.80       257
#       I-PER       0.97      0.93      0.95      1156

# avg / total       0.97      0.97      0.97     46564


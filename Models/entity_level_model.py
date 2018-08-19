
# coding: utf-8

import tensorflow as tf
import os

from sklearn import metrics
import numpy as np
import pickle as pk
import os

def write_file(data):
    with open('new_file','a') as f:
        f.write(data+'\n')




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


# In[2]:

#https://arxiv.org/abs/1603.01354#https: 
#@monk \m/


class CNN_BILSTM_CRF_NETWORK (object):
    
    
    
    def __init__(self,batch_size,word_length,sentence_length,word_dim,chr_vocab,char_dim,labels_categories,word_em=None,word_embeda=None):
        
        
        

        

        
        
        
        #placeholders
        self.input_sentences       = tf.placeholder(name='sentence',
                                               dtype=tf.int32,
                                               shape=[None,None])
        self.input_char_sentence   = tf.placeholder(name='char_sentence',
                                               dtype=tf.int32,
                                               shape=[None,None,None])
        self.labels                = tf.placeholder(name='labels',
                                               dtype=tf.int32,
                                               shape=[None,None])
        self.sequence_len          = tf.placeholder(name='sequence_lengths',
                                               dtype=tf.int32,
                                               shape=[None])
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
            forward_cell = tf.contrib.rnn.LSTMCell(num_units=300)
            backward_cell = tf.contrib.rnn.LSTMCell(num_units=300)
            
            model_output , (fs,fw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                                     cell_bw=backward_cell,
                                                                     inputs=lstm_input,
                                                                     sequence_length=self.sequence_len,
                                                                     dtype=tf.float32)
            
            # Instead of taking last output concat and reshape
            rnn_output_con = tf.concat([model_output[0],model_output[1]],2)
            
            
            #hidden_units = 200 since its bidirectional so 2*200
        rnn_output = tf.reshape(rnn_output_con,[-1,2*300])
        rnn_output_drop = tf.nn.dropout(rnn_output,
                                        keep_prob=self.keep_prob)
        
        
        
        
        
        #linear_projection without activation function
        
        projection_weight = tf.get_variable(name='projection_weight',
                                           shape=[2*300,labels_categories],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-0.01,0.01))
        projection_bias   = tf.get_variable(name='bias',
                                           shape=[labels_categories],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        projection        =  tf.add(tf.matmul(rnn_output_drop,projection_weight),projection_bias)
        
        self.projection_output = tf.reshape(projection,[-1,sentence_length,
                                                   labels_categories])
        
        self.probability_distribution = tf.nn.softmax(self.projection_output, name='self.projection_output')

        
        
        
        
        
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
    
    


    def preprocess_test(self,logits,seq_len):
        final_real_d=[]
        for m in zip(logits,seq_len):
            real_p=m[0][:m[1]]
            real_arg=[]
            for sn in real_p:
                real_arg.append(np.argmax(sn))
            final_real_d.append(real_arg)
        return final_real_d
        

    def accuracy_calulation(self,real_,predicted_,seq_len_p):
        

        mean_entity_level=[]
        mean_f1_score=[]
        for real_h,seq_lenh,predicted_h in zip(real_,predicted_,seq_len_p):
            real_data = real_h[:predicted_h]
            predicted_data = seq_lenh
            
 
            True_positive=0
            False_positive=0
            False_negative=0
            Entity_level=[]
            for i in zip(real_data,predicted_data):
                if i[0]==i[1] and i[0]!=3:
                    True_positive+=1
                if i[0]!=i[1] and i[1]!=3:
                    False_positive+=1
                if i[0]!=i[1] and i[0]!=3:
                    False_negative+=1
                if i[0]==i[1] and i[0]!=3:
                    Entity_level.append(1)
                elif i[0]!=i[1] and i[0]!=3:
                    Entity_level.append(0)
                    
            



            mean_entity_level.append(Entity_level.count(1)/len(Entity_level))
            
            try:
                Precision = True_positive / (True_positive + False_positive)
                Recall = True_positive / (True_positive + False_negative)
            except ZeroDivisionError:
                Precision =0
                Recall=0
                
                
            try:
                f1_score = 2 * (Precision * Recall) / (Precision + Recall)
                mean_f1_score.append(f1_score)
            except Exception:
                mean_f1_score.append(0)

        return {'entity_level_accuracy': np.mean(mean_entity_level) , 'f1_score': np.mean(mean_f1_score) }
    
            
    
            
        


    def test(self, sess, ids2labels, label_num):
        predict_label = []
        test_label = []
#         batch_num = 10
#         final_accu=[]
#         for i in range(batch_num):
#             batch_words_ids = np.array(test_words2ids[i*10 : (i+1)*10])
#             batch_chars_ids = np.array(test_chars2ids[i*10 : (i+1)*10]) 
#             batch_labels = np.array(test_labels2ids[i*10 : (i+1)*10])
#             batch_sequence_lengths = np.array(test_sequence_lengths[i*10 : (i+1)*10])

            
        predication, trans_matrixs = sess.run([self.projection_output, self.probability_distribution], 
                                                 feed_dict = {self.input_sentences: test_words2ids, 
                                                                  self.input_char_sentence: test_chars2ids,
                                                              self.sequence_len: test_sequence_lengths,self.keep_prob:1})
        return self.accuracy_calulation( test_labels2ids,self.preprocess_test(trans_matrixs,test_sequence_lengths),test_sequence_lengths)

        
        
    def train(self,sess):
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        num_batch = len(train_words2ids) // 10
        accur_board=[0]
        for epo in range(1000):
            for i in range(num_batch):
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

                p=self.accuracy_calulation(batch_labels,data_d['predicted_labels'],batch_sequence_lengths)
    
                

    
                if i % 10 == 0:
                    print({'a': p,"loss":show_loss,'epoch':epo,'iteration':i},'\n')
                if i%300 ==0:
                    print('start_tesrting')

                    print({"testing_accuracy":self.test(sess, ids2labels, 4)})
                    write_file(str({"testing_accuracy":self.test(sess, ids2labels, 4),'epoch':epo,'leader_board':accur_board}))
                    print(accur_board[-1],self.test(sess, ids2labels, 4)['f1_score'])
                    if accur_board[-1] < self.test(sess, ids2labels, 4)['f1_score']:
                        print('new_record_found')
                        saver.save(sess, 'Models/'+str(epo))
                        accur_board.append(self.test(sess, ids2labels, 4)['f1_score'])
                    else:
                        print('no new_record_found',accur_board)
                    
                    
                

        saver = tf.train.Saver()
        saver.save(sess, 'Models/sfinal.ckpt')
        


def main(_):


    print('finish converting')

    with tf.Session() as sess:
        print('begin training')
        
        
        
        

        
        model = CNN_BILSTM_CRF_NETWORK(10,70,150,200,58,30,4,word_em=1,word_embeda=word_embedding)
        
        
        model.train(sess)
if __name__ == '__main__':
  tf.app.run()






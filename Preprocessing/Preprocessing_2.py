
# coding: utf-8

# In[1]:


# Use better way for encoding ner tags

import numpy as np
import pickle as pk

def char_embedding_matrix(sentences, d):
    chars = []
    embedding_matrix = []
    char_id = {}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                chars.append(char)
    chars = set(chars)

    char_id['NUL'] = 0
    char_id['Un_known'] = len(chars) + 1
    embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (d, 1)))
    for i, char in enumerate(chars):
        char_id[char] = i+1
        embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (d, 1)))
    embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (d, 1))) 
    embedding_matrix = np.reshape(embedding_matrix, [-1, d])
    embedding_matrix = embedding_matrix.astype(np.float32)
    return (char_id, chars, embedding_matrix)


    
def word_embedding_matrix(path, sentences, d):
    word_id = {}   
    vocab = []
    with open(path, encoding="utf8") as fp:
        for sentence in sentences:
            for word in sentence:
                vocab.append(word.lower())
        vocab = set(vocab)
        vocabulary = []
        embedding_matrix = [np.random.uniform(-1.0, 1.0, (d, 1))]
        for line in fp.readlines():
            if line.split()[0] in vocab:
                vocabulary.append(line.split()[0])
                embedding_matrix.append(np.reshape(np.array(line.split()[1:]), (d, 1)))
#   mapping words and ids, add a null and a unknown symbol
    word_id['NUL'] = 0
    word_id['Un_known'] = len(vocabulary) + 1
    embedding_matrix.append(np.random.uniform(-1.0, 1.0, (d, 1)))
    embedding_matrix = np.reshape(embedding_matrix, [-1, d])
    embedding_matrix = embedding_matrix.astype(np.float32)
#    embedding_matrix.dtype = 'float32' 
    for i, word in enumerate(vocabulary):
        word_id[word] = i+1
    
    word_max = 50
    for sentence in sentences:
        word_max = max(word_max, len(sentence))
        
    return (word_id, vocabulary, embedding_matrix)




def build_label_ids(labels):
    label = []
    label_id = {}
    for sentence in labels:
        for word in sentence:
            label.append(word)
    label = set(label)
#    label_num = len(label)
    label_id['NUL'] = 0
    for i, label in enumerate(label):
        label_id[label] = i + 1
    id_label = {v : k for k, v in label_id.items()}
    label_num = len(label_id.values())
           
    return (label_id, id_label, label_num)



def word2id(word_id, vocabulary, X, max_lenth):    
    X2id = []
    sequence_lengths = []
    for sentence in X:
        index = []
        for word in sentence:
            if word.lower() in vocabulary:
                index.append(word_id[word.lower()])
            else:
                index.append(word_id['Un_known'])
        sequence_lengths.append(len(index))
        index = np.pad(index, (0, max_lenth-len(index)), 'constant')
        X2id.append(index)
    return (X2id, sequence_lengths)





def char2id(char_id, vocabulary, X, max_sentence, max_word):
    X2id = []
    for sentence in X:
        word_index = []
        for word in sentence:
            char_index = []
            for char in word:
                if char in vocabulary:
                    char_index.append(char_id[char])
                else:
                    char_index.append(char_id['Un_known'])                
#pad each word ids with 0, pad each sentence ids with 0
            char_index = np.pad(char_index, (0, max_word-len(char_index)), 'constant')
            word_index.append(char_index)
        word_index = np.pad(word_index, ((0, max_sentence-len(word_index)), (0, 0)), 'constant')
        X2id.append(word_index)
    return X2id        
               

    
    
def label2id(label_id, X, max_sentence):
    X2id = []
    for sentence in X:
        index = []
        for label in sentence:
            index.append(label_id[label])
        index = np.pad(index, (0, max_sentence-len(index)), 'constant') 
        X2id.append(index)    
    return X2id
       

    
    

#dataset preparation

#with stop words and without stop words
#encoding method , normal method 

#file_path = ade_raw data path
#save_path = where all converted file will be saved


def pre_first(file_path,save_path,word_embedding_size,char_embedding_size,max_sentence,max_word,stop_wor=None,bilou=None):
    
    from nltk.corpus import stopwords
    stop_words =stopwords.words('english')
    
    def remove_stop(text_d):
        text_ = text_d.split()
        new_ = []
        for m in text_:
            if m not in stop_words:
                new_.append(m.lower().strip().replace(',','').replace('.',''))
        return new_

    def remove_pun(text_d):
        text_ = text_d.split()
        new_ = []
        for m in text_:
            new_.append(m.lower().strip().replace(',','').replace('.',''))
        return new_
    
    sentences_f =[]
    drug_names =[]
    reactions_f =[]
    
    with open(file_path,'r') as f:
        for line in f:
            actual_data=line.strip().split('|')
            if stop_wor:
                sents = remove_stop(actual_data[1].lower())
                    
            else:
                sents = remove_pun(actual_data[1].lower())

            
            adverse_ef = actual_data[2].lower()
            drug_name= actual_data[5].lower()
        
            drug_names.append(remove_pun(drug_name))
            reactions_f.append(remove_pun(adverse_ef))
            sentences_f.append(sents)
            
    final_labels=[] 
    final_sentences=[]
    
    if bilou:
        
        
        for se_i,drug_j,reac_k in zip(sentences_f,drug_names,reactions_f):
            curee=['empty_']*len(se_i)
            for r,dru in enumerate(se_i):
                for g,m in enumerate(drug_j):
                    if m in dru:
                        curee[r]='drug_'+str(g)
                
                for d,f in enumerate(reac_k):
                    if f in dru:
                        curee[r]='reaction_'+str(d)
            final_labels.append(curee)
            final_sentences.append(se_i)
    
    else:
        for se_i,drug_j,reac_k in zip(sentences_f,drug_names,reactions_f):
            curee=['empty_']*len(se_i)
            for r,dru in enumerate(se_i):
                for g,m in enumerate(drug_j):
                    if m in dru:
                        curee[r]='drug_'
                
                for d,f in enumerate(reac_k):
                    if f in dru:
                        curee[r]='reaction_'

            final_labels.append(curee)
            final_sentences.append(se_i)
            
            
    train_da_int = int(len(final_sentences)*0.9)
    train_sentences = final_sentences[:train_da_int]
    train_labels = final_labels[:train_da_int]
    
    test_sentences = final_sentences[train_da_int:]
    test_labels = final_labels[train_da_int:]
    
    print('finish reading')
    print('begin initializing char embedding, each char represented by 30 dimension number')
    chars2ids, char_vocabulary, char_embedding = char_embedding_matrix(train_sentences ,char_embedding_size)
    
    with open('./'+ str(save_path) + '/chars2ids','wb') as f:
        pk.dump(chars2ids,f)
        
    with open('./'+ str(save_path) + '/char_vocabulary','wb') as f:
        pk.dump(char_vocabulary,f)
        
    with open('./'+ str(save_path) + '/char_embedding','wb') as f:
        pk.dump(char_embedding,f)
        
    print('training_chr_ids')
    
    train_chars2ids  = char2id(chars2ids, char_vocabulary, train_sentences, max_sentence, max_word)
    
    with open('./'+ str(save_path) + '/train_chars2ids','wb') as f:
        pk.dump(train_chars2ids,f)

    test_chars2ids = char2id(chars2ids, char_vocabulary, test_sentences, max_sentence, max_word)
    
    with open('./'+ str(save_path) + '/test_chars2ids','wb') as f:
        pk.dump(test_chars2ids,f)
        
    print('finish initializing')
    
    print('begin reading pre-trained word embedding, each word represented by 100 dimension number')  
    
    words2ids, word_vocabulary, word_embedding = word_embedding_matrix('d300.txt', 
                                                                        train_sentences, word_embedding_size)
    
    with open('./'+ str(save_path) + '/words2ids','wb') as f:
        pk.dump(words2ids,f)
        
    with open('./'+ str(save_path) + '/word_vocabulary','wb') as f:
        pk.dump(word_vocabulary,f)
        
    with open('./'+ str(save_path) + '/word_embedding','wb') as f:
        pk.dump(word_embedding,f)
        
    print('finish reading')
    print('begin converting words and chars to ids seperately')
    
    train_words2ids, train_sequence_lengths = word2id(words2ids, word_vocabulary, train_sentences, max_sentence)
    
    with open('./'+ str(save_path) + '/train_words2ids','wb') as f:
        pk.dump(train_words2ids,f)
        
    with open('./'+ str(save_path) + '/train_sequence_lengths','wb') as f:
        pk.dump(train_sequence_lengths,f)
        
        
    
    test_words2ids, test_sequence_lengths = word2id(words2ids, word_vocabulary, test_sentences, max_sentence)

    with open('./'+ str(save_path) + '/test_words2ids','wb') as f:
        pk.dump(test_words2ids,f)

    with open('./'+ str(save_path) + '/test_sequence_lengths','wb') as f:
        pk.dump(test_sequence_lengths,f)



    labels2ids, ids2labels, label_num = build_label_ids(train_labels)

    with open('./'+ str(save_path) + '/labels2ids','wb') as f:
        pk.dump(labels2ids,f)

    with open('./'+ str(save_path) + '/ids2labels','wb') as f:
        pk.dump(ids2labels,f)


    with open('./'+ str(save_path) + '/label_num','wb') as f:
        pk.dump(label_num,f)



    train_labels2ids = label2id(labels2ids, train_labels, max_sentence)

    with open('./'+ str(save_path) + '/train_labels2ids','wb') as f:
        pk.dump(train_labels2ids,f)
        
        

    test_labels2ids = label2id(labels2ids, test_labels, max_sentence)


    with open('./'+ str(save_path) + '/test_labels2ids','wb') as f:
        pk.dump(test_labels2ids,f)



    return 'finish converting'

            
    



# In[2]:


first=pre_first(
    
    file_path                      ='ade.txt',
    save_path                      ='test_1',
    word_embedding_size            =200,
    char_embedding_size            =30,
    max_sentence                   =150,
    max_word                       =70,
    bilou =                     1
    
    )


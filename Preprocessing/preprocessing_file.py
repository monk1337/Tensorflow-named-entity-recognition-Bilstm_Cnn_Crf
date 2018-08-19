
# coding: utf-8

# In[1]:


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
            curee=[0]*len(se_i)
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
            curee=[0]*len(se_i)
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

            
    



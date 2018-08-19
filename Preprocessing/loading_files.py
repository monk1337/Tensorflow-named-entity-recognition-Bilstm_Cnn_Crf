
# coding: utf-8

# In[1]:


import pickle as pk

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


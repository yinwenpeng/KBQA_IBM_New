import nltk
import numpy
import codecs

path='/mounts/data/proj/wenpeng/Dataset/freebase/SimpleQuestions_v2/'

def load_word2vec_to_init(rand_values, file):

    readFile=open(file, 'r')
#     line_count=1
    for line in readFile:
        tokens=line.strip().split()
        id=int(tokens[0].strip())
#         print id, len(tokens[1:]), line
        rand_values[id]=numpy.array(map(float, tokens[1:]))
#         line_count+=1                                            
    readFile.close()
    print 'initialization over...'
    return rand_values
def create_wordVocab_word2GloveEmb():
    readFile=codecs.open('/mounts/data/proj/wenpeng/Dataset/glove.6B.50d.txt', 'r', 'utf-8')
    dim=50
    line_control=1000
    line_start=0
    glove={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            glove[tokens[0]]=map(float, tokens[1:])
#             line_start+=1
#             if line_start==line_control:
#                 break
    glove_vocab=set(glove.keys())
    readFile.close()
    print 'glove loaded over...'   
    
    #word vocab
    word2id={}
    id2emb={}
    raw_word_vocab=set()
    raw_char_vocab=set()
    triple_files=['annotated_fb_data_train_PNQ_50nega_str&des.txt', 'annotated_fb_data_valid_PNQ_50nega_str&des.txt', 'annotated_fb_data_test_PNQ_50nega_str&des.txt']
    for i in range(3):
        print i, '...'
        train_file_triples=codecs.open(path+triple_files[i], 'r', 'utf-8')
        file_line=0
        for line in train_file_triples:
            parts=line.strip().split('\t')
            triples=parts[:51]
            entity_des=parts[51:-1]
            for triple in triples:
                tri_parts=triple.strip().lower().split(' == ')
                if len(tri_parts)!=3:
                    print 'weird triple:', triple
                    exit(0)
                relation=tri_parts[1].strip()
                relation_words=relation.split('_')
                for relation_word in relation_words:
                    if len(relation_word)>0:
                        if relation_word.find(' ')>=0:
                            print 'relation:', relation, triple
                            exit(0)
                        raw_word_vocab.add(relation_word)
                head=list(tri_parts[0].strip().lower())
                for head_char in head:
#                     if len(head_char)>1:
#                         print head_char
#                         exit(0)
                    raw_char_vocab.add(head_char)

#                 tail=list(tri_parts[2].decode('utf-8').strip().lower())
#                 for tail_char in tail:
#                     raw_char_vocab.add(tail_char)
#                 print raw_char_vocab
#                 exit(0)
            for des in entity_des:
                truncate_des=des.strip().lower().split()[:20]#nltk.word_tokenize(des.strip().lower().decode('utf-8'))[:20]
                for des_word in truncate_des:
                    if len(des_word)>0:
                        if des_word.find(' ')>=0:
                            print 'truncate_des:', truncate_des
                            exit(0)
                        raw_word_vocab.add(des_word)
            file_line+=1
            if file_line%10000==0:
                print file_line
        train_file_triples.close()   
    print 'triple files loaded over'
    question_files=['annotated_fb_data_train_mention_remainQ.txt', 'annotated_fb_data_valid_mention_remainQ.txt', 'annotated_fb_data_test_mention_remainQ.txt']
    for i in range(3):
        question_open=codecs.open(path+question_files[i], 'r', 'utf-8')
        for line in question_open:
            parts=line.strip().split('\t')
            question=parts[-1].lower().split()
            for q_word in question:
                if q_word.find(' ')>=0:
                    print 'question:', question
                    exit(0)
                raw_word_vocab.add(q_word)
        question_open.close()
    print 'question files loaded over'    
    random_emb=list(numpy.random.uniform(-0.01,0.01,dim))        
    for r_word in raw_word_vocab:
        if word2id.get(r_word, -1)==-1:
            new_id=len(word2id)
            word2id[r_word]=new_id
            if r_word in glove_vocab:
                id2emb[new_id]=glove.get(r_word)
            else:
                id2emb[new_id]=random_emb
            
    write_vocab=codecs.open(path+'word_vocab.txt', 'w', 'utf-8')
    for word, id in word2id.iteritems():
        if word.find(' ')>=0:
            print 'weird word:', word
            exit(0)
        write_vocab.write(word+'\t'+str(id)+'\n')
    write_vocab.close()
    
    write_word_emb=codecs.open(path+'word_emb.txt', 'w', 'utf-8')
    word_size=len(id2emb)
    for i in range(word_size):
        write_word_emb.write(str(i)+'\t'+' '.join(map(str, id2emb.get(i)))+'\n')
    write_word_emb.close()

    write_char_vocab=codecs.open(path+'char_ids.txt', 'w', 'utf-8')
    char_index=0
    for char in raw_char_vocab:               
        write_char_vocab.write(char+'\t'+str(char_index)+'\n')
        char_index+=1
    write_char_vocab.close()
    print 'all write over'
        

def load_train_test(triple_files, question_files, max_char_len, max_des_len, max_relation_len, max_Q_len, neg_all):
    #load char_vocab, word_vocab


    char2id={}
    word2id={}

    result=[]
    for i in range(3):
        print i, '...'
        train_file_triples=codecs.open(path+triple_files[i], 'r', 'utf-8')
        train_file_mentions=codecs.open(path+question_files[i], 'r', 'utf-8')
        pos_entity_char=[]
        pos_entity_des=[]
        relations=[]
        
        entity_char_lengths=[]
        entity_des_lengths=[]
        relation_lengths=[]
    #     tail_entity_des=[]
        for line in train_file_triples:
            parts=line.strip().split('\t')
            triples=parts[:neg_all+1]
            char_ids=[] # first max_char_len are always positive
            r_word_ids=[]
            des_ids=[]
            char_lens=[]
            r_word_lens=[]
            des_word_lens=[]
            for triple in triples:
                tri_parts=triple.strip().split(' == ')
                h=list(tri_parts[0].strip().lower())[:max_char_len]
                
                len_temp=len(h)
                left=(max_char_len-len_temp)/2
                right=max_char_len-left-len_temp
                char_lens+=[left, len_temp, right]
                char_ids+=[0]*left
                for h_char in h:
                    char_id=char2id.get(h_char)
                    if char_id is None:
                        char_id=len(char2id)+1  #start from 1
                        char2id[h_char]=char_id
                    char_ids.append(char_id)
                char_ids+=[0]*right
             
                r_words=tri_parts[1].strip().lower().split('_')[:max_relation_len]
                len_temp=len(r_words)
                left=(max_relation_len-len_temp)/2
                right=max_relation_len-left-len_temp
                r_word_lens+=[left, len_temp, right]
                r_word_ids+=[0]*left
                for r_word in r_words:
                    word_id=word2id.get(r_word)
                    if word_id is None:
                        word_id=len(word2id)+1 # start from 1
                        word2id[r_word]=word_id
                    r_word_ids.append(word_id)
                r_word_ids+=[0]*right
                
                t=list(tri_parts[2].strip().lower())[:max_char_len]
                
                len_temp=len(t)
                left=(max_char_len-len_temp)/2
                right=max_char_len-left-len_temp
                char_lens+=[left, len_temp, right]
                char_ids+=[0]*left
                for t_char in t:
                    char_id=char2id.get(t_char)
                    if char_id is None:
                        char_id=len(char2id)+1
                        char2id[t_char]=char_id
                    char_ids.append(char_id)
                char_ids+=[0]*right  
            pos_entity_char.append(char_ids)
            relations.append(r_word_ids)
            entity_char_lengths.append(char_lens)
            relation_lengths.append(r_word_lens)
            
            
            deses= parts[neg_all+1:-1]
            for des in deses:
                truncate_des=des.strip().lower().split()[:max_des_len]#nltk.word_tokenize(des.strip().lower().decode('utf-8'))[:20]
                len_temp=len(truncate_des)
                left=(max_des_len-len_temp)/2
                right=max_des_len-left-len_temp
                des_word_lens+=[left, len_temp, right]
                des_ids+=[0]*left
                for des_word in truncate_des:
                    id=word2id.get(des_word)
                    if id is None:
                        id=len(word2id)+1
                        word2id[des_word]=id
                    des_ids.append(id)
                des_ids+=[0]*right
            pos_entity_des.append(des_ids)   
            entity_des_lengths.append(des_word_lens)
                         
        train_file_triples.close()
        
        mention_char_ids=[]
        remainQ_word_ids=[]
        mention_char_lens=[]
        remainQ_word_lens=[]
        for line in  train_file_mentions:
            m_char_ids=[]
            Q_word_ids=[]
            parts=line.strip().lower().split('\t')       
            mention=parts[2]
            remainQ=parts[3]    
            m=list(mention.strip())[:max_char_len]
            
            len_temp=len(m)
            left=(max_char_len-len_temp)/2
            right=max_char_len-left-len_temp
            mention_char_lens.append([left, len_temp, right])
            m_char_ids+=[0]*left
            for m_char in m:
                char_id=char2id.get(m_char)
                if char_id is None:
                    char_id=len(char2id)+1
                    char2id[m_char]=char_id
                m_char_ids.append(char_id)
            m_char_ids+=[0]*right    
            mention_char_ids.append(m_char_ids)          
                    
            Q_words=remainQ.strip().lower().split()[:max_Q_len]
            len_temp=len(Q_words)
            left=(max_Q_len-len_temp)/2
            right=max_Q_len-left-len_temp
            remainQ_word_lens.append([left, len_temp, right])
            Q_word_ids+=[0]*left
            for Q_word in Q_words:
                word_id=word2id.get(Q_word)
                if word_id is None:
                    word_id=len(word2id)+1
                    word2id[Q_word]=word_id
                Q_word_ids.append(word_id)
            Q_word_ids+=[0]*right    
            remainQ_word_ids.append(Q_word_ids)
            

        
                   
        result_temp=(pos_entity_char, pos_entity_des, relations, entity_char_lengths, entity_des_lengths, relation_lengths, mention_char_ids, remainQ_word_ids, mention_char_lens, remainQ_word_lens)
        result.append(result_temp)

    read_char_file=codecs.open(path+'char_ids.txt', 'w', 'utf-8')
    for char, id in char2id.iteritems():
        read_char_file.write(char+'\t'+str(id)+'\n')
    read_char_file.close()
    print 'char ids written over'


    read_word_file=codecs.open(path+'word_vocab.txt', 'w', 'utf-8')
    for word, id in word2id.iteritems():
        read_word_file.write(word+'\t'+str(id)+'\n')
    read_word_file.close()
    print 'word vocab written over'


    readFile=codecs.open('/mounts/data/proj/wenpeng/Dataset/glove.6B.50d.txt', 'r', 'utf-8')
    dim=50
#     line_control=1000
#     line_start=0
    glove={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            glove[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'glove loaded over...'  
    write_word_emb=codecs.open(path+'word_emb.txt', 'w', 'utf-8')
    random_emb=list(numpy.random.uniform(-0.01,0.01,dim))     
    for word, id in word2id.iteritems():
        emb=glove.get(word, random_emb)
        write_word_emb.write(str(id)+'\t'+' '.join(map(str, emb))+'\n')
    write_word_emb.close()
    print 'initialized word embs written over'
    return result, len(word2id), len(char2id)
    
def load_test_or_valid(testfile, char2id, word2id, max_char_len, max_des_len, max_relation_len, max_Q_len, test_size):

    length_per_example=[]
    
    train_file_triples=codecs.open(path+testfile, 'r', 'utf-8')
#         train_file_mentions=codecs.open(path+question_files[i], 'r', 'utf-8')
    pos_entity_char=[]
    entity_char_lengths=[]
    entity_scores=[]
    
    pos_entity_des=[]
    entity_des_lengths=[]
    
    relations=[]
    relation_lengths=[]
    
    mention_char_ids=[]
    mention_char_lens=[]
    remainQ_word_ids=[]
    remainQ_word_lens=[]

    line_co=0
    for line in train_file_triples:
        parts=line.strip().split('\t')
        supposed_triple_size=int(parts[0])
        length_per_example.append(supposed_triple_size)
        parts=parts[1:]
        if len(parts)!=supposed_triple_size*4:
            print 'row length problem:', len(parts)
            exit(0)
        triples=parts[:supposed_triple_size]
        names=parts[supposed_triple_size: supposed_triple_size*2]
        deses=parts[supposed_triple_size*2:supposed_triple_size*3]
        men_Qs=parts[supposed_triple_size*3:]
        #first load relations and entity scores
        r_word_ids=[]
        r_word_lens=[]        
        entity_score=[]
        for triple in triples:
#             print triple
            tri_parts=triple.strip().split('==')
            score=tri_parts[2]
            if score=='None':
                score='0.0'
            entity_score.append(float(score))
            #r_words=[tri_parts[1].strip().lower()]+tri_parts[1].strip().lower().split('_')
            r_words=tri_parts[1].strip().lower().split('_')
	    r_words=r_words[:max_relation_len]
            len_temp=len(r_words)
            left=(max_relation_len-len_temp)/2
            right=max_relation_len-left-len_temp
            r_word_lens+=[left, len_temp, right]
            r_word_ids+=[0]*left
            for r_word in r_words:
                word_id=word2id.get(r_word)
                if word_id is None:
                    word_id=len(word2id)+1 # start from 1
                    word2id[r_word]=word_id
                r_word_ids.append(word_id)
            r_word_ids+=[0]*right            
            
        relations.append(r_word_ids)
        relation_lengths.append(r_word_lens)
        entity_scores.append(entity_score)
        
        #names       
        char_ids=[] # first max_char_len are always positive
        char_lens=[]
        
        for name in names:
            h=list(name.strip().lower())[:max_char_len]
            
            len_temp=len(h)
            left=(max_char_len-len_temp)/2
            right=max_char_len-left-len_temp
            char_lens+=[left, len_temp, right]
            char_ids+=[0]*left
            for h_char in h:
                char_id=char2id.get(h_char)
                if char_id is None:
                    char_id=len(char2id)+1  #start from 1
                    char2id[h_char]=char_id
                char_ids.append(char_id)
            char_ids+=[0]*right            
        entity_char_lengths.append(char_lens)
        pos_entity_char.append(char_ids)
        
        #des
        des_word_lens=[]
        des_ids=[]
        for des in deses:
            truncate_des=des.strip().lower().split()[:max_des_len]#nltk.word_tokenize(des.strip().lower().decode('utf-8'))[:20]
            len_temp=len(truncate_des)
            left=(max_des_len-len_temp)/2
            right=max_des_len-left-len_temp
            des_word_lens+=[left, len_temp, right]
            des_ids+=[0]*left
            for des_word in truncate_des:
                id=word2id.get(des_word)
                if id is None:
                    id=len(word2id)+1
                    word2id[des_word]=id
                des_ids.append(id)
            des_ids+=[0]*right
        pos_entity_des.append(des_ids)   
        entity_des_lengths.append(des_word_lens)
                     
        #men_Q

        m_char_ids=[]
        m_char_len=[]
        Q_word_ids=[]
        Q_word_len=[]
        for men_q in men_Qs:
            parts=men_q.strip().split('==')
            m=list(parts[0])[:max_char_len]
            
            len_temp=len(m)
            left=(max_char_len-len_temp)/2
            right=max_char_len-left-len_temp
            m_char_len+=[left, len_temp, right]
            m_char_ids+=[0]*left
            for m_char in m:
                char_id=char2id.get(m_char)
                if char_id is None:
                    char_id=len(char2id)+1
                    char2id[m_char]=char_id
                m_char_ids.append(char_id)
            m_char_ids+=[0]*right    
         
                    
            Q_words=parts[1].strip().lower().split()[:max_Q_len]
            len_temp=len(Q_words)
            left=(max_Q_len-len_temp)/2
            right=max_Q_len-left-len_temp
            Q_word_len+=[left, len_temp, right]
            Q_word_ids+=[0]*left
            for Q_word in Q_words:
                word_id=word2id.get(Q_word)
                if word_id is None:
                    word_id=len(word2id)+1
                    word2id[Q_word]=word_id
                Q_word_ids.append(word_id)
            Q_word_ids+=[0]*right    
        remainQ_word_ids.append(Q_word_ids)
        remainQ_word_lens.append(Q_word_len)
        mention_char_ids.append(m_char_ids) 
        mention_char_lens.append(m_char_len)
        
        line_co+=1
        if line_co==test_size:
            break       
    result=(pos_entity_char, pos_entity_des, relations, entity_char_lengths, entity_des_lengths, relation_lengths, mention_char_ids, remainQ_word_ids, mention_char_lens, remainQ_word_lens, entity_scores)
    
    print 'load', line_co, 'test examples over'

    return result, length_per_example, word2id, char2id        
def load_train(trainfile, testfile, max_char_len, max_des_len, max_relation_len, max_Q_len, train_size, test_size, mark):
    #load char_vocab, word_vocab
    char2id={}
    word2id={}


    supposed_triple_size=100
    train_file_triples=codecs.open(path+trainfile, 'r', 'utf-8')
#         train_file_mentions=codecs.open(path+question_files[i], 'r', 'utf-8')
    pos_entity_char=[]
    entity_char_lengths=[]
    entity_scores=[]
    
    pos_entity_des=[]
    entity_des_lengths=[]
    
    relations=[]
    relation_lengths=[]
    
    mention_char_ids=[]
    mention_char_lens=[]
    remainQ_word_ids=[]
    remainQ_word_lens=[]

    line_co=0
    for line in train_file_triples:
        parts=line.strip().split('\t')
        parts=parts[1:]
        if len(parts)!=supposed_triple_size*4:
            print 'row length problem:', len(parts)
            exit(0)
        triples=parts[:supposed_triple_size]
        names=parts[supposed_triple_size: supposed_triple_size*2]
        deses=parts[supposed_triple_size*2:supposed_triple_size*3]
        men_Qs=parts[supposed_triple_size*3:]
        #first load relations and entity scores
        r_word_ids=[]
        r_word_lens=[]        
        entity_score=[]
        for triple in triples:
#             print triple
            tri_parts=triple.strip().split('==')
            score=tri_parts[2]
            if score=='None':
                score='0.0'
            entity_score.append(float(score))
            r_words=tri_parts[1].strip().lower().split('_')
            r_words=r_words[:max_relation_len]
            len_temp=len(r_words)
            left=(max_relation_len-len_temp)/2
            right=max_relation_len-left-len_temp
            r_word_lens+=[left, len_temp, right]
            r_word_ids+=[0]*left
            for r_word in r_words:
                word_id=word2id.get(r_word)
                if word_id is None:
                    word_id=len(word2id)+1 # start from 1
                    word2id[r_word]=word_id
                r_word_ids.append(word_id)
            r_word_ids+=[0]*right            
            
        relations.append(r_word_ids)
        relation_lengths.append(r_word_lens)
        entity_scores.append(entity_score)
        
        #names       
        char_ids=[] # first max_char_len are always positive
        char_lens=[]
        
        for name in names:
            h=list(name.strip().lower())[:max_char_len]
            
            len_temp=len(h)
            left=(max_char_len-len_temp)/2
            right=max_char_len-left-len_temp
            char_lens+=[left, len_temp, right]
            char_ids+=[0]*left
            for h_char in h:
                char_id=char2id.get(h_char)
                if char_id is None:
                    char_id=len(char2id)+1  #start from 1
                    char2id[h_char]=char_id
                char_ids.append(char_id)
            char_ids+=[0]*right            
        entity_char_lengths.append(char_lens)
        pos_entity_char.append(char_ids)
        
        #des
        des_word_lens=[]
        des_ids=[]
        for des in deses:
            truncate_des=des.strip().lower().split()[:max_des_len]#nltk.word_tokenize(des.strip().lower().decode('utf-8'))[:20]
            len_temp=len(truncate_des)
            left=(max_des_len-len_temp)/2
            right=max_des_len-left-len_temp
            des_word_lens+=[left, len_temp, right]
            des_ids+=[0]*left
            for des_word in truncate_des:
                id=word2id.get(des_word)
                if id is None:
                    id=len(word2id)+1
                    word2id[des_word]=id
                des_ids.append(id)
            des_ids+=[0]*right
        pos_entity_des.append(des_ids)   
        entity_des_lengths.append(des_word_lens)
                     
        #men_Q

        m_char_ids=[]
        m_char_len=[]
        Q_word_ids=[]
        Q_word_len=[]
        for men_q in men_Qs:
            parts=men_q.strip().split('==')
            m=list(parts[0])[:max_char_len]
            
            len_temp=len(m)
            left=(max_char_len-len_temp)/2
            right=max_char_len-left-len_temp
            m_char_len+=[left, len_temp, right]
            m_char_ids+=[0]*left
            for m_char in m:
                char_id=char2id.get(m_char)
                if char_id is None:
                    char_id=len(char2id)+1
                    char2id[m_char]=char_id
                m_char_ids.append(char_id)
            m_char_ids+=[0]*right    
         
                    
            Q_words=parts[1].strip().lower().split()[:max_Q_len]
            len_temp=len(Q_words)
            left=(max_Q_len-len_temp)/2
            right=max_Q_len-left-len_temp
            Q_word_len+=[left, len_temp, right]
            Q_word_ids+=[0]*left
            for Q_word in Q_words:
                word_id=word2id.get(Q_word)
                if word_id is None:
                    word_id=len(word2id)+1
                    word2id[Q_word]=word_id
                Q_word_ids.append(word_id)
            Q_word_ids+=[0]*right    
        remainQ_word_ids.append(Q_word_ids)
        remainQ_word_lens.append(Q_word_len)
        mention_char_ids.append(m_char_ids) 
        mention_char_lens.append(m_char_len)
        
        line_co+=1
        if line_co==train_size:
            break       
    result=(pos_entity_char, pos_entity_des, relations, entity_char_lengths, entity_des_lengths, relation_lengths, mention_char_ids, remainQ_word_ids, mention_char_lens, remainQ_word_lens, entity_scores)
    
    print 'load', line_co, 'training examples over'

    result_test, length_per_example_test, word2id, char2id  = load_test_or_valid(testfile, char2id, word2id, max_char_len, max_des_len, max_relation_len, max_Q_len, test_size)

    read_char_file=codecs.open(path+'char_ids'+mark+'.txt', 'w', 'utf-8')
    for char, id in char2id.iteritems():
        read_char_file.write(char+'\t'+str(id)+'\n')
    read_char_file.close()
    print 'char ids written over'


    read_word_file=codecs.open(path+'word_vocab'+mark+'.txt', 'w', 'utf-8')
    for word, id in word2id.iteritems():
        read_word_file.write(word+'\t'+str(id)+'\n')
    read_word_file.close()
    print 'word vocab written over'


    readFile=codecs.open('/mounts/data/proj/wenpeng/Dataset/glove.6B.50d.txt', 'r', 'utf-8')
    dim=50
#     line_control=1000
#     line_start=0
    glove={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            glove[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'glove loaded over...'  
    write_word_emb=codecs.open(path+'word_emb'+mark+'.txt', 'w', 'utf-8')
    random_emb=list(numpy.random.uniform(-0.01,0.01,dim))     
    for word, id in word2id.iteritems():
        emb=glove.get(word, random_emb)
        write_word_emb.write(str(id)+'\t'+' '.join(map(str, emb))+'\n')
    write_word_emb.close()
    print 'initialized word embs written over'
    return result, result_test, length_per_example_test, len(word2id), len(char2id)   
    
def load_word2id_char2id(mark):
    word2id={}
    char2id={}
    read_wordfile=codecs.open(path+'word_vocab'+mark+'.txt', 'r', 'utf-8')
    for line in read_wordfile:
        parts=line.strip().split('\t')
        word2id[parts[0]]=int(parts[1])
    read_wordfile.close()
    print 'load word2id over'
    read_charfile=codecs.open(path+'char_ids'+mark+'.txt', 'r', 'utf-8')
    for line in read_charfile:
        parts=line.strip().split()
        if len(parts)==1:
#             print line, parts[0]
            char2id[' ']=int(parts[0])

        else:
            char2id[parts[0]]=int(parts[1])
    read_charfile.close()
    print 'load char2id over'
    return word2id, char2id
    
if __name__ == '__main__':

#     create_wordVocab_word2GloveEmb()    #basically not useful
#     resu=load_train_test(triple_files, question_files, 40, 20, 5, 30)
    lis=[1,2,3,4,5,6]
    print lis[1::2]






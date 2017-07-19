import pandas as pd
import numpy as np
import regex as re
import enchant
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import word2vec
from sklearn.cluster import KMeans
import math
import time
import base64
import io

class connect:
    
    def __init__(self):

        df=pd.read_csv('daily_conversation_s.csv000',names=['id','id.1','created_at','text','speciality','user_type','gender','age','doctor_id'])
        df['id'] = df.id.astype(str)
        index = []
        for i in range(len(df)):
            if (df.id[i]).isdigit() is False:
                index.append(i)
        df = df.drop(df.index[index],axis=0).reset_index(drop=True)
        df['id']=df.id.astype(int)
        df['id.1']=df['id.1'].astype(int)
        #print (df.head(5))#,names=['id','id.1','created_at','text','speciality','user_type','gender','age','doctor_id'])
        df = df.dropna(axis=0,how='any')
        df.to_csv('main.csv')
        #df=df.sort_values(by=['id.1','id'],ascending=[True,True]).reset_index(drop=True)
        self.df = df
        #print(len(df))

    def removekey(self,d, key):
        
        r = dict(d)
        del r[key]
        return r
        
    def contains_word(self,s, w):
        return (' ' + w + ' ') in (' ' + s + ' ')

    
    def cleaned(self):
        df =self.df
        
        df.drop('id',1,inplace=True)
        df.rename(columns={'id.1':'id'},inplace=True)
        #df = pd.read_csv(self.path)
        df = df.dropna(axis=0)
        
        #base conversations filtering
        
        
        d = {smallItem:0 for smallItem in list(set(df.id))}
        for_rem = [] 
        #base conversations filtering
        
        #l = list(set(df.id))
        #print (l)
        for i in d:

            df1 = df[df.id==i].reset_index(drop=True)
            if i%1000==0:
                print(i)
            for j in range(len(df1)-1):

                if (df1.user_type[j]=='Doctor') and (df1.user_type[j+1]=='User'):
                    #d=self.removekey(d,i)
                    for_rem.append(i)
                    #print (i)
                    break
        df.drop(df[df['id'].isin(for_rem)].index,inplace=True)
        print ('basic filtering done')
                    
       # df = df.loc[df['id'].isin(list(d.keys()))].reset_index(drop=True)
        
        a=df.groupby(['id'])['text'].apply(' '.join)
        df1=a.reset_index()
        
        #remove_duplicates
        ID=[]
        for i in range(len(df1)-1):
            if df1.text[i]==df1.text[i+1]:
                ID.append(df1.id[i])
        
        print('ID', ID)
        #remove garbage
        
        d=enchant.Dict("en_US")
        l3 = []#list(set(df1.id))
        
        #ind=[]
        for i in range(len(df1)):
            a=re.sub('[^0-9a-zA-Z]+', ' ', (df1.text[i]).strip()).split(" ")
            a = list(filter(None, a))

            if len(a)>0:
                #print (len(a))
                x=0
                for word in a:
                    if (d.check(word) is True):
                        #print (word)
                        x+=1

                if (x/len(a))<.75:
                    #print (df1.id[i])
                    try:
                        l3.append(df1['id'][i])
                    except:
                        print (' ')
                    #ind.append(df1.id[i])
                else:
                    for word in a:
                        if (len(word)>20) and (d.check(word) is False) and (word[:4]!='http'):
                            try:
                                
                            
                                l3.append(df1['id'][i])
                            except:
                                print (' ')
                    
                        #l3 = [x for x in list(set(df.id)) if x not in list(set(ind))]
        
        print ('close to cleaning')
        #removing conversations with user query length less than 4
        df = df.reset_index(drop=True)
        index=[]
        for i in range(len(df)):
            x = re.sub(r'[^a-zA-Z0-9]',' ',str(df.text[i]).lower())

            if (df.user_type[i]=='User') and len(x.split())<4:
                #print (df.text[i])
                index.append(i)

            if (df.user_type[i]=='Doctor') and ((len(x.split())<4) or (self.contains_word(x,'report') is True)):
                #print (df.text[i])
                index.append(i)

        df = df.drop(df.index[index],axis=0).reset_index(drop=True)

        
        #removing conversations having doc's questions           
        ques_id = []
        que = df[df.user_type=='Doctor'].reset_index(drop=True)
        for i in range(len(que)):
            if ((re.search(r'\?',que.text[i])) is None) and (self.contains_word(re.sub(r'[^a-zA-Z0-9]',' ',que.text[i].lower()),'sorry') is False):
                pass #ques_id.append(que.id[i])
            #if self.contains_word(re.sub(r'[^a-zA-Z0-9]',' ',que.text[i].lower()),'sorry') is False:
               # pass
                
            else:
                #print (que.text[i])
                ques_id.append(que.id[i])
        ques_id= list(set(ques_id))
        print('closer')
        #print(df['id'][0])
        # BARBAAD
        
        df.drop(df[df['id'].isin(ques_id)].index,inplace=True)#df[df.id not in ques_id]#df.loc[df['id'].isin(ques_id)].reset_index(drop=True)
        df.drop(df[df['id'].isin(ID)].index,inplace=True)
        df.drop(df[df['id'].isin(l3)].index,inplace=True)
        #remove duplicates within conversation
        z=pd.DataFrame(data=None)
        #y=0
        a_vals = []
        for i in list(set(df.id)):
            #a = df4[df4.id==i].reset_index(drop=True)
            a = df[df.id==i].drop_duplicates('text','first')
            # df[df.id==i] = a
            if ('Doctor' in list(a.user_type)) and ('User' in list(a.user_type)):
                a_vals.append(a)
                # z=pd.concat([a,z],axis=0)
        z = pd.concat( a_vals+[z], axis=0)
        # df.drop(df[df['id'].isin(fr_rm)].index,inplace=True)
        qdf=z.sort_index().reset_index(drop=True)
        # qdf=df.sort_index().reset_index(drop=True)
        print('more closer')
        return (qdf)
    
    
    def anonymised(self):
        
        ns = pd.read_csv('docu/conv_id2doc_id.csv')
        c2p = pd.read_csv('docu/conv_pat.csv')
        ns.first_name = ns.first_name.apply(lambda x:x.lower())
        ns.last_name = ns.last_name.astype('str')
        ns.last_name = ns.last_name.apply(lambda x:x.lower())
        c2p.name = c2p.name.apply(lambda x:x.lower())
        f = open('docu/conv_id2doc_id.csv', 'r')
        f.readline()
        doc_dict = {}
        for line in f:
            line = line.strip().replace('"', '').lower().split(',')
            doc_dict[line[2]] = (line[0] +" "+ line[1]).split()
        f = open('docu/conv_pat.csv', 'r')
        f.readline()
        pat_dict = {}
        for line in f:
            line = line.strip().replace('"', '').lower().split(',')
            pat_dict[line[0]] = line[1].split()
            

        print ((list(pat_dict.items()))[:10])

        first=[]
        n=[]
        for name in ns.first_name:
            x= name.split()
            if len(x)>1:
                #print (x[1])
                first.append(x[0])
                n.append(x[1])
            else:
                first.append(x[0])
                n.append(None)
        ns['first_name']=first
        ns['middle_name']=n

        names =[]
        surnames = []
        for name in c2p.name:
            x = name.split()
            if len(x)>1:
                names.append(x[0])
                surnames.append(x[1])
            else:
                names.append(name)
                surnames.append(None)

        c2p['name'] = names
        c2p['surname'] = surnames
        ns['id'] = pd.to_numeric(ns['id'], errors='coerce')
        clean_text=dict()
        
        qdf = self.cleaned()
        print (len(qdf))
        #qdf = qdf.head(1000)
        #qdf = qdf[qdf.id==220073].reset_index(drop=True)
        print (qdf.head(5))
        #qdf = qdf[qdf.id==11910].reset_index(drop=True)
   
        #qdf = qdf[qdf['id']!=NaN].reset_index(drop=True)
       # qdf=qdf[pd.isnull(qdf['id'])].reset_index(drop=True)
       # qdf = qdf.iloc[:100,:]
        qdf['id'] = pd.to_numeric(qdf['id'], errors='coerce')
        #print (qdf.head(10))
        print('start',time.time())
        ind_drop=[]
        d=enchant.Dict("en_US")
        print (len(qdf))
        for j in range(len(qdf)):
            #print (j)
            # f = ns[ns.id==(qdf.id[j])].reset_index(drop=True)
            #print (f)
            # g = c2p[c2p.id==qdf.id[j]].reset_index(drop=True)
            #print (g)
            # print(qdf.text[j])
            #print (qdf.id[j])
            try:
                convsn_id = str(int(qdf.id[j]))
                line=' '.join((qdf.text[j]).split())
            
            #removing new lines
                line = line.replace('\\n',' ').replace('\\\\t','').replace('\\','')
    
            #removing multiple stops
                line = re.sub('\.\.+', ' ', line).replace('\,',',')

                words=line.replace('\n',' ').strip()#.split(' ')
                words = " ".join(words.split())
                if len(words)==0:
                    qdf.drop(qdf.index[j],axis=0,inplace=True)
                else:
                    words = words.split()
                #words = re.sub('[^0-9a-zA-Z]+', ' ', words).split(" ")

                    for i in range(len(words)):
                        if not words[i]:
                            continue
                        if (len(re.sub(r'[^0-9\-]','',words[i]))>=9) and words[i][0] in '0987':#((words[i])[0]=='9' or (words[i])[0]=='0' or (words[i])[0]=='8' or (words[i])[0]=='7'):
                            #print (words[i])
                            words[i]= '9XXXXXX'
                        if len(words[i])>8 and (words[i][:2]!='9X') and  (d.check(words[i]) is False) and ((not set('aeiou').isdisjoint(words[i].lower())) is False):
                           # print (words[i])
                            words[i]=''
                        if len(words[i])>2:
                            


                        #anonymising patient name    

                            if convsn_id in pat_dict:#if g.empty is False:

                                if (re.sub(r'[^a-zA-Z0-9]','',str(words[i]).lower())) in pat_dict[convsn_id]:#[g.name[0], g.surname[0]]:#(words[i].lower()== g.name[0]) or ((words[i][:-1]).lower()==g.name[0]) or(words[i].lower()==g.surname[0]) or ((words[i][:-1]).lower()==g.surname[0]) :


                                    #print (words[i])
                                    words[i]='PATIENT'

                    
                    

                    clean_text[j]=(" ".join(words))
# print (line)
            #removing new lines




            except:
               # print (j)
                ind_drop.append(j)
    



        qdf.drop(qdf.index[ind_drop],axis=0,inplace=True)



        print('end', time.time())
        v = list(clean_text.values())
        li=[]
        for i in range(len(v)):

            words= v[i].split()

            for j in range(1,len(words)):
                if words[j-1]=='DOCTOR':
                    words[j]=''
                if words[j-1]=='PATIENT':
                    words[j]=''
                if (words[j].lower()=='doc') or (words[j].lower()=='dr'):
                    words[j]=''

            li.append(' '.join(words))
           
        qdf['clean_text']=li
        print (len(qdf))
        #qdf=qdf.dropna(axis=0).reset_index(drop=True)        
        otc = pd.read_csv('docu/rx_otc.csv',names=['id','is_otc'])
        skus = pickle.load(open('docu/sorted_skus2id.p','rb'))
        sku=dict()
        print (len(qdf))
        for name in skus:
            n = " ".join(name.split())
            sku[n]= skus[name]
        brands = pickle.load(open('docu/brands4.p','rb'))
        x=0
        tags = dict()
        d=enchant.Dict("en_US")
        for txt in qdf.clean_text:
           
                
            #print (x)
            t1 = re.sub(r'[^a-zA-Z0-9]',' ',str(txt).lower()).split()
            naam =[]
            ID = []
            links=[]
            #print (x)
            #txt_s = txt.split()
            for ws in range(5,0,-1):
                t1 = re.sub(r'[^a-zA-Z0-9]',' ',str(t1).lower())
                t1 = ' '.join(t1.split())
                t2 = t1.split()
                for i in range(len(t2)-ws):
                    name = " ".join([i for i in t2[i:i+ws]])
                    #print (name)
                    if name in skus:
                        
                        naam.append(name)
                        ID.append(skus[name])
                        t1 = t1.replace('%s'%name,'')
        

                        if otc[otc.id==int(skus[name])].reset_index(drop=True).is_otc[0]==True:
                            links.append('https://www.1mg.com/otc/%s-otc%s'%('-'.join(name.split()),skus[name]))
                        else:
                            links.append('https://www.1mg.com/drugs/%s-%s'%('-'.join(name.split()),skus[name]))
                        
            
                    
                    
                    if name in brands:
                        #print (name)
                        naam.append(name)
                        t1 = t1.replace('%s'%name,'')
        

                        
                        links.append('https://www.1mg.com/brands/%s-%s'%('-'.join(name.split()),base64.urlsafe_b64encode(('%s'%name).lower().encode('utf-8')).decode('utf-8')))
                   
            tags[x]=(','.join(ID),','.join(naam),','.join(links))
            if x%1000==0:
                print (x)
            x+=1
        qdf['sku_id']= [x[0] for x in list(tags.values())]
        qdf['skus']= [x[1] for x in list(tags.values())]
        qdf['links']= [x[2] for x in list(tags.values())]
        #print (qdf.isnull().sum())
        #print (qdf.dtypes)
        #print (qdf[qdf.id==11910])
        INDEX=[]
        for i in list(set(qdf.id)):
            if len(qdf[qdf.id==i])==1:
               # print (i)
                INDEX.append(i)
        qdf.drop(qdf[qdf['id'].isin(INDEX)].index,inplace=True)
        print (len(qdf))
        qdf['id']=qdf['id'].astype(int)
        print (qdf.dtypes)
        #print (qdf.columns)
       # qdf['doctor_id']=qdf['doctor_id'].astype(int)
        #qdf=qdf.drop(qdf.columns[:1],axis=1)
        return (qdf)
    
    def create_bag_of_centroids( self,wordlist, word_centroid_map ):        
    #
        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map
        num_centroids = max( word_centroid_map.values() ) + 1
        #
        # Pre-allocate the bag of centroids vector (for speed)
        bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
        #
        # Loop over the words in the review. If the word is in the vocabulary,
        # find which cluster it belongs to, and increment that cluster count 
        # by one
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1
        #
        # Return the "bag of centroids"
        return (bag_of_centroids)
        
        
    
    def nlp(self):
       # nltk.download()
        processed_df = pd.read_csv('anonymised3.csv')
        #print ('i am here')
        #processed_df.text1 = processed_df.text
        processed_df.text = processed_df.text.apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',str(x).lower()))#.apply(lambda y:re.sub("^\d+\s|\s\d+\s|\s\d+$",' ',y))
        a=processed_df.groupby(['id'])['text'].apply(' '.join)
        df1=a.reset_index()

        remv = ['hi','hello','hey','sir','mam','doctor','doc','please','plz','dear','suggest','http','1mgayush','www','com','medicine','medicines','daily','like','feel','tell','need','u','get','treatment','help','water','years','also','n','take','problem','drops','times','day']
        stops = (stopwords.words("english"))
        
        lmtzr = WordNetLemmatizer()
        l=[]
        d_remv = dict()
        
        remv_words = stops+remv
        for word in remv_words:
            #print (word)
            d_remv[word]=0

        for line in df1.text:

            line=' '.join(line.split())
            line=line.strip().split(' ')
            #line = "9XX ".join(filter(lambda x:x[0]=='9', line))
            #print (line)
            words = [w for w in line if not w in (d_remv)]
            li=[]
            for word in words:
                li.append(lmtzr.lemmatize(word))
            l.append(li)
            
        print ('chill about to make vector')           
        num_features = 60  # Word vector dimensionality                      
        min_word_count = 6   # Minimum word count                        
        num_workers = 4      # Number of threads to run in parallel
        context = 5      # Context window size                                                                                    
        downsampling = 1e-3  # Downsample setting for frequent words

        #print ()"Training model..."
        model = word2vec.Word2Vec(l, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling)


        print (model.most_similar('sex'))
        
        from sklearn.cluster import KMeans

        # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
        # average of 5 words per cluster
        word_vectors = model.wv.syn0
        #num_clusters = int(word_vectors.shape[0] / 20)
        num_clusters = int(math.sqrt(len(word_vectors)/2))

        # Initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans( n_clusters = num_clusters )
        idx = kmeans_clustering.fit_predict( word_vectors )
        
        # Pre-allocate an array for the training set bags of centroids (for speed)
        train_centroids = np.zeros( (df1["text"].size, num_clusters), \
            dtype="float32" )
        word_centroid_map = dict(zip( model.wv.index2word, idx ))
        # Transform the training set reviews into bags of centroids
        counter = 0
        for text in l:
            train_centroids[counter] = self.create_bag_of_centroids( text, \
                word_centroid_map )
            counter += 1
        
        
        d = dict()  #id to conv_id
        for i in range(len(df1)):
            d[i]=df1.id[i]

        d_inv = dict()  #conv_id to id
        for i in range(len(df1)):
            d_inv[df1.id[i]]=i
        #pickle.dump(d, open("docu/dict.p", "wb"))    
        #pickle.dump(d_inv, open("docu/dict_inv.p", "wb"))
        #np.save('docu/array',train_centroids)
if __name__ == "__main__":
    conn = connect()
    #df = connect.cleaned(conn)
    #df=connect.anonymised(conn)
    #print (df.head(5))
    connect.nlp(conn)
    #df.to_csv('df_updated1.csv',index=False)

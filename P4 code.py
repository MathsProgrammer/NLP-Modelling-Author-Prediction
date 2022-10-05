import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import string
import seaborn as sns
import scipy.stats as stats





knowntexts = pd.read_excel('p4data.xlsx')
unknowntext = pd.read_excel('p4unknown.xlsx')
bigdata = pd.read_excel('Big data_1.xlsx')



#print(knowntexts)
#knowntexts.shape
#print(data[:1])

vectorizer = CountVectorizer(stop_words="english")
#print(vectorizer)
all_features = vectorizer.fit_transform(knowntexts.message)
all_features.shape #number of words
vectorizer.vocabulary_


classifier = MultinomialNB()
classifier.fit(all_features, knowntexts.Author)

unknown_vectorized = vectorizer.transform(unknowntext.message)
big_vectorized = vectorizer.transform(bigdata.message)
print(classifier.predict(unknown_vectorized)) #thinks its author 1
print(classifier.predict(big_vectorized))

print(len(classifier.predict(big_vectorized)))
print(classifier.predict_proba(unknown_vectorized))



logistic = LogisticRegression()
logistic.fit(all_features, knowntexts.Author)
print(logistic.predict(unknown_vectorized))
k=logistic.predict(big_vectorized)


print(logistic.predict_proba(unknown_vectorized))
just_message =knowntexts['message']
just_message1=unknowntext["message"]

support = SVC(kernel="rbf")
support.fit(all_features, knowntexts.Author)
print(support.predict(unknown_vectorized))
print(support.predict(big_vectorized))












wordcloud1 = WordCloud(colormap="Blues").generate(just_message[0]) 
wordcloud2 = WordCloud(colormap="Blues").generate(just_message[1]) 
wordcloud3 = WordCloud(colormap="Blues").generate(just_message[2])
wordcloud4 = WordCloud(colormap="Blues").generate(just_message1[0])




plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.show()


plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis("off")
plt.show()




#Distribution of authors in data set, bar graph of how many pieces are taken from them
x = ["Jane Austin", "Charlotte Bronte", "Emily Bronte"]   
y = [50, 30, 20]
plt.bar(x,y, width = 0.4)
plt.xlabel("Author")
plt.ylabel("Count of data")
plt.title("A plot of the distribution of authorship in the data set")
plt.show() 


########################## Chi-squared

####### Words per text + plot  from the adjusted?!?!?!?!?
author = ["Jane", "Charlotte", "Emily", "Anonymous"]
wordspertext = []
wordspertext1 =[]

for i in knowntexts.message:
    lex = i.split()
    wordspertext.append(len(lex))


unknownlength = unknowntext.message[0].split()    
wordspertext.append(len(unknownlength))

for i in bigdata.message:
    lex = i.split()
    wordspertext1.append(len(lex))
print(len(wordspertext1))   

sns.barplot(x=author, y=wordspertext, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Total words per passage', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=398, s="796", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=545, s="1091", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=458, s="917", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=205, s="410", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show()

#########  number of different words
difwords=[]
difwords1=[]

for i in knowntexts.message:
    differentwords = 0
    lexi=set(i.split())
    for words in lexi:
        differentwords+=1
    difwords.append(differentwords)  
    
    

unknowntextdifwords= set(unknowntext.message[0].split()) 
difwords.append(len(unknowntextdifwords))
####for big  ###done
for i in bigdata.message:
    differentwords = 0
    lexi=set(i.split())
    for words in lexi:
        differentwords+=1
    difwords1.append(differentwords)
print(len(difwords1))


print(difwords)
sns.barplot(x=author, y=difwords, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Unique words per passage', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=191, s="383", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=296, s="592", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=282, s="564", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=127, s="255", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show()    

######words per sentence ### done
wordspersentence =[]
wordspersentence1 =[]

sentencepertext=[]
sentencepertext1=[]
for x in knowntexts.message:
    numsentences=0
    for i in x:
        if i =='.' or i == "!" or i == "?":
            numsentences+=1
    sentencepertext.append(numsentences)
numsentences=0    
for i in unknowntext.message[0]:
    if i =='.' or i == "!" or i == "?":
        numsentences+=1
sentencepertext.append(numsentences)
        

for i in range(len(sentencepertext)):
    wordspersentence.append(wordspertext[i]/sentencepertext[i])

print(wordspersentence)

for x in bigdata.message:
    numsentences=0
    for i in x:
        if i =='.' or i == "!" or i == "?":
            numsentences+=1
    if numsentences == 0:
        numsentences = 10
    sentencepertext1.append(numsentences)
print(sentencepertext1)    

for i in range(len(sentencepertext1)):
    wordspersentence1.append(wordspertext1[i]/sentencepertext1[i]) 
    
print(len(sentencepertext1))

sns.barplot(x=author, y=wordspersentence, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Average words per sentence', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=26.5, s="53", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=15.15, s="30", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=11.46, s="23", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=18.6, s="37", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show() 

######### average word length
averagewordlen=[]
totalchar=[]
averagewordlen1=[]
totalchar1=[]
for i in knowntexts.message:
    totalchar.append(len(i))
totalchar.append(len(unknowntext.message[0]))
for i in range(len(sentencepertext)):
    averagewordlen.append(totalchar[i]/wordspertext[i])
    
print(averagewordlen)    

for i in bigdata.message:
    totalchar1.append(len(i)) 

   
for i in range(len(totalchar1)):
    averagewordlen1.append(totalchar1[i]/wordspertext1[i])
    
sns.barplot(x=author, y=averagewordlen, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Average word length', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=2.85, s="5.70", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=2.81, s="5.62", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=2.94, s="5.88", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=2.748, s="5.50", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show() 

print(len(averagewordlen1))    

######## characters per sentence
charpersentence=[]
charpersentence1=[]
for i in range(len(sentencepertext)):
    charpersentence.append(totalchar[i]/sentencepertext[i])
    

for i in range(len(averagewordlen1)):
    charpersentence1.append(totalchar1[i]/sentencepertext1[i])    

print(len(charpersentence1))  

print(charpersentence)
sns.barplot(x=author, y=charpersentence, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Characters per sentence', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=151.3, s="303", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=85.15, s="170", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=68.5, s="135", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=102.45, s="205", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show()

print(totalchar)
sns.barplot(x=author, y=totalchar, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Characters per passage', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=2270, s="4539", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=3068, s="6132", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=2696, s="5392", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=1127, s="2254", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show()


####### punctuation exluding quotation and full stops per sentence
punctpersentence=[] 
puncttotal=[]
punctpersentence1=[]
puncttotal1=[]
for i in knowntexts.message:
    punct=0
    for k in i:
        if  k == "!" or k == "?" or k ==',' or k == '' or k == "(" or k == ")" or k == ";" or k == "-":
            punct+=1
    puncttotal.append(punct)  
punct=0   
for k in unknowntext.message[0]:
    if  k == "!" or k == "?" or k ==',' or k == '' or k == "(" or k == ")" or k == ";" or k == "-":
            punct+=1
puncttotal.append(punct)
    
for i in range(len(sentencepertext)):
    punctpersentence.append(puncttotal[i]/sentencepertext[i])
   
print(punctpersentence)        
            
for i in bigdata.message:
    punct=0
    for k in i:
        if  k == "!" or k == "?" or k ==',' or k == '' or k == "(" or k == ")" or k == ";":
            punct+=1
    puncttotal1.append(punct) 

for i in range(len(puncttotal1)):
    punctpersentence1.append(puncttotal1[i]/sentencepertext1[i])
print(len(punctpersentence1))    

print(punctpersentence)
sns.barplot(x=author, y=punctpersentence, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Punctuation per sentence', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=2.56, s="5.1", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=1.94, s="3.86", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=1.35, s="1.35", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=2.5, s="5", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show()

####### names per passage # how many capitals - the number of sentences to stop first capital-discuss, proper nouns
###needs to be per sentence
namespersentence=[]
totalcap=[]
namespersentence1=[]
totalcap1=[]
for i in knowntexts.message:
    count = 0
    for k in i:
        if k.isupper():
            count += 1
    totalcap.append(count)  
count=0   
for i in unknowntext.message[0]:
    if i.isupper():
        count +=1
totalcap.append(count)    
print(totalcap)    
print(sentencepertext)
for i in range(len(totalcap)):
    namespersentence.append(totalcap[i]/sentencepertext[i])
             
print(namespersentence)  



for i in bigdata.message:
    count = 0
    for k in i:
        if k.isupper():
            count += 1
    totalcap1.append(count) 
    
for i in range(len(totalcap1)):
    namespersentence1.append(totalcap1[i]/sentencepertext1[i])
    
print(len(namespersentence1))    

sns.barplot(x=author, y=namespersentence, 
           palette="Set1")
plt.xlabel('')
plt.ylabel('')
plt.title('Names and proper nouns per sentence', size=18, color='k')
plt.xticks(size=14, color='#4f4e4e')
plt.yticks([], [])
plt.text(x=0, y=1.4, s="2.8", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=1.1, s="2.2", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=0.975, s="1.95", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=1.315, s="2.6", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True);
plt.show()



########### Make vectors for chi, small case

          
table =[difwords, wordspersentence, averagewordlen, charpersentence, punctpersentence, namespersentence]
print(table)

variables = ["Unique words", "Words per sentence", "average word length", "letters per sentence", "punctuation per sentence", "names per sentence"]
author = ["Jane Austen", "Charlotte Bronte", "Emily Bronte", "Unknown"]
df = pd.DataFrame([difwords, wordspersentence, averagewordlen, charpersentence, punctpersentence, namespersentence],  variables,author)
    
print(df)    
print(df["Unknown"])   
Austen=stats.chisquare(f_obs=df["Unknown"], f_exp=df["Jane Austen"]) 
Charlotte =stats.chisquare(f_obs=df["Unknown"], f_exp=df["Charlotte Bronte"])  
Emily=stats.chisquare(f_obs=df["Unknown"], f_exp=df["Emily Bronte"]) 
print(Austen[1])
print(Charlotte[1])
print(Emily[1])
min(Austen[1], Charlotte[1], Emily[1])
########### for the 100 big ones
text=[]
for i in range(100):
    text.append(i+1)
print(text)    

bigdf = pd.DataFrame([difwords1, wordspersentence1, averagewordlen1, charpersentence1, punctpersentence1, namespersentence1],  variables,text)
print(bigdf)
###### lower p value means it is more similar
Austen=stats.chisquare(f_obs=bigdf[1], f_exp=df["Jane Austen"])
print(Austen[1])
authorchi=[]

for i in range(1,101):
    Austen=stats.chisquare(f_obs=bigdf[i], f_exp=df["Jane Austen"]) 
    Charlotte =stats.chisquare(f_obs=bigdf[i], f_exp=df["Charlotte Bronte"])  
    Emily=stats.chisquare(f_obs=bigdf[i], f_exp=df["Emily Bronte"]) 
    if min(Austen[1],Charlotte[1],Emily[1])==Austen[1]:
        authorchi.append("Jane Austen")
    elif min(Austen[1],Charlotte[1],Emily[1])==Charlotte[1]:
        authorchi.append("Charlotte Bronte")
    else:
        authorchi.append("Emily Bronte") 
        
print(authorchi)  
count=0
for i in range(100):
    if authorchi[i] == "Jane Austen":
        count +=1
print(count)    

#chi^2 confusion
x_axis_labels = (['Jane','Charlotte','Emily'])
y_axis_labels = (['Jane','Charlotte','Emily'])
mc=[[.52,.32,.16],[.0667,.5666,.3666],[.1,.3,.6]] 
plt.figure(figsize=(6,6))
sns.heatmap(mc,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True,fmt='.2%',linewidths=.5,square=True,cmap="Blues")
plt.ylabel("Real author")
plt.xlabel("Predicted author")
all_sample_title="Chi-squared: accuracy = 55%"
plt.title(all_sample_title,size=15)
plt.show()  

#log likelihood confusion
x_axis_labels = (['Jane','Charlotte','Emily'])
y_axis_labels = (['Jane','Charlotte','Emily'])
mc=[[.74,.16,.1],[.03333,.7333,.2],[.15,.2,.65]] #37,8,5   #1,22,6 #3,4,13
plt.figure(figsize=(6,6))
sns.heatmap(mc,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True,fmt='.2%',linewidths=.5,square=True,cmap="Blues")
plt.ylabel("Real author")
plt.xlabel("Predicted author")
all_sample_title="Log-likelihood: accuracy = 72%"
plt.title(all_sample_title,size=15)
plt.show() 

#Naive Bayes
x_axis_labels = (['Jane','Charlotte','Emily'])
y_axis_labels = (['Jane','Charlotte','Emily'])
mc=[[.66,.16,.18],[.13333,.6333,.2333],[.2,.2,.6]] #33,8,9   #4,19,7 #4,4,12
plt.figure(figsize=(6,6))
sns.heatmap(mc,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True,fmt='.2%',linewidths=.5,square=True,cmap="Blues")
plt.ylabel("Real author")
plt.xlabel("Predicted author")
all_sample_title="Naive Bayes: accuracy = 64%"
plt.title(all_sample_title,size=15)
plt.show()

#Logistic regression
x_axis_labels = (['Jane','Charlotte','Emily'])
y_axis_labels = (['Jane','Charlotte','Emily'])
mc=[[.54,.22,.24],[.23333,.5666,.2],[.2,.16,.64]] #27,11,12   #7,17,6 #4,3,13
plt.figure(figsize=(6,6))
sns.heatmap(mc,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True,fmt='.2%',linewidths=.5,square=True,cmap="Blues")
plt.ylabel("Real author")
plt.xlabel("Predicted author")
all_sample_title="Logistic regression: accuracy = 57%"
plt.title(all_sample_title,size=15)
plt.show()


#SVM
x_axis_labels = (['Jane','Charlotte','Emily'])
y_axis_labels = (['Jane','Charlotte','Emily'])
mc=[[.96,.04,.0],[.8666,.1,.0333],[.85,.05,.1]] #48,2,0   #26,3,1 #17,1,2
plt.figure(figsize=(6,6))
sns.heatmap(mc,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True,fmt='.2%',linewidths=.5,square=True,cmap="Blues")
plt.ylabel("Real author")
plt.xlabel("Predicted author")
all_sample_title="SVM: accuracy = 53%"
plt.title(all_sample_title,size=15)
plt.show()


count=0
for i in unknowntext.message[0].split():
    if len(i) > 8:
        count+=1
print(sentencepertext)
print(count/11)    

#############################log-likelihood

print(knowntexts.message[0])

#strips punctuation

A = knowntexts.message[0]
A = A.strip(string.punctuation)

B = knowntexts.message[1]
B = B.strip(string.punctuation)

C = knowntexts.message[2]
C = C.strip(string.punctuation)

D = unknowntext.message[0]
D = D.strip(string.punctuation)


state = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p","q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " ", ",", ")", ";", ":", "'", "‘"," ","- "]
#create word2id and id2word dictionary
char2id_dict = {}
for index, char in enumerate(state):
    char2id_dict[char] = index

#makes them lower case
def matrix_maker(text):
    transition_matrix = np.zeros((35, 35))
    
    for i in range(len(text)-1):
        current_char = text[i].lower()
        next_char = text[i+1].lower()   # makes the chosen text lower case
        
        if (current_char in state) & (next_char in state):
            current_char_id = char2id_dict[current_char]
            next_char_id = char2id_dict[next_char]
            transition_matrix[current_char_id][next_char_id] = transition_matrix[current_char_id][next_char_id] + 1
            sum_of_each_row_all  = np.sum(transition_matrix, 1)
    
    for i in range (35):
            single_row_sum = sum_of_each_row_all[i]
            if (sum_of_each_row_all [i] == 0):
                single_row_sum = 1
            
            transition_matrix[ i,: ] =  transition_matrix[ i,: ] / single_row_sum
    return transition_matrix

MatrixA = matrix_maker(A)
MatrixB = matrix_maker(B)
MatrixC = matrix_maker(C)



log_likelihood_austen = 0
log_likelihood_eyre = 0
log_likelihood_bronte = 0
author=[]

for k in bigdata.message:
    k = k.strip(string.punctuation)
    log_likelihood_austen = 0
    log_likelihood_eyre = 0
    log_likelihood_bronte = 0
    for i in range(len(k)-1):
        current_char = k[i].lower()
        next_char = k[i+1].lower()    #makes unammed all lower case
    if (current_char in state) & (next_char in state):
            current_char_id = char2id_dict[current_char]
            next_char_id = char2id_dict[next_char]    # assign index of character from state
            if MatrixA[current_char_id][next_char_id] != 0:
                log_likelihood_austen += np.log(MatrixA[current_char_id][next_char_id])
            if MatrixB[current_char_id][next_char_id] != 0:
                log_likelihood_eyre += np.log(MatrixB[current_char_id][next_char_id])
            if MatrixC[current_char_id][next_char_id] != 0:
                log_likelihood_bronte += np.log(MatrixC[current_char_id][next_char_id])
    if min(log_likelihood_austen, log_likelihood_eyre, log_likelihood_bronte) == log_likelihood_austen:
        author.append("Jane Austen")
    elif min(log_likelihood_austen, log_likelihood_eyre, log_likelihood_bronte) == log_likelihood_eyre:    
        author.append("Charlotte Bronte")
    else:
        author.append("Emily Bronte")
print(author)
count=0
for i in range(100):
    if author[i] == "Emily Bronte":
        count +=1
print(count)    

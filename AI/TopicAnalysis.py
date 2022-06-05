'''from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
x, y = load_iris(return_X_y=True)
print(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
print("Number of mislabeled points out of a total %d points: %d"%(x_test.shape[0], (y_test != y_pred).sum()))
'''

print("Loading...")


training = False #to be set to false when not training
#for the lists later: no. of blanks is number of topics because yes. Each topic is assigned a certain "id". 
data = [[],[],[],[],[],[],[],[],[],[],[]]
freqs = [[],[],[],[],[],[],[],[],[],[],[]]
chances = []
topiclookup = ['comarchitecture', 'algs', 'flowcharts', 'python', 'validation', 'safeuse', 'numbersystems', 'logiccircuits', 'spreadsheets', 'networks', "spam"] #index of occurrence of topic name in list is the topic "id"

#reading machine learning data into the lists "data" and "freqs", also I just gave up and made weird variable names
fd = open("TAMLdata.txt", 'r')
ff = open("TAMLfreqs.txt", 'r')
for x in range(len(data)):
    data[x] = fd.readline().split()
    freqs[x] = list(map(int, ff.readline().split())) #this is just a way to read the values into a list but make them all int.
print(data)
print(freqs)

# import module for tokenization, lemmatization and stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import GaussianNB -- I tried using this module to do nb but uh 
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#^ was because I was lazy to code out smtg that gives all the letters of the alphabet lol

'''
def input(s=''): #This is a python builtin but then I made this to override it to perform auto training :)
    fin = open('autotraintestcases.txt','r')
    print(s)
    fin.close()
    return fin.readline()'''

def filtertext(): 
    # input the text
    text = input("Text: ")
    word_tokenize(text)

    tokens = word_tokenize(text)
    
    # initiate the lemmatizer object
    lemmatizer = WordNetLemmatizer()

    #print("rocks :", lemmatizer.lemmatize("rocks"))
    new_tokens = [] 
    for token in tokens: 
        new_tokens.append(lemmatizer.lemmatize(token))
    #print(new_tokens)
    
    #assign to globally set stopwords to a local set
    stop_words = set(stopwords.words('english')+[''])
    #filter the stopwords and non-alphanumeric characters from the token
    filtered_tokens = [''.join(ch for ch in token if ch in letters) for token in new_tokens if not ''.join(ch for ch in token if ch in letters).lower() in stop_words]

    return filtered_tokens
    '''vectorizer = TfidfVectorizer()
    vectorizer.fit(filtered_tokens)
    print(vectorizer.vocabulary_)
    print(vectorizer.idf_)
    vector = vectorizer.transform([filtered_tokens[0]])
    print(vector.shape)
    print(vector.toarray())
    return vector.toarray()'''

def test():
    global training
    global data
    global freqs
    t = filtertext()
    if training:
        rawtopics = input("Topics: ").split() #just a way of reading the input into the list "rawtopics" so that yes
        topics = [topiclookup.index(x) for x in rawtopics] #translating into topic code
        for i in t:
            for a in topics: #to each topic in there
                if i in data[a]:
                    #print(data) just some testing stuff because this was wrong at one point
                    #print(data[a])
                    #print(i)
                    #print(freqs)
                    freqs[a][data[a].index(i)] += 1
                else:
                    data[a].append(i)
                    freqs[a].append(1)
        print(data)
        print(freqs)
        return
    else:
        chances = [1,1,1]
        for i in t:
            for a in range(len(topiclookup)): 
                if i in data[a]:
                    chances[a] *= (freqs[a][data[a].index(i)]+1)/(sum(freqs[a]))
                else:
                    chances[a] *= 1/(sum(freqs[a]))
        return topiclookup[chances.index(max(chances))]

def settrain(b):
    global training
    training = b
    return

def save():
    wd = open("TAMLdata.txt", 'w')
    wf = open("TAMLfreqs.txt", 'w')
    global data
    global freqs
    for d in data:
        for i in d:
            wd.write(str(i)+' ')
        wd.write('\n')
    for f in freqs:
        for i in f:
            wf.write(str(i)+' ')
        wf.write('\n')
    wd.close()
    wf.close()
    return

print("Done!\n\n\nTopic test")


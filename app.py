from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3

#from tkinter import *
#from tkinter import messagebox

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
data = [[],[],[],[],[],[],[],[],[],[],[]]
freqs = [[],[],[],[],[],[],[],[],[],[],[]]
chances = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
topiclookup = ['comarchitecture', 'algs', 'flowcharts', 'python', 'validation', 'safeuse', 'numbersystems', 'logiccircuits', 'spreadsheets', 'networks', 'spam'] #index of occurrence of topic name in list is the topic "id"
fd = open("TAMLdata.txt", 'r')
ff = open("TAMLfreqs.txt", 'r')
for x in range(len(data)):
    data[x] = fd.readline().split()
    freqs[x] = list(map(int, ff.readline().split())) 


cur_dir = os.path.dirname(os.path.abspath(__file__))
db_file = os.path.join(cur_dir, 'bulletin.db')
app = Flask(__name__)
os.path.join(cur_dir, "TAMLfreqs.txt")
os.path.join(cur_dir, "TAMLdata.txt")

## in case db is locked, run this
# db = sqlite3.connect(db_file) 
# db.close()


def filtertext(text):
    global letters
    word_tokenize(text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    new_tokens = [] 
    for token in tokens: 
        new_tokens.append(lemmatizer.lemmatize(token))
    stop_words = set(stopwords.words('english')+[''])
    filtered_tokens = [''.join(ch for ch in token if ch in letters) for token in new_tokens if not ''.join(ch for ch in token if ch in letters).lower() in stop_words]
    return filtered_tokens

def test(text):
    global training
    global data
    global freqs
    t = filtertext(text)
    chances = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in t:
        for a in range(len(topiclookup)-1): 
            if i in data[a]:
                chances[a] *= (freqs[a][data[a].index(i)]+1)/(sum(freqs[a]))
            else:
                chances[a] *= 1/(sum(freqs[a]))
    chances[-1] = 0.0123**len(t)
    return topiclookup[chances.index(max(chances))]


@app.route('/')
def login():
    return render_template('login.html')# login page


@app.route('/<username>/', methods = ['GET', 'POST'])
def class_code(username):
    if request.method == 'GET': # get class code page
        return render_template('class_code_dashboard.html')

    else: # show dashboard.
        class_code = request.form['code'] 
        if username == 'student':
            return render_template('student_after_class_code.html', class_code = class_code, username = username)
        else:
            return render_template('teacher_after_class_code.html', class_code = class_code, username = username)

@app.route('/<username>/<class_code>/submit_questions/')
def submit_questions(username, class_code): # takes in the qns
    return render_template('submit_questions.html', class_code = class_code, username = username)

@app.route('/<username>/<class_code>/bulletin/', methods = ['GET', 'POST'])
def bulletin(username, class_code): # user doesn't ask qns, directly go bulletin.
    if request.method == 'GET':
        dataset = []
        topics = []
        db = sqlite3.connect(db_file) 

        query = """
        SELECT topic FROM bulletin
        WHERE ClassCode = ?
        """
        cursor = db.execute(query, (class_code,))
        unprocessed_topics = cursor.fetchall() # all the topics, including repeats
        for topic in unprocessed_topics:
            if topic[0] not in topics:
                topics.append(topic[0])
        # print(topics)
        for topic in topics:
            query = """
            SELECT ID, LessonDate, Question FROM bulletin
            WHERE ClassCode = ? 
            AND Topic = ?
            """
            cursor = db.execute(query, tuple([class_code, topic]))
            temp_rows = cursor.fetchall() # data for respective topics
            # print(temp_rows)
            temp_lst = []
            for row in temp_rows:
                temp_lst.append([row[0], row[1], row[2]])
            dataset.append([topic, temp_lst])
        # print(dataset)
        db.close()
        return render_template('bulletin.html', dataset = dataset, class_code = class_code, username = username, topics = topics)
    else: # POST, student has asked a qns and show the bulletin afterwards.
        db = sqlite3.connect(db_file)
        topics = []
        l_date = request.form['l_date']
        qns = request.form['qns'] 
        
        topic = test(qns)
        
        if topic == 'spam':
            #return render_template('submit_questions.html', class_code = class_code, username = username, errormsg = messagebox.showwarning("showwarning", "Warning"))
            return render_template('submit_questions.html', class_code = class_code, username = username, errormsg = "Spam detected! If this was not intentional, please rephrase your question. ")
        
        query = """
        INSERT INTO bulletin (ClassCode, Question, LessonDate, Topic)
        VALUES (?,?,?,?)
        """
        db.execute(query, (class_code, qns, l_date, topic))
        db.commit()

        # select data part 
        dataset = []
        db = sqlite3.connect(db_file) 

        # gather a list of tuples of topics
        query = """
        SELECT topic FROM bulletin
        WHERE ClassCode = ?
        """
        cursor = db.execute(query, (class_code,))
        unprocessed_topics = cursor.fetchall() # all the topics, including repeats
        for topic in unprocessed_topics:
            if topic[0] not in topics:
                topics.append(topic[0])
        # print(topics)
        for topic in topics:
            query = """
            SELECT ID, LessonDate, Question FROM bulletin
            WHERE ClassCode = ? 
            AND Topic = ?
            """
            cursor = db.execute(query, tuple([class_code, topic]))
            temp_rows = cursor.fetchall() # data for respective topics
            # print(temp_rows)
            temp_lst = []
            for row in temp_rows:
                temp_lst.append([row[0], row[1], row[2]])
            dataset.append([topic, temp_lst])
        # print(dataset)
        db.close()
        return render_template('bulletin.html', dataset = dataset, class_code = class_code, username = username, topics = topics)

@app.route('/<username>/<class_code>/delete/<int:id>', methods = ['POST'])
def delete(username, class_code, id): 
    if username == 'student':
        return 'Access Denied'
    else: # teacher
        # delete query, update database
        db = sqlite3.connect(db_file)
        query = """
        DELETE FROM bulletin
        WHERE ID = ?
        """
        db.execute(query, (id,))
        db.commit() 
        db.close()       
        return redirect(url_for('bulletin', class_code = class_code, username = username))

if __name__ == '__main__':
    app.run(debug=True) 
    # set debug to False if you are using python IDLE as 
    # your IDE.

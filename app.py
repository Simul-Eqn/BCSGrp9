from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3

cur_dir = os.path.dirname(os.path.abspath(__file__))
db_file = os.path.join(cur_dir, 'bulletin.db')
app = Flask(__name__)

## in case db is locked, run this
# db = sqlite3.connect(db_file) 
# db.close()

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
        db = sqlite3.connect(db_file) 
        query = """
        SELECT ID, LessonDate, Question FROM bulletin
        WHERE ClassCode = ?
        """
        cursor = db.execute(query, (class_code,))
        dataset = cursor.fetchall()
        db.close()
        return render_template('bulletin.html', dataset = dataset, class_code = class_code, username = username)
    else: # POST, student has asked a qns and show the bulletin afterwards.
        db = sqlite3.connect(db_file)
        l_date = request.form['l_date']
        qns = request.form['qns'] 

        query = """
        INSERT INTO bulletin (ClassCode, Question, LessonDate)
        VALUES (?,?,?)
        """
        db.execute(query, (class_code, qns, l_date))

        db.commit()

        query = """
        SELECT ID, LessonDate, Question FROM bulletin
        WHERE ClassCode = ?
        """
        cursor = db.execute(query, (class_code,))
        dataset = cursor.fetchall()
        db.close()
        return render_template('bulletin.html', dataset = dataset, class_code = class_code, username = username)

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
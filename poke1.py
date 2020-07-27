# importing modules
from flask import Flask, request, render_template
import numpy as np
from sentence_transformers import SentenceTransformer
import praw

model = SentenceTransformer('bert-base-nli-mean-tokens')
r = praw.Reddit(client_id='cfBg3wRLP9kY4g',
                client_secret='08gBhc1GPgXL64lis5Bn7tXIvsY',
                user_agent='Tim')

def loadQuestionsLinks():
    questions = []
    answers = []
    with open("questionsandlinks.txt",encoding='utf-8') as myfile:
        data = myfile.readlines()
    for a in data:
        betw = a.split("\t")
        questions.append(betw[0])
        answers.append(betw[1].replace("\n",""))
    return questions, answers

# declaring app name
app = Flask(__name__)
questions, answers = loadQuestionsLinks()
doc_vecs = np.load("sentenceencodingsquestionsbig.npy")

# making list of pokemons
Pokemons = ["Pikachu", "Charizard", "Squirtle", "Jigglypuff",
            "Bulbasaur", "Gengar", "Charmander", "Mew", "Lugia", "Gyarados"]

def getRecommends(text1):
    rel_questions = []
    rel_links = []
    query = text1
    query_vec = model.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:4]
    for idx in topk_idx:
        rel_questions.append(questions[idx])
        rel_links.append(answers[idx])
    #for a in len(rel_questions):
    return rel_questions, rel_links

def getTopComment(links):
    tops = []
    for a in links:
        url = a
        submission = r.submission(url=a)
        submission.comment_sort = "top"
        comment = [comment.body for comment in submission.comments if hasattr(comment, "body")]
        tops.append(comment[0])
    return tops

@app.route('/')
def home():
    return render_template('search2.html')

@app.route('/', methods=['POST'])
def getQuestions():
    text1 = request.form['text1']
    Pokemons = getRecommends(text1)



@app.route('/poketest')
def result_site():
    text1 = request.args.get("q")
    results, links = getRecommends(text1)
    tops = getTopComment(links)
    return render_template("poketest.html", len=len(results), Pokemons=results, Links=links, Tops=tops)

# defining home page
"""
@app.route('/')
def homepage():
    # returning index.html and list
    # and length of list to html page
    return render_template("poketest.html", len=len(Pokemons), Pokemons=Pokemons)
"""

app.run(use_reloader=True, debug=True)


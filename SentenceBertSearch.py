# Sentence_Bert Version of ELI5_Search
from flask import Flask, request, render_template
import numpy as np
from sentence_transformers import SentenceTransformer
import praw
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity

# Load Sentence-Bert Model (approx. 400 MB)
model = SentenceTransformer('bert-base-nli-mean-tokens')
# Reddit Login tokens
r = praw.Reddit(client_id='XXXX',
                client_secret='XXX',
                user_agent='XXX')
# ADD YOUR API KEY

#Load Questions and Links
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

# creating app
app = Flask(__name__)

# Load Questions and Links
questions, answers = loadQuestionsLinks()

# Load pre-encoded questions
doc_vecs = np.load("sentenceencodingsquestionsbig.npy")

# Encode User-Query and calculate cosine similarity between query and all questions.
# Return best 4 fitting questions and their answer links.
def getRecommends(text1):
    rel_questions = []
    rel_links = []
    query = text1
    query_vec = model.encode([query])[0]
    # compute normalized dot product as score
    #score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    score = cosine_similarity([query_vec],doc_vecs)
    topk_idx = np.argsort(score)[0][::-1][:4]
    #topk_idx = np.argsort(score)[::-1][:4]
    print(topk_idx)
    for idx in topk_idx:
        rel_questions.append(questions[idx])
        rel_links.append(answers[idx])
    #for a in len(rel_questions):
    return rel_questions, rel_links

# Crawl top-answers for questions from reddit
def getTopComment(links):
    tops = []
    for a in links:
        try:
            url = a
            submission = r.submission(url=a)
            submission.comment_sort = "top"
            comment = [comment.body for comment in submission.comments if hasattr(comment, "body")]
            tops.append(comment[0])
        except:
            tops.append("LINK OUTDATED")
    return tops

# Render Search Website
@app.route('/')
def home():
    return render_template('home_bert.html')

# Compute distances and render results website
@app.route('/results_bert')
def result_site():
    text1 = request.args.get("q")
    results, links = getRecommends(text1)
    tops = getTopComment(links)
    return render_template("results_bert.html", len=len(results), Questions=results, Links=links, Tops=tops)

if __name__=='__main__':
    app.run(use_reloader=True, debug=True, port=5001)


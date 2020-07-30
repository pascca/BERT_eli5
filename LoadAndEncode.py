# This file converts the downloaded file explainlikeimfive_qalist.json from download_reddit_qalist.py to
# a text file 'questionsandlinks.txt' with only the questions and links.
# You can also encode all the questions with SentenceTransformer (sentence_bert) and save them as npy file for faster
# usage.
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# load Sentence-Bert model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# read explainlikeimfive_qalist.json and create questionsandlinks.txt, returns questions and links additionally
def readJsonQuestions():
    ques = []
    urls = []
    with open("explainlikeimfive_qalist.json") as myfile:
        data = json.load(myfile)
        for a in data[1:]:
            ques.append(a[1]['title'][0])
            urls.append(a[1]['url'])
    with open("questionsandlinks.txt","w",encoding="utf-8") as myfile:
        for a in range(len(ques)):
            myfile.write(ques[a]+"\t"+urls[a]+"\n")
    return ques,urls

# Encode the questions with the Sentence Bert model
def encodeSentsBert(summaries):
    sentence_embeddings = model.encode(summaries)
    with open('sentenceencodingsquestionsbig.npy', 'wb') as f:
        np.save(f, sentence_embeddings)
    return sentence_embeddings

# Method to load questions and links (answers) from previously created questionsandlinks.txt file
def loadQuestionsLinks():
    questions = []
    answers = []
    with open("questionsandlinks.txt",encoding='utf-8') as myfile:
        data = myfile.readlines()
    for a in tqdm(data):
        betw = a.split("\t")
        questions.append(betw[0])
        answers.append(betw[1].replace("\n",""))
    return questions, answers

# Run script with these settings for full preprocessing experience
if __name__=='__main__':
    questions, answers = readJsonQuestions()
    #questions, answers = loadQuestionsLinks()
    #questions,answers = readQuestionsAndAnswers()
    doc_vecs = encodeSentsBert(questions)





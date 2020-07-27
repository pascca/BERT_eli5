import json
from bert_serving.client import BertClient
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#bc = BertClient(ip='localhost',check_length=False)
model = SentenceTransformer('bert-base-nli-mean-tokens')

def readQuestionsAndAnswers():
    ques = []
    with open("elidataset/questions",encoding='utf-8') as myfile:
        qs = myfile.readlines()
        for a in qs:
            ques.append(a.replace("ELI5: ","").replace("ELI5:","").replace("[ELI5]","").replace("[ELI5] ","").replace("\n",""))
    with open("elidataset/answers",encoding='utf-8') as myfile:
        qs = myfile.read()
        answers = qs.split("###################################################################")
    return ques, answers[:-1]

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

#start server first bert-serving-start -model_dir C:\Users\d073873\PycharmProjects\MovieCompareTest\BERT\uncased_L-4_H-256_A-4 -num_worker=1
def encodeAll(summaries):
    summaryArrays = bc.encode(summaries)
    with open('bertencodingsquestions.npy', 'wb') as f:
        np.save(f, summaryArrays)
    return summaryArrays

def encodeSentsBert(summaries):
    sentence_embeddings = model.encode(summaries)
    with open('sentenceencodingsquestionsbig.npy', 'wb') as f:
        np.save(f, sentence_embeddings)
    return sentence_embeddings

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

#questions, answers = readJsonQuestions()
questions, answers = loadQuestionsLinks()
#questions,answers = readQuestionsAndAnswers()
#doc_vecs = encodeSentsBert(questions)
doc_vecs = np.load("sentenceencodingsquestionsbig.npy")
while True:
    query = input('your question: ')
    query_vec = model.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:5]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))
        print('> Answer: \t%s' % (answers[idx]))




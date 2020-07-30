# Finding Similar Questions from ELI5 with Sentence-Bert

This project builds a semantic-search web-service to find similar questions that were already asked by users of [Reddit's ELI5](https://www.reddit.com/r/explainlikeimfive/) subreddit by utilizing [Sentence-Bert](https://github.com/UKPLab/sentence-transformers). 

Users can ask questions about concepts and scienitific phenomena and get four similar questions that were already asked by the ELI5 community with the top-rated answer. Additionally, for every question-answer pair on the result page there is a 'Read more' link which directs the user to the original thread.

Example question 'What makes the earth rotate?" and results:

![alt text](https://github.com/pascca/BERT_eli5/blob/master/github_pics/ex_quest.png "ex_quest.png")

![alt text](https://github.com/pascca/BERT_eli5/blob/master/github_pics/ex_answers.png "ex_answers.png")

# How to use

1) Copy this repository.

2) Install requirements.txt with: `pip3 install -r requirements.txt`

3) Ask author of repository for encoded files and question-answer-pairs, namely `sentenceencodingsquestionsbig.npy` and `questionsandlinks.txt` -> Download and position the files on same hierarchy level as `SentenceBertSearch.py` and `TfidfSearch.py`

4) Run `SentenceBertSearch.py`(Sentence-Bert version) or `TfidfSearch.py`(TFIDF-Baseline) and go to and `http://127.0.0.1:5001/`(Sentence-Bert version) or `http://127.0.0.1:5000/`(TFIDF-Baseline)

5) Ask questions - get answers


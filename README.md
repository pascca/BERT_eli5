# Finding Similar Questions from ELI5 with Sentence-Bert

This project builds a semantic-search web-service to find similar questions that were already asked by users of [Reddit's ELI5](https://www.reddit.com/r/explainlikeimfive/) subreddit by utilizing [Sentence-Bert](https://github.com/UKPLab/sentence-transformers). 

Users can ask questions about concepts and scienitific phenomena and get four similar questions that were already asked by the ELI5 community with the top-rated answer. Additionally, for every question-answer pair on the result page there is a 'Read more' link which directs the user to the original thread.

Example question 'What makes the earth rotate?" and results:

![alt text](https://github.com/pascca/BERT_eli5/blob/master/github_pics/ex_quest.png "ex_quest.png")

![alt text](https://github.com/pascca/BERT_eli5/blob/master/github_pics/ex_answers.png "ex_answers.png")

# How to use

The code is optimized for Python 3.8.

1) Copy this repository.

2) Install torch version >= 1.2 fitting to your OS. This step can be problematic if you are using Windows and/or PyCharm.
  -> For Windows/PyCharm, try to run this command: `pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html`

3) Install requirements.txt with: `pip3 install -r requirements.txt`

4) Download `sentenceencodingsquestionsbig.npy`,`questionsandlinks.txt` and `explainlikeimfive_qalist.json` [here](https://drive.google.com/drive/folders/1HZ0Top-SVQc-FpwxIONPqvHD_gh7oR1d?usp=sharing) -> Position the files on the same hierarchy level as `SentenceBertSearch.py` and `TfidfSearch.py`

5) Run `SentenceBertSearch.py`(Sentence-Bert version) or `TfidfSearch.py`(TFIDF-Baseline) and open `http://127.0.0.1:5001/`(Sentence-Bert version) or `http://127.0.0.1:5000/`(TFIDF-Baseline)

6) Ask questions about concepts and phenomena (e.g. 'Why do we feel love?') - get layman-friendly answers

# Data

The initial data-file `explainlikeimfive_qalist.json` was created with a script from Facebook's project [ELI5](https://github.com/facebookresearch/ELI5) by using following command:

`python download_reddit_qalist.py -Q`

This script downloads all posts between 2011 and 2018 in a specific json-format. This web-app only needs the questions and links. These were extracted with the usage of `LoadAndEncode.py` into `questionsandlinks.txt`. The same script was used to create the sentence-bert encodings of the questions `sentenceencodingsquestionsbig.npy` which are utlized in `SentenceBertSearch.py`.

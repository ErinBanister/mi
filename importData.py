import sqlite3
import pandas as pd
import numpy as np
import re
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def saveTable(path, df, tablename):
    conn = sqlite3.connect(path)
    df.to_sql(tablename, conn, if_exists='replace', index=False)


def execute_query(path, query):
    conn = sqlite3.connect(path)
    results = conn.cursor().execute(query).fetchall()
    return results


def readSQL(path, query):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(query, conn)
    return df


def sqlQuery(spath, sources):
    sources_str = ["'%s'" % s for s in sources]
    query = "SELECT * FROM articles WHERE source IN (%s)" % ",".join(sources_str)
    return readSQL(spath, query)


def getLabels(lpath):
    lab = []
    with open(lpath) as fin:
        fin.readline()
        for line in fin:
            l = line.strip().split(",")
            ngs = float(l[10]) if len(l[10]) > 0 else np.nan
            ngc = int(l[11]) if len(l[11]) > 0 else np.nan
            mbl = l[33] if len(l[33]) > 0 else np.nan
            mbf = int(l[34]) if len(l[34]) > 0 else np.nan
            v = [l[0], ngs, ngc, mbl, mbf]
            lab.append(v)
    return pd.DataFrame(lab, columns=['name', "NG_Score", "NG_Class", "MBFC_Label", "MBFC_Factual_Reporting"])


def findUpper(text):
    return re.findall(r'[\b\[A-Z]{4,}\]\b', text)

def findCalls(text):
    return re.findall(r'([\[][^\]]*[\]])', text)


import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.tokens import Doc

#nlp = spacy.load('en_core_web_trf', disable=['tagger', 'parser', 'ner'])
# nlp = nlp.add_pipe('sentencizer')
nlp = spacy.load('en_core_web_trf')
n2 = spacy.load('en_core_web_trf')
n2 = n2.add_pipe('spacytextblob')

sent_analyzer = SentimentIntensityAnalyzer()

def sentiment_scores(docx):
    return sent_analyzer.polarity_scores(docx.text)


Doc.set_extension("sentimenter", getter=sentiment_scores)


def getTextScores(df):
    cols = ['index', 'cleanText', 'subjectivity', 'polarity', 'sentiment', 'length', 'entities', 'pos_count']
    docs = nlp.pipe(df, batch_size=2000, n_process=5)
    vals = []
    i = 0

    for ix, d in enumerate(docs):
        if i % 100 == 0:
            print('text row', i)

        # lemmatize
        txt = " ".join(token.lemma_ for token in d if not token.is_stop)
        txt = txt.replace("\r\n", "")
        people = [e for e in d.ents if e.label_ == "PERSON"]
        POS_counts = d.count_by(spacy.attrs.POS)
        pos = []
        for k, v in sorted(POS_counts.items()):
            pos.append([k, d.vocab[k].text, v])
        v = [ix, txt, d._.blob.subjectivity, d._.blob.polarity, d._.sentimenter['compound'], len(d), people, pos]
        vals.append(v)
    return pd.DataFrame(vals, columns=cols)


def createFiles():
    fp = open('./data/Text_Callouts.csv', 'w')
    fp.write('articleID,callout')
    fp.close()
    fp = open('./data/Text_Cleaned.csv', 'w')
    fp.write('articleID,cleanText')
    fp.close()
    fp = open('./data/Text_Entities.csv', 'w')
    fp.write('articleID,entity')
    fp.close()
    fp = open('./data/Text_POS.csv', 'w')
    fp.write('articleID,posID,posCount,posType')
    fp.close()
    fp = open('./data/Text_Scores.csv', 'w')
    fp.write('articleID,subjectivity, polarity, sentiment, len, noEntsSub, noEntsPol, noEntsSen, noEntsLen')
    fp.close()
    fp = open('./data/Title_Callouts.csv', 'w')
    fp.write('articleID,callout')
    fp.close()
    fp = open('./data/Title_Cleaned.csv', 'w')
    fp.write('articleID,cleanTitle')
    fp.close()
    fp = open('./data/Title_Entities.csv', 'w')
    fp.write('articleID,entity')
    fp.close()
    fp = open('./data/Title_POS.csv', 'w')
    fp.write('articleID,posID,posCount,posType')
    fp.close()
    fp = open('./data/Title_Scores.csv', 'w')
    fp.write('articleID,subjectivity, polarity, sentiment, len, noEntsSub, noEntsPol, noEntsSen, noEntsLen')
    fp.close()


def saveToCSV(name, v):
    with open('./data/' + name, 'a', newline='') as file:
        fw = csv.writer(file)
        for x in range(len(v)):
            fw.writerow(v[x])


def getScores(df, ttype):
    import time
    ll = len(df)
    st = time.time()
    tbl_callout = ttype +"_Callouts"
    tbl_entities = ttype + "_Entities"
    tbl_scores = ttype + "_Scores"
    tbl_pos = ttype + "_POS"
    tbl_cln = ttype + "_Cleaned"
    # cols = ['articleID', 'subjectivity', 'polarity', 'sentiment', 'length', 'noEntsSubj', 'noEntspolarity', 'noEntsSentiment', 'noEntsLength']

    docs = nlp.pipe(df, batch_size=30)
    # conn = sqlite3.connect(".\data\misinfo.db")
    # with conn:
    # titleScores, titleEnts, titleCallouts, titlePOS
    nx = time.time()
    for ix, d in enumerate(docs):
        if int(ix) % 1000 == 0:
            print(ttype + ' row' + str(int(ix)) + ' of ' + str(ll), 'total time ', ((time.time()-st)/60)/60, 'last time', ((time.time()-nx)/60)/60)
            nx = time.time()

        # Get Callouts
        callouts = findCalls(d.text)
        if len(callouts) > 0:

            tca = [(int(ix), str(c)) for c in callouts]
            saveToCSV(tbl_callout+".csv", [[int(ix), str(c)] for c in callouts])
            # # sql = 'INSERT INTO ' + "TC" + '(articleID,callout) VALUES(?,?) '
            # sql = 'INSERT INTO ' + tbl_callout + '(articleID,callout) VALUES(?,?) '
            # c = conn.cursor()
            # c.executemany(sql, tca)

        # Named Entities
        people = [e for e in d.ents if e.label_ == "PERSON"]
        if len(people) > 0:
            tea = [(int(ix), str(p)) for p in people]
            saveToCSV(tbl_entities + ".csv", [[int(ix), str(p)] for p in people])
            # sql = 'INSERT INTO ' + tbl_entities + '(articleID,entity) VALUES(?,?)'
            # c = conn.cursor()
            # c.executemany(sql, tea)

        # Get Parts-of-Speech
        POS_counts = d.count_by(spacy.attrs.POS)
        # tpa = [(int(ix), int(k), int(v), str(d.vocab[k].text)) for k, v in sorted(POS_counts.items())]
        saveToCSV(tbl_pos + ".csv", [[int(ix), int(k), int(v), str(d.vocab[k].text)] for k, v in sorted(POS_counts.items())])
        # sql = 'INSERT INTO ' + tbl_pos + ' (articleID, posID, posCount, posType) VALUES(?,?,?,?)'
        # c = conn.cursor()
        # c.executemany(sql, tpa)

        # Score Cleaned Text
        cleaned = str(d)
        for p in people:
            cleaned = cleaned.replace(str(p), "")
        for c in callouts:
            cleaned = cleaned.replace(str(c), "")
        # sql = 'INSERT INTO ' + tbl_cln + ' (articleID,cleanText) VALUES(?,?)'
        # c = conn.cursor()
        # c.execute(sql, (int(ix), str(cleaned)))
        saveToCSV(tbl_cln + ".csv", [[int(ix), str(cleaned)]])
        # txt = str(d)
        # nn = n2(txt)
        b = SpacyTextBlob(d)
        pol = float(b.get_polarity(d))
        subj = float(b.get_subjectivity(d))
        sent = float(d._.sentimenter['compound'])
        tt = nlp(cleaned)
        bt = SpacyTextBlob(tt)
        cpol = float(bt.get_polarity(tt))
        csubj = float(bt.get_subjectivity(tt))
        csent = float(tt._.sentimenter['compound'])

        # v = (int(ix), subj, pol, sent, len(nn), csubj, cpol, csent, len(tt))
        # v = (int(ix), float(dd._.blob.subjectivity), float(dd._.blob.polarity), float(dd._.sentimenter['compound']), len(dd), float(t._.blob.subjectivity), float(t._.blob.polarity), float(t._.sentimenter['compound']), len(t))
        saveToCSV(tbl_scores + ".csv", [[int(ix), subj, pol, sent, len(d), csubj, cpol, csent, len(tt)]])
        # sql = 'INSERT INTO ' + tbl_scores + '(articleID,subjectivity, polarity, sentiment, len, noEntsSub, noEntsPol, noEntsSen, noEntsLen) VALUES(?,?,?,?,?,?,?,?,?)'
        #
        # c = conn.cursor()
        # c.execute(sql, v)

    # conn.commit()
    # conn.close()
    return




# if __name__ == "__main__":
sqlFile = ".\data\misinfo.db"
labelFile = ".\data\labels.csv"
# pipeline = ['sentimenter']
fullLabels = getLabels(labelFile)

# Clean Data
# remove rows with no labels
idx = np.where(np.all(fullLabels.iloc[:, 1:].isnull(), axis=1))[0]
labels = fullLabels[~fullLabels.index.isin(idx)]
data = pd.read_csv("./data_pre1.csv", header=0, index_col=0)
txt = data.loc[:, 'content'].to_list()
title = data.loc[:, 'name'].to_list()


getScores(title[213484:], "Title")
getScores(txt, "Text")
# r = SpacyTextBlob(nlp(title[0]))

    #
    # fullData = sqlQuery(sqlFile, labels.name.to_list())
    # noTitles = fullData[fullData.name.str.len() == 0]
    # noContent = fullData[fullData.content.str.len() == 0]
    # missingData = pd.concat([noTitles, noContent]).drop_duplicates()
    # data = fullData[~fullData.index.isin(missingData.index.to_list())].reset_index(drop=True)
    # data.to_csv("data_pre1.csv")

    # titleScores, titleEnts, titleCallouts, titlePOS = getTitleScores(title)
    # titleScores, titleEnts, titleCallouts, titlePOS = getTitleScores(title[:20])

    # saveTable(sqlFile, dfTitleScores, "NELA_titleScores")
    # dfTitleScores.to_csv('dfTitleScores.csv')
    # dfTextScores = getTextScores(txt)
    # saveTable(sqlFile, dfTextScores, "NELA_textScores")
    # dfTextScores.to_csv('dfTextScores.csv')
    # dfTitleScores = getTitleScores(title)
    # saveTable(sqlFile, dfTitleScores, "NELA_titleScores")

    # MBFC Factual Reporting Labels:
    # VERY HIGH = a score of 0, which means the source is always factual
    # HIGH = a score of 1 – 2, which means the source is almost always factual
    # MOSTLY FACTUAL = a score of 3 – 4, which means the source is usually factual but may have failed a fact check or two that was not properly corrected promptly
    # MIXED = a score of 5 – 6, which means the source does not always use proper sourcing or sources to other biased/mixed factual sources
    # LOW = a score of 7 – 9, which means the source rarely uses credible sources and is not trustworthy for reliable information
    # VERY LOW = a score of 10, which means the source rarely uses credible sources and is not trustworthy for reliable information at all

# Spacy POS Tagging
# POS	DESCRIPTION	EXAMPLES
# ADJ	adjective	*big, old, green, incomprehensible, first*
# ADP	adposition	*in, to, during*
# ADV	adverb	*very, tomorrow, down, where, there*
# AUX	auxiliary	*is, has (done), will (do), should (do)*
# CONJ	conjunction	*and, or, but*
# CCONJ	coordinating conjunction	*and, or, but*
# DET	determiner	*a, an, the*
# INTJ	interjection	*psst, ouch, bravo, hello*
# NOUN	noun	*girl, cat, tree, air, beauty*
# NUM	numeral	*1, 2017, one, seventy-seven, IV, MMXIV*
# PART	particle	*’s, not,*
# PRON	pronoun	*I, you, he, she, myself, themselves, somebody*
# PROPN	proper noun	*Mary, John, London, NATO, HBO*
# PUNCT	punctuation	*., (, ), ?*
# SCONJ	subordinating conjunction	*if, while, that*
# SYM	symbol	*$, %, §, ©, +, −, ×, ÷, =, :), 😝*
# VERB	verb	*run, runs, running, eat, ate, eating*
# X	other	*sfpksdpsxmsa*
# SPACE	space



# spacy.tokens.token.Token.set_extension("anonymized", default = "")
#
# def cleanCols(df):
#     d = df.copy()
#     d['text'] = d.content.apply(lambda text: " ".join(token.lemma_ for token in nlp(text) if not token.is_stop))
#     d['text'] = d['text'].str.replace("\r\n", "")
#     d['title'] = d.name.apply(lambda text: " ".join(token.lemma_ for token in nlp(text) if not token.is_stop))
#     d['title'] = d['title'].str.replace("\r\n", "")
#     return d
# tPOS_counts = t.count_by(spacy.attrs.POS)
# tpos = []
# for k, v in sorted(tPOS_counts.items()):
#     tpos.append([k, t.vocab[k].text, v])

# v = [ix, d._.blob.subjectivity, d._.blob.polarity, d._.sentimenter['compound'], len(d), str(people), str(pos), str(callouts), t._.blob.subjectivity, t._.blob.polarity, t._.sentimenter['compound'], len(t), tpos]

# pos.append([k, d.vocab[k].text, v])


# for k, v in sorted(POS_counts.items()):
#
#     print(tea)
#     sql = ''' INSERT INTO TE(articleID,entity) VALUES(?,?) '''
#     c = conn.cursor()
#     c.executemany(sql, tea)
#     tp.append([1, ix, k, v, d.vocab[k].text])

# Remove entities, callouts & reevaluate


    # cols = ['index', 'subjectivity', 'polarity', 'sentiment', 'length', 'entities', 'pos_count', 'callouts', 'subjNoEntsCalls', 'polNoEntsCalls', 'sentNoEntsCalls', 'lenNoEntsCalls', 'posNoEntsCalls']
#
# import spacy
# # use python -m spacy download en_core_web_trf
#
# # sp = spacy.load('en_core_web_trf')
# import string
# from spacy.lang.en import English
# from spacy.lang.en.stop_words import STOP_WORDS
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.metrics import accuracy_score
# from sklearn.base import TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.svm import LinearSVC
#
# # nltk.download('vader_lexicon')
# # nltk.download("stopwords")
# # nltk.download("wordnet")
# # nltk.download("omw-1.4")

# import nltk
# import sklearn
# import vaderSentiment
# from sklearn.utils import shuffle
# import spacy_transformers


# warnings.simplefilter(action='ignore', category=Warning)
# pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 2000)

# Adapted from https://github.com/MELALab/nela-gt-2019


# from datetime import datetime
#
# # import spacy_transformers
# from spacy.tokens import DocBin
# from spacytextblob.spacytextblob import SpacyTextBlob
#
# # from spacy import displacy
#
# # python -m textblob.download_corpora

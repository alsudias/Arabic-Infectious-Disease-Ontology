from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
def featureExtraction1(data):
    vectorizer = TfidfVectorizer(min_df=7, max_df=0.75, ngram_range=(1,3))
    tfidf_data = vectorizer.fit_transform(data)
    return tfidf_data

def featureExtraction2(data):
    vectorizer =  CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 3),
    min_df = 2
)
    count_data = vectorizer.fit_transform(data)
    
    return count_data


document_text = open('Zika5clean.txt', 'r',encoding='utf-8')
book = document_text.read()


sentences = book.split('.')
#print(sentences)
count_data = featureExtraction2(sentences)
#print((count_data))
print("******************************************")
count_data1 = featureExtraction1(sentences)
#print((count_data1))

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

#load a set of stop words
stopwords=get_stop_words("arabic-stop-words-list.txt")
tfidf_vectorizer=CountVectorizer(analyzer = 'word',lowercase = False,max_df=0.85,stop_words=stopwords,max_features=10000)
#tokenizer=tokenize_and_stem,stop_words='arabic',
tfidf = tfidf_vectorizer.fit_transform(sentences)
print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

print("Now, let’s look at 20 words from our vocabulary:\n")
print(list(tfidf_vectorizer.vocabulary_.keys())[:20])#print the keywords from the document
print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

kmeans = KMeans(n_clusters=5).fit(tfidf)

terms = tfidf_vectorizer.get_feature_names()
#print(terms)
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(sentences)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
df1 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
#######print(df1.sort_values(by=["tfidf"],ascending=False))
#lines_for_predicting = ["علاج الضنك", "الحرارة من أعراض الضنك"]
#kmeans.predict(tfidf_vectorizer.transform(lines_for_predicting))

#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(document_text)
#print(vectorizer.get_feature_names())

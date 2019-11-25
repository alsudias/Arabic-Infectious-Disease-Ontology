import yake

document_text = open("Info5clean.txt", 'r',encoding='utf-8')
book = document_text.read()

print('\n########################################################\n')
language = "ar"
max_ngram_size = 2
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(book)

for kw in keywords:
    print(kw[0])

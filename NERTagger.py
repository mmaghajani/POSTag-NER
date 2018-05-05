# -*- coding: utf-8 -*-

from nltk.tag import StanfordNERTagger

st = StanfordNERTagger(
    '/home/mma137421/stanford-ner/ner-model.ser.gz',
    '/home/mma137421/stanford-ner/stanford-ner-3.9.1.jar',
    encoding='utf-8')

test = dict()
with open("data/NERte.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        test[words[0].strip()] = words[1].strip()
    file.close()

test_words = list(map(lambda x: x[0], test.items()))
test_label = list(map(lambda x: x[1], test.items()))

cnt = 0
classified_text = st.tag(test_words)

for i in range(0,len(classified_text)):
    if classified_text[i][1] == test_label[i]:
        cnt += 1

print("Accuracy : ", cnt/len(classified_text))
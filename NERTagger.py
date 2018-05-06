# -*- coding: utf-8 -*-

import nltk.tag.stanford as st
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    else:
        #         print('Confusion matrix, without normalization')
        tmp = 2

    #     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

tagger = st.StanfordNERTagger(
    '/home/mma137421/stanford-ner/ner-model.ser.gz',
    '/home/mma137421/stanford-ner/stanford-ner.jar')

test = list()
classes = set()
with open("data/NERte.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        if words[0].strip() != '':
            test.append((words[0].strip(), words[1].strip()))
            classes.add(words[1].strip())
    file.close()

test_words = list(map(lambda x: x[0], test))
test_label = list(map(lambda x: x[1], test))

cnt = 0

classified_text = tagger.tag(test_words)
predict_labels = list(map(lambda x: x[1], classified_text))

for i in range(0, len(classified_text)):
    if classified_text[i][1] == test_label[i]:
        cnt += 1

print("Accuracy : ", cnt/len(classified_text))

confusion_matrix_all = confusion_matrix(test_label, predict_labels, binary=False)

plt.figure()
plot_confusion_matrix(confusion_matrix_all, classes)
plt.show()

plt.figure()
plot_confusion_matrix(confusion_matrix_all, classes, normalize=True)
plt.show()
print(confusion_matrix_all)
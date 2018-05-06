from nltk.tag import hmm
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


train = list()
sequence = list()
cnt = 0
classes = set()
with open("data/POStrutf.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        classes.add(words[2].strip())
        sequence.append((words[0].strip(), words[2].strip()))
        if words[0].strip() == ".":
            train.append(sequence)
            cnt = 0
            sequence = list()
        cnt += 1
    file.close()

test = list()
with open("data/POSteutf.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        test.append((words[0].strip(), words[2].strip()))
    file.close()

test_words = list(map(lambda x: x[0], test))
test_label = list(map(lambda x: x[1], test))

tagger = hmm.HiddenMarkovModelTagger.train(train)

cnt = 0
predict = tagger.tag(test_words)
predict_labels = list(map(lambda x: x[1], predict))

for i in range(0, len(predict)):
    if predict[i][1] == test_label[i]:
        cnt += 1
print("Accuracy : ", cnt / len(predict))

confusion_matrix_all = confusion_matrix(test_label, predict_labels, binary=False)

plt.figure()
plot_confusion_matrix(confusion_matrix_all, classes)
plt.show()

plt.figure()
plot_confusion_matrix(confusion_matrix_all, classes, normalize=True)
plt.show()
print(confusion_matrix_all)

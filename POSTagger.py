from nltk.tag import hmm
import pprint
train = list()
sequence = list()
cnt = 0
with open("data/POStrutf.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        sequence.append((words[0].strip(), words[2].strip()))
        if words[0].strip() == ".":
            train.append(sequence)
            cnt = 0
            sequence = list()
        cnt += 1
    file.close()

test = dict()
with open("data/POSteutf.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        test[words[0].strip()] = words[2].strip()
    file.close()

test_words = list(map(lambda x: x[0], test.items()))
test_label = list(map(lambda x: x[1], test.items()))

tagger = hmm.HiddenMarkovModelTagger.train(train)

cnt = 0
predict = tagger.tag(test_words)
predict_words = list(map(lambda x:x[0], predict))

for i in range(0,len(predict)):
    if predict[i][1] == test[predict[i][0]]:
        cnt += 1
print("Accuracy : ", cnt/len(predict))

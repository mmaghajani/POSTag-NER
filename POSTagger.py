from nltk.tag import hmm

train = list()
sequence = list()
cnt = 0
with open("data/POStrutf.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        sequence.append((words[0].strip(), words[2].strip()))
        if cnt is 20:
            train.append(sequence)
            cnt = 0
            sequence = list()
        cnt += 1
    train.append(sequence)
    cnt = 0
    sequence = list()
    file.close()

test = dict()
with open("data/POSteutf.txt", 'r') as file:
    for line in file.readlines():
        words = line.split("\t")
        test[words[0].strip()] = words[2].strip()
    file.close()

test_words = list(map(lambda x: x[0], test.items()))
test_label = list(map(lambda x: x[1], test.items()))

print("train data", test_words)


# Setup a trainer with default(None) values
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train)

cnt = 0
predict = tagger.tag(test_words)
predict_words = list(map(lambda x:x[0], predict))
print(predict_words)
for i in range(0,len(predict_words)):
    if predict_words[i] not in test_words:
        print(predict_words[i])

for i in range(0,len(predict)):
    print(predict[i][0], predict[i][1], test[predict[i][0]])
    if predict[i][1] == test[predict[i][0]]:
        cnt += 1
print(cnt/len(predict))

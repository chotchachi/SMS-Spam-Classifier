from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import Utils
import json
import coremltools
import pandas as pd


def multinomialNB():
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    aucscore = roc_auc_score(y_test, predictions)
    print(aucscore)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(pd.DataFrame(confusion_matrix(y_test, predictions),
                       columns=['Predicted Spam', "Predicted Ham"], index=['Actual Spam', 'Actual Ham']))
    print(f'\nTrue Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')

    print(f'True Positive Rate: {(tp / (tp + fn))}')
    print(f'Specificity: {(tn / (tn + fp))}')
    print(f'False Positive Rate: {(fp / (fp + tn))}')


def bernoulliNB():
    model = BernoulliNB(alpha=1e-10)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    aucscore = roc_auc_score(y_test, predictions)
    print(aucscore)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(pd.DataFrame(confusion_matrix(y_test, predictions),
                       columns=['Predicted Spam', "Predicted Ham"], index=['Actual Spam', 'Actual Ham']))
    print(f'\nTrue Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')

    print(f'True Positive Rate: {(tp / (tp + fn))}')
    print(f'Specificity: {(tn / (tn + fp))}')
    print(f'False Positive Rate: {(fp / (fp + tn))}')


def svm():
    model = LinearSVC()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    aucscore = roc_auc_score(y_test, predictions)
    print(aucscore)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(pd.DataFrame(confusion_matrix(y_test, predictions),
                       columns=['Predicted Spam', "Predicted Ham"], index=['Actual Spam', 'Actual Ham']))
    print(f'\nTrue Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')

    print(f'True Positive Rate: {(tp / (tp + fn))}')
    print(f'Specificity: {(tn / (tn + fp))}')
    print(f'False Positive Rate: {(fp / (fp + tn))}')


if __name__ == '__main__':
    print(">>>>>>>>>>>>>>>>>>>> START >>>>>>>>>>>>>>>>>>>>")

    # pretreatment
    labels, content = Utils.loadCorpus('corpus.txt')
    corpus = [Utils.removeSpecialCharacter(d) for d in content]
    labels = Utils.mapLabelToNumber(labels)
    bag_word = Utils.exportBagWords(corpus)
    vectors = Utils.convertToVector(corpus, bag_word)

    # save bag words
    # with open("bagword.json", "w") as outfile:
    #     json.dump(bag_word, outfile)

    X_train, X_test, Y_train, Y_test = train_test_split(vectors, labels, random_state=0)

    # print("\nStart train using MultinomialNB")
    # multinomialNB()
    # print("\nStart train using BernoulliNB")
    # bernoulliNB()
    # print("\nStart train using SVM")
    # svm()

    # train model using naive bayes
    # clf1 = BernoulliNB(alpha=1e-10)
    # clf1.fit(vectors, labels)

    # train model using svm
    svm = LinearSVC()
    svm.fit(vectors, labels)

    # create coreML model
    svmModel = coremltools.converters.sklearn.convert(svm, 'message', 'label')
    svmModel.author = 'Thanh Quang'
    svmModel.short_description = 'Classify whether message is spam or not'
    svmModel.input_description['message'] = 'vector spam 0 - 1'
    svmModel.save('detect_spam_svm.mlmodel')

    # coreml_model = coremltools.converters.sklearn.convert(clf1)
    # coreml_model.author = 'Thanh Quang'
    # coreml_model.short_description = "Classify whether message is spam or not"
    # coreml_model.input_description["message"] = "vector of message to be classified"
    # coreml_model.output_description["spam_or_not"] = "Whether message is spam or not"
    # coreml_model.save("detect_spam.mlmodel")

    # test
    test_texts = [
        'Cho vay tieu dùng ca nhan,Lai xuat hap hap dan,lien he 099.666.8888',
        'Rảnh không mi, đi học bơi với t nè, chừ luôn',
        'Em đi học về chưa',
        'Gọi ngay, còn duy nhất 20 căn giá rẻ. ĐT xxxxxxx',
        'Ban duoc FE CREDIT ho tro vay den 50 TRIEU VND, tra gop tu 439,000d/thang…',
        'Nếu mà cô quyết định như này thì tối qua anh không để em một mình đâu',
        'Bạn đang làm gì đấy, inbox mình nhờ chút chuyện với nhé',
        'Chương trình khuyến mãi lớn nhất trong năm, giảm giá 50% tất cả các mặt hàng, free quẹt thẻ, 100% mua là trúng'
    ]

    y_test = [1, 0, 0, 1, 1, 0, 0, 1]
    x_test = [Utils.handleMessage(text, bag_word) for text in test_texts]

    # bernoulliPredictions = clf1.predict(x_test)
    svmPredictions = svm.predict(x_test)

    # accuracy
    # print("Bernoulli ex", bernoulliPredictions)
    # print('Bernoulli: Training size = %d, accuracy = %.2f%%' % \
    #       (len(vectors), accuracy_score(y_test, bernoulliPredictions) * 100))

    print("Svm ex", svmPredictions)
    print('Svm: Training size = %d, accuracy = %.2f%%' % \
          (len(vectors), accuracy_score(y_test, svmPredictions) * 100))

    # # confusion matrix
    # tn, fp, fn, tp = confusion_matrix(y_test, bernoulliPredictions).ravel()
    # print(pd.DataFrame(confusion_matrix(y_test, bernoulliPredictions),
    #                    columns=['Predicted Spam', "Predicted Ham"], index=['Actual Spam', 'Actual Ham']))
    # print(f'\nTrue Positives: {tp}')
    # print(f'False Positives: {fp}')
    # print(f'True Negatives: {tn}')
    # print(f'False Negatives: {fn}')
    #
    # print(f'True Positive Rate: {(tp / (tp + fn))}')
    # print(f'Specificity: {(tn / (tn + fp))}')
    # print(f'False Positive Rate: {(fp / (fp + tn))}')
    #
    # count_spam, count_non_spam = Utils.getCountSpamOrNotSpam(labels)
    # _p_spam = Utils.smoothing(count_spam, (count_non_spam + count_spam))
    # _p_non_spam = Utils.smoothing(count_non_spam, (count_non_spam + count_spam))
    #
    # bayes_matrix = Utils.np.zeros((len(bag_word), 4))  # create matrix
    # Utils.configureBayesMatrix(bag_word, vectors, labels, count_spam, count_non_spam, bayes_matrix)
    # spam = _p_spam
    # not_spam = _p_non_spam
    #
    # smvv = [svm.predict([Utils.handleMessage(d, bag_word)])[0] for d in corpus]
    # preds = [clf1.predict([Utils.handleMessage(d, bag_word)])[0] for d in corpus]
    # pred = [Utils.predict(d, bag_word, spam, not_spam, bayes_matrix) for d in corpus]
    # print(accuracy_score(labels, smvv) * 100)
    # print(accuracy_score(labels, pred) * 100)
    # print('%', accuracy_score(labels, preds) * 100)

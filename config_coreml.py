import coremltools

if __name__ == '__main__':
    model = coremltools.models.MLModel('detect_spam_ok.mlmodel')

    model.author = 'Thanh Quang'
    model.short_description = 'Classify whether message is spam or not'
    model.input_description['message'] = 'vector spam 0 - 1'
    model.save('detect_spam_new.mlmodel')

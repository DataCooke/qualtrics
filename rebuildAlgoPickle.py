import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from google.cloud import storage

training = pd.read_csv('C:/Users/jcooke/PycharmProjects/qualtrics/newLabeledDat.csv')

training = training.loc[:, ~training.columns.str.contains('^Unnamed')]
print(training)


train = training.sample(frac=.8, random_state=164)
test = training.drop(train.index)

train = pd.DataFrame(train)
train = train.reset_index(drop=True)
stop_words = set(stopwords.words('english'))
train['response'] = train['response'].str.replace("[^\w\s]", "").str.lower()
train['response'] = train['response'].apply(
    lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

train['sentenceLemmatized'] = train.response.apply(lemmatize_text)
train = pd.DataFrame(train)
train['sentenceLemmatized'] = [','.join(map(str, l)) for l in train['sentenceLemmatized']]
train['sentenceLemmatized'] = train['sentenceLemmatized'].str.replace(',', ' ')
train = train[['sentenceLemmatized', 'sentiment']].apply(tuple, axis=1)

## process test

test = pd.DataFrame(test)

test = test.reset_index(drop=True)
stop_words = set(stopwords.words('english'))
test['response'] = test['response'].str.replace("[^\w\s]", "").str.lower()
test['response'] = test['response'].apply(
    lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
test['sentenceLemmatized'] = test.response.apply(lemmatize_text)
test = pd.DataFrame(test)
test['sentenceLemmatized'] = [','.join(map(str, l)) for l in test['sentenceLemmatized']]
test['sentenceLemmatized'] = test['sentenceLemmatized'].str.replace(',', ' ')
test = test[['sentenceLemmatized', 'sentiment']].apply(tuple, axis=1)
testNonPairs = [(a) for a, b in test]
print("this is test")
print(test)

dictionaryTest = set(word.lower() for passage in test for word in word_tokenize(passage[0]))
print("this is dictionaryTest")
print(dictionaryTest)

dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
dictExport = pd.DataFrame(dictionary)
dictExport.to_csv("dictExport.csv")
print("dictionary here")
print(dictionary)
print(type(dictionary))
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]

tTest = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in test]
print('tTest is below')
print(tTest[:1])
classifier = nltk.NaiveBayesClassifier.train(t)

# added trying to check classifier accuracy
print("Naive Bayes Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, tTest)) * 100)

saved_model = pickle.dumps(classifier)
pickle.dump(classifier, open("naiveBayesModel.p", "wb"))


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    #client = storage.Client()
    #bucket = client.get_bucket("test_feedback_nlp")
    #blob = bucket.blob(f"dictionary/naiveBayesModel.p")
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

bucket_name = "test_feedback_nlp"
source_file_name = "C:/Users/jcooke/PycharmProjects/qualtrics/naiveBayesModel.p"
destination_blob_name = "dictionary/naiveBayesModel.p"

upload_blob(bucket_name, source_file_name, destination_blob_name)

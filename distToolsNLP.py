import http.client
import zipfile
import os
import io
import base64
import json
import requests
from pandas.io.json import json_normalize
import pandas as pd
import chardet

import http.client
import zipfile
import os
import io
import base64
import json
import requests
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import chardet
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sent_tokenize
from nltk import word_tokenize
from google.cloud import bigquery
import pickle


clientID = os.getenv('clientID')
clientsecret = os.getenv('clientsecret')

#create the Base64 encoded basic authorization string
clientID=f"{clientID}"
clientsecret=f"{clientsecret}"
auth = "{0}:{1}".format(clientID, clientsecret)
encodedBytes=base64.b64encode(auth.encode("utf-8"))
authStr = str(encodedBytes, "utf-8")

#create the connection
conn = http.client.HTTPSConnection("iad1.qualtrics.com")
body = "grant_type=client_credentials"
headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
}
headers['Authorization'] = 'Basic {0}'.format(authStr)

#make the request
conn.request("POST", "/oauth2/token", body, headers)
res = conn.getresponse()
data = res.read()
token = json.loads(data.decode("utf-8"))
#print(token)
token = token["access_token"]
#os.environ["token"] = token


# create the request
conn = http.client.HTTPSConnection("co1.qualtrics.com")
body = ''
headers = {
  'Authorization': f'Bearer {token}'
}

# make the request
conn.request("GET", "/API/v3/whoami", body, headers)
res = conn.getresponse()
data = res.read()
#print(data.decode("utf-8"))




# Setting user Parameters
dataCenter = "co1"

bearerToken = token # call the function defined in the previous code example
baseUrl = "https://{0}.qualtrics.com/API/v3/surveys".format(dataCenter)
headers = {
    "authorization": "bearer " + bearerToken,
    }

response = requests.request("GET", baseUrl, headers=headers)
#print(response.text)

#end of test api call
#added statics stuff
requestCheckProgress = 0.0
progressStatus = "inProgress"
#end added stuff

bearerToken = token # call the function defined in the previous code example
baseUrl = "https://co1.qualtrics.com/API/v3/surveys/SV_8wbVPpmD5l5ItCt/export-responses/"
headers = {
    "authorization": "bearer " + bearerToken,
    "content-type": "application/json"
    }

body = {
  "format": "csv",
  "filterId": "9ea61539-3cf9-4fa0-86e5-9e01ee19fc36",
  "startDate": "2020-11-11T00:00:00-07:00",
  "endDate": "2020-11-24T00:00:00-07:00"
}

downloadRequestResponse = requests.request("POST", baseUrl, headers=headers, json=body)
progressId = downloadRequestResponse.json()["result"]["progressId"]
print(downloadRequestResponse.text)




# Step 2: Checking on Data Export Progress and waiting until export is ready
while progressStatus != "complete" and progressStatus != "failed":
    print("progressStatus=", progressStatus)
    requestCheckUrl = baseUrl + progressId
    requestCheckResponse = requests.request("GET", requestCheckUrl, headers=headers)
    print(requestCheckResponse.text)
    requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
    print("Download is " + str(requestCheckProgress) + " complete")
    progressStatus = requestCheckResponse.json()["result"]["status"]

    if progressStatus == "failed":
        raise Exception("export failed")

    if progressStatus == "complete":
        fileId = requestCheckResponse.json()["result"]["fileId"]

    # Step 3: Downloading file
        requestDownloadUrl = baseUrl + fileId + '/file'
        requestDownload = requests.request("GET", requestDownloadUrl, headers=headers, stream=True)
        data = requestDownload.content


    # Step 4: Unzipping the file
        #use below to extract data as csv in folder MyQualtricsDownload
        #data1 = zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall("MyQualtricsDownload")
        request = zipfile.ZipFile(io.BytesIO(requestDownload.content))
        use_cols = ["StartDate", "ResponseId", "UserLanguage", "Q1_Browser", "Q1_Operating System", "CurrURL", "Where did we wander off track? Where did we have problems? What\ndid you lov... EN", "Shareyour thoughts. How's our website design? Would you like to see any ad... EN", "Please\n  describe the main reason for your visit: EN"]
        dataDf = pd.read_csv(request.open(request.namelist()[0]), usecols=use_cols)
        #print("dataDf pandas dataframe below")
        #print(dataDf)
        #dataDf.to_csv('data.csv')
        print('*** DATA PULL COMPLETE ***')

print(" ")
###prepare file for future output
''' output will include duplicate values for ALL columns where a user answered multiple questions. 
Each row represents one response where a single user could answer multiple questions and thus multiple responses.
Therefore, when counting the number of Chrome browsers, you need to sum unique responses where browser is chrome'''
dataDf.columns = dataDf.iloc[1]
dataDf = dataDf.drop([0,1])
dataDf = dataDf.reset_index(drop=True)

dataDf = dataDf.rename(columns={
    '{\"ImportId\":\"startDate\",\"timeZone\":\"Z\"}':'startDate'
    , '{\"ImportId\":\"_recordId\"}':'responseID'
    , '{\"ImportId\":\"userLanguage\"}':'userLanguage'
    , '{\"ImportId\":\"QID7_BROWSER\"}':'browser'
    , '{\"ImportId\":\"QID7_OS\"}':'os'
    , '{\"ImportId\":\"CurrURL\"}':'currUrl'
    , '{\"ImportId\":\"QID29_TEXT_TRANSLATEDeneqqpuqm\"}':'q29'
    , '{\"ImportId\":\"QID3_TEXT_TRANSLATEDenayfj8yo\"}':'q3'
    , '{\"ImportId\":\"QID83_TEXT_TRANSLATEDeneh72e33\"}':'q83'
})

DfQ29 = dataDf.drop(columns=['q3', 'q83'])
DfQ3 = dataDf.drop(columns=['q29', 'q83'])
DfQ83 = dataDf.drop(columns=['q29', 'q3'])

DfQ29 = DfQ29.rename(columns={'q29':'response'})
DfQ3 = DfQ3.rename(columns={'q3':'response'})
DfQ83 = DfQ83.rename(columns={'q83':'response'})

DfQ29 = DfQ29.dropna(axis=0, subset=['response'])
DfQ3 = DfQ3.dropna(axis=0, subset=['response'])
DfQ83 = DfQ83.dropna(axis=0, subset=['response'])

DfQ29["question"] = "q29"
DfQ3["question"] = "q3"
DfQ83["question"] = "q83"

DfQ29.to_csv("DfQ29.csv")
DfQ3.to_csv("DfQ3.csv")
DfQ83.to_csv("DfQ83.csv")

all_dfs = [DfQ29, DfQ3, DfQ83]
for df in all_dfs:
    df.columns = ["startDate", "responseID", "userLanguage", "browser", "os", "currUrl", "response", "question"]
all_dfs = pd.concat(all_dfs).reset_index(drop=True)

classifyData = all_dfs['response']
classifyData = pd.DataFrame(classifyData)


stop_words = set(stopwords.words('english'))
classifyData['response'] = classifyData['response'].str.replace("[^\w\s]", "").str.lower()
classifyData['response'] = classifyData['response'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text) ]

classifyData['sentenceLemmatized'] = classifyData.response.apply(lemmatize_text)
classifyData = pd.DataFrame(classifyData)
classifyData['sentenceLemmatized'] = [','.join(map(str, l)) for l in classifyData['sentenceLemmatized']]
classifyData['sentenceLemmatized'] = classifyData['sentenceLemmatized'].str.replace(',', ' ')
classifyData = classifyData[['sentenceLemmatized']].apply(tuple, axis=1)

#pull dictionary from google cloud storage

from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):

    """Downloads a blob from the bucket."""
    bucket_name = "test_feedback_nlp"
    source_blob_name = "dictExport.csv"
    destination_file_name = "https://console.cloud.google.com/storage/browser/test_feedback_nlp"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


classifyFeatures = [({word: (word in word_tokenize(x[0])) for word in dictionary}) for x in classifyData]

print(type(classifyFeatures))
print(classifyFeatures)
print(type(all_dfs))
print(all_dfs)





all_dfs.to_csv("dataDf.csv")


#dataDf.to_csv("dataDf.csv")
#print("dataframe below")
#print(dataDf)



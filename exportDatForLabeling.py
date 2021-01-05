import http.client
import zipfile
import io
import base64
import json
import requests
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from google.cloud import bigquery
import pickle
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

### Use below if you need to Build startDate and endDate variables beginning on the first day of the previous month and ending on the last day of the previous month
'''
today = date.today()
d = today - relativedelta(months=24)

first_day = date(d.year, d.month, 1)
print("startDate: " + first_day.strftime("%Y-%m-%d"))

startDate = first_day.strftime("%Y-%m-%d")

last_day = date.today() - relativedelta(days=1)
print("endDate: " + last_day.strftime("%Y-%m-%d"))
endDate = last_day.strftime("%Y-%m-%d")'''

### build start date based upon last date from most recent training data set and end date is yesterday.

training = pd.read_csv('C:/Users/jcooke/PycharmProjects/qualtrics/labeledDat.csv')
training['startDate'] = pd.to_datetime(training['startDate'])
startDate = training['startDate'].max().date() + datetime.timedelta(days=1)
startDate = str(startDate)

print(startDate)
### manual entry of start date below
#startDate = "2020-11-11"
last_day = date.today() - relativedelta(days=1)
endDate = last_day.strftime("%Y-%m-%d")
print("startDate: " + startDate)
print("endDate: " + last_day.strftime("%Y-%m-%d"))

### End building startDate and endDate

### download resources needed for use in cloud function. This is only needed when running on console on first run locally

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')

#### defining function for accessing required keys from google secret manager

def access_secret_version(project_id, secret_id, version_id):
    global payload
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    """

    # Import the Secret Manager client library.
    from google.cloud import secretmanager

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"


    # Access the secret version.
    response = client.access_secret_version(request={"name": name})


    # WARNING: Do not print the secret in a production environment - this
    # snippet is showing how to access the secret material.
    payload = response.payload.data.decode("UTF-8")
    access_secret_version.variable = payload
    return(payload)

### calling function to get qualtrics key from secret manager

access_secret_version("nu-skin-corp", "dist-tools-nlp-qualtrics-client-secret", "latest")
clientsecret = payload

### calling function to get qualtrics client id from secret manager

access_secret_version("nu-skin-corp", "dist-tools-nlp-qualtrics-client-id", "latest")
clientID = payload

### use qualtrics client id and client secret to request qualtrics key required to pull data

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
token = token["access_token"]


### final qualtrics key has been pulled. Moving on to pulling data.

### Below only needed if you want to see what user and user id is used for qualtrics data

# create the request
#conn = http.client.HTTPSConnection("co1.qualtrics.com")
#body = ''
#headers = {
#'Authorization': f'Bearer {token}'
#}

# make the request
#conn.request("GET", "/API/v3/whoami", body, headers)
#res = conn.getresponse()
#data = res.read()
#print(data.decode("utf-8"))

### Above only needed if you want to see what user and user id is used for qualtrics data

# Initialize requestCheckProgress and progressStatus
requestCheckProgress = 0.0
progressStatus = "inProgress"

# creating parameters for qualtrics data pull.

bearerToken = token # call the function defined in the previous code example

### "/surverys/********/export-responses/"  - the **** portion of the url is the qualtrics survey ID

baseUrl = "https://co1.qualtrics.com/API/v3/surveys/SV_8wbVPpmD5l5ItCt/export-responses/"
headers = {
    "authorization": "bearer " + bearerToken,
    "content-type": "application/json"
    }

body = {
"format": "csv",
"filterId": "9ea61539-3cf9-4fa0-86e5-9e01ee19fc36",
"startDate": startDate + "T00:00:00-07:00",
"endDate": endDate + "T00:00:00-07:00"
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
        print('*** DATA PULL COMPLETE ***')

 ### data pulled from qualtrics complete

print(" ")
###prepare file for future output
''' output will include duplicate values for ALL columns where a user answered multiple questions. 
Each row represents one response where a single user could answer multiple questions and thus multiple responses.
Therefore, when counting the number of Chrome browsers, you need to sum unique responses where browser is chrome'''

### fix the headers/column heads

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

### create list of responses for each question

DfQ29 = dataDf.drop(columns=['q3', 'q83'])
DfQ3 = dataDf.drop(columns=['q29', 'q83'])
DfQ83 = dataDf.drop(columns=['q29', 'q3'])

DfQ29 = DfQ29.rename(columns={'q29':'response'})
DfQ3 = DfQ3.rename(columns={'q3':'response'})
DfQ83 = DfQ83.rename(columns={'q83':'response'})

DfQ29 = DfQ29.dropna(axis=0, subset=['response'])
DfQ3 = DfQ3.dropna(axis=0, subset=['response'])
DfQ83 = DfQ83.dropna(axis=0, subset=['response'])

### create new column indicating which question each response belongs to

DfQ29["question"] = "q29"
DfQ3["question"] = "q3"
DfQ83["question"] = "q83"

### stack all resonses on top of one another

all_dfs = [DfQ29, DfQ3, DfQ83]
for df in all_dfs:
    df.columns = ["startDate", "responseID", "userLanguage", "browser", "os", "currUrl", "response", "question"]
all_dfs = pd.concat(all_dfs).reset_index(drop=True)

all_dfs = all_dfs.drop(['userLanguage', 'browser', 'os', 'currUrl'], axis=1)
all_dfs["sentiment"] = ""



all_dfs = training.append(all_dfs)
all_dfs = all_dfs.loc[:, ~all_dfs.columns.str.contains('^Unnamed')]
all_dfs = all_dfs.reset_index(drop=True)
all_dfs['response'].replace('', np.nan, inplace=True)
all_dfs.dropna(subset=['response'], inplace=True)
all_dfs.to_csv(r'C:/Users/jcooke/PycharmProjects/qualtrics/partLabeled.csv')

### export new training data ready for labeling

all_dfs.to_csv(r'C:/Users/jcooke/PycharmProjects/qualtrics/partLabeled.csv')

print("New data ready for labeling: partLabeled.csv")
# create training set

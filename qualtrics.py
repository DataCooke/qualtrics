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
print(token)
token = token["access_token"]
os.environ["token"] = token


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
print(data.decode("utf-8"))



# Setting user Parameters
dataCenter = "co1"

bearerToken = token # call the function defined in the previous code example
baseUrl = "https://{0}.qualtrics.com/API/v3/surveys".format(dataCenter)
headers = {
    "authorization": "bearer " + bearerToken,
    }

response = requests.request("GET", baseUrl, headers=headers)
print(response.text)

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
  "startDate": "2018-01-01T00:00:00-07:00",
  "endDate": "2020-09-27T00:00:00-07:00"
}

downloadRequestResponse = requests.request("POST", baseUrl, headers=headers, json=body)
progressId = downloadRequestResponse.json()["result"]["progressId"]
print(downloadRequestResponse.text)

"""print(response.text)
requestID = (response.json())
progressId = requestID["result"]["progressId"]
requestID = requestID["meta"]["requestId"]
print(requestID)"""


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
        #print(dir(requestDownload))
       # print(requestDownload.content)
        #print(pd.json_normalize(data))
       # df = pd.DataFrame(data)
        #data1 = pd.read_csv(data, encoding='gzip')
       # print(data1)
       # data1 =data.decode('utf-16').strip()
       # print(vars(requestDownload))
        #data1 = io.open(data, encoding='utf-16', errors="ignore")
        #print(type(io.BytesIO(data)))
        #rawData = pd.read_csv(io.StringIO(data.decode('utf-8', errors="ignore")))
        #df = pd.read_csv(data.decode("utf-16", errors="replace"), sep="\t")
        #df = pd.read_json(data.decode("utf-16", errors="ignore"))
       # df = pd.read_csv(data, sep="\t", encoding="utf-16", header=[0,1])
        #df = pd.read_csv(requestDownload.content, encoding='unicode_escape')

    # Step 4: Unzipping the file
        data1 = zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall("MyQualtricsDownload")
        use_cols = [
            0, 1, 4, 5, 6
            , 7, 8, 15, 16, 17
            , 18, 19, 20, 24, 25
            , 26, 27, 28, 29, 30
            , 31, 32, 33, 34, 35
            , 37, 38, 39, 40, 41
            , 42, 43, 44, 45, 46
            , 47, 48, 49, 50, 51
            , 52, 53, 54, 55, 56
            , 57, 58, 59, 60, 61
            , 62, 63, 64, 67, 68
            , 69, 70, 71, 72, 73
        ]
        #use_cols = ["StartDate", "EndDate", "UserLanguage"]
        df=pd.read_csv('C:/Users/jcooke/PycharmProjects/qualtrics/MyQualtricsDownload/NuSkin.com Feedback Tab v5.1.csv', usecols=use_cols)
        print(df.head())
        print('Complete')

        #data = requestDownload.content
        #df = pd.read_csv((io.BytesIO(requestDownload.content)))
        #df = pd.read_csv("/Users/jcooke/PycharmProjects/qualtrics/MyQualtricsDownload/NuSkin.com Feedback Tab v5.1.tsv", encoding= 'utf-16')
        #print(df.describe())
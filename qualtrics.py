import http.client
import zipfile
import os
import io
import base64
import json
import requests
import pandas as pd

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
  "startDate": "2020-08-01T00:00:00-07:00",
  "endDate": "2020-09-01T00:00:00-07:00"
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
        df = pd.read_csv(io.BytesIO(data), encoding='unicode_escape')
        #df = pd.read_csv(requestDownload.content, encoding='unicode_escape')

    # Step 4: Unzipping the file
        zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall("MyQualtricsDownload")
        print('Complete')

        #data = requestDownload.content
        #df = pd.read_csv((io.BytesIO(requestDownload.content)))
        #df = pd.read_csv("/Users/jcooke/PycharmProjects/qualtrics/MyQualtricsDownload/NuSkin.com Feedback Tab v5.1.csv", encoding= 'unicode_escape')
        #print(df.describe())
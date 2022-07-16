import json
import requests
import sys
import os

XNAT_HOST = os.environ["XNAT_HOST"]
XNAT_USER = os.environ["XNAT_USER"]
XNAT_PASS = os.environ["XNAT_PASS"]

project = sys.argv[1]
subject = sys.argv[2]
session = sys.argv[3]
custom_variable = sys.argv[4] 


fieldname_key = 'xnat:experimentdata/fields/field/name'
fieldvalue_key = 'xnat:experimentdata/fields/field/field'
session_key = 'xnat:subjectassessordata/id'

# Original command: curl -u user:pw https://xnat-dev-sat1.nrg.wustl.edu/data/projects/glioma_dev/subjects/XNAT_S00987/experiments?columns=xnat:experimentData/fields/field/field,xnat:experimentData/fields/field/name

# # ~~~~~~~~~~~~~~~~~ Request using jsession

# # Followed jsession request from: https://stackoverflow.com/questions/12737740/python-requests-and-persistent-sessions
# # Used this website to convert curl to python request: https://curlconverter.com/#python

# s = requests.Session()
# # print(f"[REQUEST] {XNAT_HOST + '/data/JSESSION'}, credentials = {XNAT_USER}/{XNAT_PASS}")
# s.post(XNAT_HOST + '/data/JSESSION', auth=(XNAT_USER, XNAT_PASS))

# params = (('columns', f"{fieldvalue_key},{fieldname_key}"),)

# # print(f"[REQUEST] {XNAT_HOST}/data/projects/{project}/subjects/{subject}/experiments")
# response = s.get(f"{XNAT_HOST}/data/projects/{project}/subjects/{subject}/experiments", params=params)
# # print("response", response.json())
# resp = response.json()['ResultSet']['Result']


# ~~~~~~~~~~~~~~~~~ Request using os.popen()
req=f"curl -k -u {XNAT_USER}:{XNAT_PASS} {XNAT_HOST}/data/projects/{project}/subjects/{subject}/experiments?columns={fieldvalue_key},{fieldname_key}"
print("req=", req)
resp = os.popen(req).read() # comes as string
resp = eval(resp)['ResultSet']['Result'] # eval and get the dict


# Parse response
print("resp", resp)

flag = [i[fieldvalue_key] for i in resp if (i[fieldname_key].lower() == custom_variable and i[session_key] == session)][0]
# print("flag =", flag)

if flag == 'true':
	sys.exit(0)
else:
	sys.exit(1)
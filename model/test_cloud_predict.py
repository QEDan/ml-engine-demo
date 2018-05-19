#!/usr/bin/env python

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()

api = discovery.build('ml', 'v1', credentials=credentials,
                      discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')
PROJECT = 'ml-engine-demo-201303'
parent = 'projects/{}/models/{}/versions/{}'.format(PROJECT, 'regression_model', 'regression_model_v_0_2')

request_data = {'instances': [{'X':500.0}, {'X':1.0}]}

response = api.projects().predict(body=request_data, name=parent).execute()

print("response={}".format(response))

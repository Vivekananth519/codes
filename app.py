from datetime import datetime
import json
import uuid
import joblib
import numpy
import pytz
from fastapi import FastAPI, HTTPException
import pandas
from pydantic import BaseModel
import os
import traceback
from azure.storage.blob import ContainerClient, BlobServiceClient
import yaml
from helper import DBHelper, load_best_model

os.environ['deployment_env'] = 'dev'

# Path information
containerName= 'analytics'
connectionString_TCNP = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
tcnp_vidhi_l0 = ContainerClient.from_connection_string(conn_str=connectionString_TCNP,container_name=containerName)
train_path = f'/Databricks/PipelinesData/PSC_DL/DSF/train/v1/'


print('Loading Model', flush=True)
model, cutoff = load_best_model(train_path=train_path, tcnp_vidhi_l0 = tcnp_vidhi_l0)
print('Model Load successful', flush=True)

app = FastAPI()

db_appdata = DBHelper(database='appdata')
db_insight = DBHelper(database='staging')


class DSFInput(BaseModel):
    POLNUM: str
    ANNUAL_PREMIUM: int
    SUM_ASSURED: int
    POLICY_TERM: int
    PRODUCT_CODE: str
    EDUCATION_QUALIFICATION: int
    CUST_OCCUPATION: int
    ANNUAL_INCOME: int
    CUST_AGE: int
    CAMS_CODE: str
    MARITAL_STATUS: str


class DataModel:
    def __init__(self, dsf_input):
        self.dsf_input = dict(dsf_input)


    def validate(self):
        data_validation = []
        for key, value in self.dsf_input.items():
            if value == '':
                data_validation.append(f"{key} consist blank Value")
            if value == 'NA':
                data_validation.append(f"{key} consist <NA> Value")

        if len(data_validation) > 1:
            return False, f"Validation Failed : {data_validation}"
        return True, ""


    @staticmethod
    def get_data_manager_code(manager_code):
        manager_data_query = f''' SELECT * FROM dsf_onboarding_api_cams_perf  WHERE CAMS_CODE = '{manager_code}' '''
        status, manager_data = db_insight.get_query_data(query=manager_data_query)
        if len(manager_data) == 0:
            status = False
        return status, manager_data

    @staticmethod
    def get_data_product_code(product_code):
        product_data_query = f''' SELECT * FROM dsf_onboarding_api_product_type WHERE PROD_CD = '{product_code}' '''
        status, product_data = db_insight.get_query_data(query=product_data_query)
        if len(product_data) == 0:
            status = False
        return status, product_data

    @staticmethod
    def get_data_cams_details(CAMS_CODE):
        cams_data_query = f''' SELECT * FROM DSF_ONBOARDING_CAMS_DETAILS WHERE WPO_CAMS_CD = '{CAMS_CODE}' '''
        status, cams_data = db_insight.get_query_data(query=cams_data_query)
        if len(cams_data) == 0:
            status = False
        return status, cams_data


    @staticmethod
    def edu_cat(predict_data):
        if predict_data == 4:  # graduate
            return 'GRADUATE'
        elif predict_data == 5:  # post graduate
            return 'POST GRADUATE'
        elif predict_data == 2:  # 10th
            return 'UNDER GRADUATE'
        elif predict_data == 3:  # 12th
            return 'UNDER GRADUATE'
        elif predict_data == 14:  # diploma
            return 'UNDER GRADUATE'
        elif predict_data == 15:  # vocational/iti
            return 'UNDER GRADUATE'
        elif predict_data == 7:  # illeterate
            return 'UNDER GRADUATE'
        elif predict_data == 1:  # below ssc
            return 'UNDER GRADUATE'
        else:
            return 'OTHERS'


    @staticmethod
    def occ(predict_data):
        if predict_data==2:    # self employed
            return 'SELF EMPLOYED'
        elif predict_data==3:   #  business owner
            return 'BUSINESS OWNER'
        elif predict_data==1:  #  salaried
            return 'SALARIED'
        elif predict_data==6:  # student/juvenile
            return 'NON EARNERS'
        elif predict_data==4:  # housewife
            return 'NON EARNERS'
        elif predict_data==7:  # others
            return 'OTHERS'
        elif predict_data==8:  # agriculture/farmer
            return 'SELF EMPLOYED'
        else:
            return 'OTHERS'

   


    def process_data(self, predict_data):
        # EDUCATION_QUALIFICATION_cat
        predict_data['EDUCATION_QUALIFIATION_CAT'] = predict_data['EDUCATION_QUALIFIATION_CAT'].astype(int)
        predict_data['EDUCATION_QUALIFIATION_CAT'] = predict_data['EDUCATION_QUALIFIATION_CAT'].fillna(4)
        predict_data['EDUCATION_QUALIFIATION_CAT'] = predict_data['EDUCATION_QUALIFIATION_CAT'].apply(self.edu_cat)
        
        # Occupation 
        predict_data['OCCUPATION_CLASS_CAT'] = predict_data['OCCUPATION_CLASS_CAT'].astype(int)
        predict_data['OCCUPATION_CLASS_CAT'] = predict_data['OCCUPATION_CLASS_CAT'].fillna(1)
        predict_data['OCCUPATION_CLASS_CAT'] = predict_data['OCCUPATION_CLASS_CAT'].apply(self.occ)

        # # Annual Income
        # predict_data['apbyincome'] = predict_data['ANP'] / predict_data['ANNUAL_INCOME']
        predict_data['INCOME_BUCKET'] = numpy.where(predict_data['ANNUAL_INCOME']<300000, 'Upto 3L', 
                                                    numpy.where(predict_data['ANNUAL_INCOME']<500000, '3 to 5L', 
                                                                numpy.where(predict_data['ANNUAL_INCOME']<1000000, '5 to 10L', 
                                                                        numpy.where(predict_data['ANNUAL_INCOME']<1500000, '10 to 15L', 
                                                                                numpy.where(predict_data['ANNUAL_INCOME']<2000000, '15 to 20L', 
                                                                                        numpy.where(predict_data['ANNUAL_INCOME']<2500000, '20 to 25L', '25L & Above'))))))
        # predict_data['apbyincome'] = predict_data['apbyincome'].replace([numpy.inf, -numpy.inf], numpy.nan)
        # predict_data['apbyincome'] = predict_data['apbyincome'].fillna(0).round(3)
        return predict_data


    @staticmethod
    def decile_score_model(probability):

        if probability >= cutoff[0]:
            return 1
        elif cutoff[1] <= probability < cutoff[0]:
            return 2
        elif cutoff[2] <= probability < cutoff[1]:
            return 3
        elif cutoff[3] <= probability < cutoff[2]:
            return 4
        elif cutoff[4] <= probability < cutoff[3]:
            return 5
        elif cutoff[5] <= probability < cutoff[4]:
            return 6
        elif cutoff[6] <= probability < cutoff[5]:
            return 7
        elif cutoff[7] <= probability < cutoff[6]:
            return 8
        elif cutoff[8] <= probability < cutoff[7]:
            return 9
        else:
            return 10


@app.post('/predict')
def predict(request_data: DSFInput):
    request_id = uuid.uuid4()
    environment = os.environ['deployment_env']
    try:
        # ********************************** INIT ****************************************
        timezone = pytz.timezone('Asia/Kolkata')
        request_date = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S.%f')
        request_data = dict(request_data)
        print(f'{"=" * 150}\nRequest received : {request_id}\n', json.dumps(request_data, indent=3), flush=True)
        policy_number = request_data['POLNUM']
        data_model = DataModel(request_data)

        api_request_columns = ['requestId', 'requestDate', 'environment', 'policyNo', 'requestData', 'status',
                               'comments']
        api_request_data = [request_id, request_date, environment, policy_number, str(json.dumps(request_data)),
                            'Request Init', '']
        db_appdata.insert_record(table='dsf_onboarding_api_request_response', columns=api_request_columns,
                                 data=api_request_data)
        

        # ********************************** VALIDATIONS **********************************
        print(f'Validating Data', flush=True)
        validation_columns = ['status', 'comments']
        status, comment = data_model.validate()
        if not status:
            validation_values = ['Validation Failed', comment]
            db_appdata.update_api_request(request_id=request_id, columns=validation_columns, values=validation_values)
            return HTTPException(status_code=500, detail=comment)

        validation_data = ['Validated Successfully', comment]
        db_appdata.update_api_request(request_id=request_id, columns=validation_columns, values=validation_data)
        print(f'Data validated successfully', flush=True)

         # ********************************** DATA FETCH **********************************
        print(f'Fetching Data : [MANAGER CODE : {request_data["CAMS_CODE"]}] ', flush=True)
        manager_status, manager_data = data_model.get_data_manager_code(request_data['CAMS_CODE'])
        print(f'Fetching Data : [PRODUCT CODE : {request_data["PRODUCT_CODE"]}] ', flush=True)
        product_status, product_data = data_model.get_data_product_code(request_data['PRODUCT_CODE'])
        print(f'Fetching Data : [CAMS CODE : {request_data["CAMS_CODE"]}] ', flush=True)
        cams_status, cams_data = data_model.get_data_cams_details(request_data['CAMS_CODE'])

        CAMS_LAPSED_RATIO = 0.2
        CAMS_VINTAGE = 0
        # WPO_ZONE_NAME_CAT = 1
        # PROD_TYPE_CAT = 3
        BRANCH_LAPSE_RATE = 0.2

        if manager_status:
            CAMS_LAPSED_RATIO = manager_data[0]['CAMS_LAPSED_RATIO']
            data_fetch_values = ['Manager Data Fetch Successfully', json.dumps(manager_data[0])]
            data_fetch_columns = ['status', 'managerData']
            db_appdata.update_api_request(request_id=request_id, columns=data_fetch_columns, values=data_fetch_values)
        if product_status:
            PROD_TYPE_CAT = product_data[0]['PROD_TYPE_CAT']
            data_fetch_values = ['Product Data Fetch Successfully', json.dumps(product_data[0])]
            data_fetch_columns = ['status', 'productData']
            db_appdata.update_api_request(request_id=request_id, columns=data_fetch_columns, values=data_fetch_values)
        if cams_status:
            CAMS_VINTAGE = cams_data[0]['CAMS_VINTAGE']
            WPO_ZONE_NAME_CAT = cams_data[0]['WPO_ZONE_NAME_CAT']
            data_fetch_values = ['Branch Data Fetch Successfully', json.dumps(str(cams_data[0]))]
            data_fetch_columns = ['status', 'branchData']
            db_appdata.update_api_request(request_id=request_id, columns=data_fetch_columns, values=data_fetch_values)

        # ********************************** PRE-PROCESSING *******************************
        predict_data = pandas.DataFrame([{
            "POLNUM": request_data['POLNUM'],
            "POLICY_TERM": request_data['POLICY_TERM'],
            "ANNUAL_PREMIUM": request_data['ANNUAL_PREMIUM'],
            "SUM_ASSURED": request_data['SUM_ASSURED'],
            "EDUCATION_QUALIFICATION": request_data['EDUCATION_QUALIFICATION'],
            "OCCUPATION_CLASS": request_data['CUST_OCCUPATION'], 
            "ANNUAL_INCOME": request_data['ANNUAL_INCOME'],
            "INSURED_AGE": request_data['CUST_AGE'],
            # "PROD_TYPE_CAT" : PROD_TYPE_CAT,
            # "WPO_ZONE_NAME_CAT": WPO_ZONE_NAME_CAT,
            "AGENT_VINTAGE": CAMS_VINTAGE,
            "AGENT_LAPSE_RATE": CAMS_LAPSED_RATIO,
            "MARITAL_STATUS": request_data['MARITAL_STATUS'],
            "BRANCH_LAPSE_RATE": BRANCH_LAPSE_RATE
             }])

        # ********************************** Before Pre-processing *******************************
        prediction_data_columns = ['preProcessing', 'postProcessing', 'predictionData']
        before_processing = predict_data.to_dict(orient='records')[0]
        db_appdata.update_api_request(request_id=request_id, columns=prediction_data_columns[:1],
                                      values=[json.dumps(before_processing)])
        print('Processing Data \n', json.dumps(before_processing, indent=3), flush=True)

        # ********************************** Pre-processing *******************************
        predict_data = data_model.process_data(predict_data)
        after_processing = predict_data.to_dict(orient='records')[0]
        db_appdata.update_api_request(request_id=request_id, columns=prediction_data_columns[1:2],
                                      values=[json.dumps(after_processing)])
        print('Processed Data \n', json.dumps(after_processing, indent=3), flush=True)

        # ********************************** Prediction data pre-prepare *******************************
        remove_columns = ["POLNUM"]
        predict_data = predict_data.drop(remove_columns, axis=1)
        prediction_data = predict_data.to_dict(orient='records')[0]
        db_appdata.update_api_request(request_id=request_id, columns=prediction_data_columns[2:3],
                                      values=[json.dumps(prediction_data)])
        print('Prediction Data \n', json.dumps(prediction_data, indent=3), flush=True)


        # ********************************** Predict results *******************************

        probability = model.predict_proba(predict_data)[:, 1]
        decile = data_model.decile_score_model(probability)
        print("Decile Score :", decile, flush=True)
        print("Probability :", probability, flush=True)
        prediction_results_columns = ['Decile_Rank', 'Probability_Score']
        db_appdata.update_api_request(request_id=request_id, columns=prediction_results_columns,
                                      values=[decile, probability[0]])

        response_dict = {"Decile Score": decile}

        requests_status_columns = ['status', 'comments']
        db_appdata.update_api_request(request_id=request_id, columns=requests_status_columns,
                                      values=['Completed', ''])
        return response_dict

    except Exception as Err:
        print('Error : \n', traceback.format_exc(), flush=True)
        exception_columns = ['status', 'comments']
        db_appdata.update_api_request(request_id=request_id, columns=exception_columns,
                                      values=['Failed', traceback.format_exc()])
        raise HTTPException(status_code=500, detail=str(Err))


@app.get('/')
def get():
    return {'status': 200}

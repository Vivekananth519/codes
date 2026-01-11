import os
import pyodbc
import mlflow
from azure.storage.blob import ContainerClient, BlobServiceClient
import yaml
import joblib
import pickle

def download_yaml_data(remote_path, tcnp_vidhi_l0 = None):
    downloaded_blob = tcnp_vidhi_l0.download_blob(remote_path).readall().decode("utf-8")
    output = yaml.safe_load(downloaded_blob)
    return output

def download_pickle_data(remote_path,use_joblib = False, tcnp_vidhi_l0 = None):
    downloaded_blob = tcnp_vidhi_l0.download_blob(remote_path).readall()
    if use_joblib:
        with open('temp_files/temp.pkl','wb') as file:
            file.write(downloaded_blob)
        output = joblib.load('temp_files/temp.pkl')
    else:
        output = pickle.load(downloaded_blob)
    del downloaded_blob
    return output

class DBHelper:
    def __init__(self, database='appdata') -> None:
        if database == 'appdata':
            self.connection_str = os.environ['SQLCONNSTR_appdata_db']
        if database == 'staging':
            self.connection_str = os.environ['SQLCONNSTR_staging_db']

    def run_query(self, query, parameters=None):
        with pyodbc.connect(self.connection_str) as cnxn:
            try:
                if parameters is None:
                    curr = cnxn.execute(query)
                else:
                    curr = cnxn.execute(query, parameters)
                if curr.rowcount >= 0:
                    return True, ''
            except Exception as Err:
                print(f'Failed to insert/Update Record :{Err}', flush=True)
                print(f'Query : {query}\nData : {parameters}', flush=True)
                return False, str(Err)

    def get_query_data(self, query):
        with pyodbc.connect(self.connection_str) as cnxn:
            try:
                data = []
                curr = cnxn.execute(query)
                columns = [column[0] for column in curr.description]
                for row in curr.fetchall():
                    data.append(dict(zip(columns, row)))
                return True, data
            except Exception as Err:
                print(f'Failed to read data :{Err}', flush=True)
                print(f'Query : {query}', flush=True)
                return False, str(Err)

    def insert_record(self, table, columns, data):
        query = f'''INSERT INTO {table}({','.join(columns)}) 
        values({','.join(['?' for _ in range(len(data))])})'''
        status, comment = self.run_query(query=query, parameters=data)
        return status, comment

    def update_api_request(self, request_id, columns, values):
        query = f'''UPDATE Cboi_Onboarding_API_Request_Response SET {','.join([f'{name}=?' for name in columns])} 
        where requestId=?
        '''
        status, comment = self.run_query(query=query, parameters=values + [request_id])
        return status, comment
    
def load_best_model(train_path: str = None, tcnp_vidhi_l0 = None):
    config_file = download_yaml_data(f'{train_path}config/config.yaml', tcnp_vidhi_l0 = tcnp_vidhi_l0)
    # client = mlflow.MlflowClient()
    # model_versions = client.search_model_versions(f"run_id='{config_file['best_run_id']}'")
    # best_model = model_versions[0].source
    best_model = config_file['best_model_type']
    # model = mlflow.sklearn.load_model(best_model)
    model = download_pickle_data(f'{train_path}{best_model}.pkl', use_joblib = True, tcnp_vidhi_l0 = tcnp_vidhi_l0)
    return model, config_file['cutOff']

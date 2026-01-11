# Importing relevant libraries
import pandas as pd
import sys
import numpy as np
import seaborn as sns
import optuna
# from sklearn import metrics
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn import metrics
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.Model_Summary import Summary_Automation
from src.data_fetch import *
pd.set_option('display.max_columns', None)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from functools import partial

# This keeps column names after transformations so downstream steps see names.
from sklearn import set_config
set_config(transform_output="pandas")

class model_evaluate:

    def __init__(self, processed_path, config_file_path, model_path, report_path, valid = None, vidhi_con = None, model_type = None, experiment_name = None, channel = None, n_trials = 10):
        self.vidhi_con = vidhi_con
        self.processed_path = processed_path
        self.train = download_parquet_data(self.processed_path, 'train.parquet', self.vidhi_con)
        self.eval = download_parquet_data(self.processed_path, 'test.parquet', self.vidhi_con)
        self.config_file_path = config_file_path
        if valid is not None:
            self.valid = download_parquet_data(self.processed_path, 'oot.parquet', self.vidhi_con)
        else:
            self.valid = valid
        self.config_file = download_yaml_data(self.config_file_path + 'config.yaml', tcnp_vidhi_l0=self.vidhi_con)
        self.target = self.config_file['target']
        self.s_feature = self.config_file['selected_features']
        self.model_path = model_path
        self.report_path = report_path
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.channel = channel
        self.n = n_trials

    # Decile summary generation function
    @staticmethod
    def decile_fun(score, prob_list):
        if score >= prob_list[0]:
            decile = 1
        elif score >= prob_list[1] < prob_list[0]:
            decile = 2
        elif score >= prob_list[2] < prob_list[1]:
            decile = 3
        elif score >= prob_list[3] < prob_list[2]:
            decile = 4
        elif score >= prob_list[4] < prob_list[3]:
            decile = 5
        elif score >= prob_list[5] < prob_list[4]:
            decile = 6
        elif score >= prob_list[6] < prob_list[5]:
            decile = 7
        elif score >= prob_list[7] < prob_list[6]:
            decile = 8
        elif score >= prob_list[8] < prob_list[7]:
            decile = 9
        else:
            decile = 10
        return decile
    
    @staticmethod
    def detect_feature_types(X: pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        return num_cols, cat_cols

    
    def log_explainerdashboard_artifacts(self, model, dashboard_title: str = "Model Explainer"):
        # Construct explainer on validation split (you can switch to your full test set)
        explainer = ClassifierExplainer(
            model = model,                # Pipeline with .predict_proba is fine
            X=self.valid[self.s_feature],
            y=self.valid[self.target],
            model_output='probability'  
        )

        
        db = ExplainerDashboard(
            explainer,
            title=dashboard_title,
            importances=False,
            contributions=False,
            shap_dependence=False,
            shap_interaction=False,
            # optionally disable other tabs too
            whatif=False,
            decision_trees=False,
            pdp=False,        # if your version exposes PDP
            precision=False
        )

        # Persist explainer to disk (joblib is the recommended default)
        db.save_html("explainer_dashboard.html")
        joblib.dump(explainer, "explainer.joblib")  # explainer.dump(...) also works
        mlflow.log_artifact("explainer.joblib", artifact_path='dashboard')
        mlflow.log_artifact("explainer_dashboard.html", artifact_path='dashboard')
        os.remove('explainer.joblib')
        os.remove('explainer_dashboard.html')

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols, cat_cols = self.detect_feature_types(X)

        num_pipeline = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler(with_mean=True))
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_encoder", ce.TargetEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ],
            remainder="drop", # or 'passthrough' if you want other columns
            verbose_feature_names_out=False
        )
        return preprocessor

    # Decile Summary
    def decile_summary(self, prob, actual, prob_list, req_dig=True):
        Decile = [self.decile_fun(col, prob_list) for col in prob]
        results = pd.DataFrame({'Sum': actual,'Count': actual,'Probability': prob}).reset_index(drop = True)
        results['Decile'] = Decile
        decile_sum = results.groupby('Decile')['Sum'].sum().reset_index()
        decile_cumsum = decile_sum['Sum'].cumsum().reset_index()
        decile_cumsum.columns = ['Decile', 'CumSum']
        decile_cumsum['Decile'] = decile_cumsum['Decile'] + 1
        decile_count = results.groupby('Decile')['Count'].count().reset_index()
        Decile_sum=decile_sum.join(decile_count.set_index('Decile'),on='Decile')
        Decile_sum=Decile_sum.join(decile_cumsum.set_index('Decile'),on='Decile')
        Decile_sum['gain'] = Decile_sum['CumSum']/decile_sum['Sum'].sum()
        Decile_sum['Event Rate']=Decile_sum['Sum']/Decile_sum['Count']
        # if req_dig:
        #     ax = plt.figure(figsize=(12, 8))
        #     plt.title('Decile Score - Cumulative Gain Plot')
        #     sns.lineplot(
        #         x = Decile_sum['Decile'], y=Decile_sum['gain']*100, label='model')
        #     sns.lineplot(x=Decile_sum['Decile'],
        #                 y=Decile_sum['Decile']*10, label='avg')

        #     return Decile, Decile_sum, ax
        # else:
        return Decile, Decile_sum
    
    @staticmethod
    def get_optimum_threshold(df, target='Target', score='Score'):

        fpr, tpr, thresholds = metrics.roc_curve(df[target], df['Score'])
        roc_auc = metrics.auc(fpr, tpr)

        ####################################
        # The optimal cut off would be where tpr is high and fpr is low
        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        ####################################
        i = np.arange(len(tpr))  # index for df
        roc = pd.DataFrame({
            'fpr': pd.Series(fpr, index=i),
            'tpr': pd.Series(tpr, index=i),
            '1-fpr': pd.Series(1-fpr, index=i),
            'tf': pd.Series(tpr - (1-fpr), index=i),
            'thresholds': pd.Series(thresholds, index=i)
        })

        cutoff_df = roc.iloc[(roc.tf-0).abs().argsort()
                             [:1]].reset_index(drop=True)

        return roc_auc, cutoff_df 


    @staticmethod
    def get_classification_report(clf, X, y, thres=0.5):
        x_train_proba = clf.predict_proba(X)[:, 1]
        x_train_pred = np.where(x_train_proba > thres, 1, 0)
        clf_report = classification_report(y, x_train_pred)
        return clf_report
        
    def model_result(self, test_df, features, target, iftrain = 'Yes', model = None, threshold = None):
        train_result = pd.DataFrame(test_df[target].copy()) 
        y_pred_train =[x[1] for x in model.predict_proba(test_df[features])] # feature needs to be replaced with s_feature
        train_result['Score'] = y_pred_train
        roc_auc, cutoff_df = self.get_optimum_threshold(train_result, target=target)
        if threshold is None:
            print('Optimal Cutoff: ',cutoff_df['thresholds'].values[0])
            thresholds = cutoff_df['thresholds'].values[0]
            print(f'ROC AUC Score at an Optimum Threshold: \n {cutoff_df} \n ROC AUC Score:',roc_auc_score(test_df[target],y_pred_train))
        else: 
            thresholds = np.float64(threshold)
        train_pred = np.where(y_pred_train > thresholds, 1, 0)
        class_report = classification_report(test_df[target], train_pred)
        print('Classification Report: \n', class_report)
        if iftrain == 'Yes':
            train_result['Decile'] = 10 - pd.qcut(train_result['Score'], 10, labels=False, duplicates='drop')
            decile_prob = list(train_result.groupby('Decile')['Score'].min())
            return train_result, decile_prob, thresholds, classification_report(test_df[target], train_pred, output_dict=True), roc_auc
        else:
            return train_result, classification_report(test_df[target], train_pred, output_dict=True), roc_auc

    def objective(self, trial, model_type):
        if model_type == 'lightgbm':
            params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'subsample': trial.suggest_loguniform('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.5, 1),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 1)
            }
            model = lgb.LGBMClassifier(**params, class_weight = 'balanced', random_state = 42)
        elif model_type == 'xgboost':
            params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'gamma': trial.suggest_loguniform('gamma', 0.01, 5),
            'subsample': trial.suggest_loguniform('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.5, 1),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 1)
            }
            model = xgb.XGBClassifier(**params, random_state = 42)
        elif model_type == 'randomforest':
            params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = RandomForestClassifier(**params, class_weight = 'balanced', random_state = 42)
        elif model_type == 'logistic':
            params = {
            'C': trial.suggest_loguniform('C', 0.01, 10)
            }
            model = LogisticRegression(**params, random_state = 42)
        preprocessor = self.build_preprocessor(self.train[self.s_feature])
        model = Pipeline(steps = [('preprocess', preprocessor),('model', model)])
        model.fit(self.train[self.s_feature], self.train[self.target])
        preds = model.predict_proba(self.eval[self.s_feature])[:, 1]
        auc = roc_auc_score(self.eval[self.target], preds)
        return auc

    def optimize(self, objective = None):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n)
        best_trial = study.best_trial
        print(f'Best parameters: {best_trial.params}')
        print(f'Best accuracy score: {best_trial.value}')
        self.best_params = best_trial.params
        return self.best_params

    # Function to optimize and evaluate models
    def optimize_and_evaluate(self):

        mlflow.set_experiment(experiment_name=self.experiment_name)

        models = [  'lightgbm',
                    'xgboost',
                    'randomforest',
                    'logistic']

        results = {}
        # model_type = self.model_type
        # space = models[model_type]
        for model_type in models:
            objective_ = lambda trial: self.objective(trial, model_type = model_type)
            best = self.optimize(objective = objective_)
            
            model = None
            if model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**best, class_weight = 'balanced', random_state = 42)
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(**best, random_state = 42)
            elif model_type == 'randomforest':
                model = RandomForestClassifier(**best, class_weight='balanced', random_state = 42)
            elif model_type == 'logistic':
                model = LogisticRegression(**best, class_weight='balanced', random_state = 42)
            
            preprocessor = self.build_preprocessor(self.train[self.s_feature])
            model = Pipeline(steps = [('preprocess', preprocessor),('model', model)])
            
            # Fitting the model with the best parameters
            model.fit(self.train[self.s_feature], self.train[self.target])

            # Getting feature importance
            if model_type != 'logistic':
                importance = pd.DataFrame({
                    'Feature': self.s_feature,
                    'Importance': model['model'].feature_importances_
                }).sort_values(by='Importance', ascending=False)
            else:
                # Getting feature importance
                model_coeff = model['model'].coef_

                # Creating a DataFrame for better visualization
                importance = pd.DataFrame({
                    'Feature': self.s_feature,
                    'Importance': model_coeff[0]
                }).sort_values(by='Importance', ascending=False)
            importance['Abs_Importance'] = abs(importance['Importance'])
            importance['Percentage']=(importance['Abs_Importance']/importance['Abs_Importance'].sum())*100
            importance['Percentage']=importance['Percentage'].round(2)
            importance = importance.sort_values('Percentage', ascending = False)

            # if not os.path.exists('../../Model Summary/'):
            #     os.makedirs('../../Model Summary/')

            if self.valid is None:
                Automation = Summary_Automation()
                psi, csi = Automation.master_call(
                    df_train=self.train,
                    sel_features =  self.s_feature,
                    df_ytrain=self.train[self.target],
                    df_test=self.eval,
                    df_ytest=self.eval[self.target],
                    df_oot=self.eval,
                    df_yoot=self.eval[self.target],
                    target=self.target, 
                    model=model,
                    prob_bins=None,
                    save_path=f'{model_type}_with_{self.n}_trials_Model_Summary.xlsx',
                    n_bins = 10)
            else:
                Automation = Summary_Automation()
                psi, csi = Automation.master_call(
                    df_train=self.train,
                    sel_features =  self.s_feature,
                    df_ytrain=self.train[self.target],
                    df_test=self.eval,
                    df_ytest=self.eval[self.target],
                    df_oot=self.valid,
                    df_yoot=self.valid[self.target],
                    target=self.target, 
                    model=model,
                    prob_bins=None,
                    save_path=f'{model_type}_with_{self.n}_trials_Model_Summary.xlsx',
                    n_bins = 10)

            # Train Result
            train_result, decile_prob, threshold, train_class_report, train_roc_auc = self.model_result(self.train, self.s_feature, self.target, iftrain = 'Yes', model = model, threshold = None)
            training_decile, train_decile_summary = self.decile_summary(prob = train_result['Score'], actual = train_result[self.target], prob_list = decile_prob)
            
            # Test Result
            eval_result, eval_class_report, eval_roc_auc = self.model_result(self.eval, self.s_feature, self.target, iftrain = 'No', model = model, threshold = threshold)
            eval_decile, eval_decile_summary = self.decile_summary(prob = eval_result['Score'], actual = eval_result[self.target], prob_list = decile_prob)

            if self.valid is None:
                results[model_type] = [model, train_decile_summary, eval_decile_summary, train_class_report, eval_class_report]
            else:
                # Test Result
                valid_result, valid_class_report, valid_roc_auc = self.model_result(self.valid, self.s_feature, self.target, iftrain = 'No', model = model, threshold = threshold)
                valid_decile, valid_decile_summary = self.decile_summary(prob = valid_result['Score'], actual = valid_result[self.target], prob_list = decile_prob, req_dig=False)
                results[model_type] = [model, train_decile_summary, eval_decile_summary, valid_decile_summary, train_class_report, eval_class_report, valid_class_report]
            pickle_dump(model, f'{model_type}.pkl', self.model_path, self.vidhi_con)
            ### MLFLOW Model log and tracking
            with mlflow.start_run(run_name = f'{model_type} Tuned with Optuna using {self.n} trials'):
                mlflow.log_params({f"{k}": v for k, v in best.items()})

                # Log ROC AUC for train, test
                mlflow.log_metric("train_roc_auc", round(train_roc_auc,4))
                mlflow.log_metric("test_roc_auc", round(eval_roc_auc,4))
                mlflow.log_metric("Optimum CutOff", round(threshold,4))

                #Log Accuracy for train, test
                mlflow.log_metric("train_accuracy", round(train_class_report['accuracy'], 4))
                mlflow.log_metric("test_accuracy", round(eval_class_report['accuracy'],4))

                #Log Precision for train, test
                mlflow.log_metric("train_precision", round(train_class_report['1']['precision'],4))
                mlflow.log_metric("test_precision", round(eval_class_report['1']['precision'],4))

                #Log Recall for train, test
                mlflow.log_metric("train_recall", round(train_class_report['1']['recall'],4))
                mlflow.log_metric("test_recall", round(eval_class_report['1']['recall'],4))

                #Log F1 Score for train, test
                mlflow.log_metric("train_f1_score", round(train_class_report['macro avg']['f1-score'],4))
                mlflow.log_metric("test_f1_score", round(eval_class_report['macro avg']['f1-score'],4))

                #log 3 Deciles capture rate
                mlflow.log_metric('train_d3_capture', round(results[model_type][1]['gain'][2],4))
                mlflow.log_metric('test_d3_capture', round(results[model_type][2]['gain'][2],4))
                mlflow.log_metric('d3_diff_train_test', abs(round(results[model_type][2]['gain'][2],4) - round(results[model_type][2]['gain'][1],4)))

                if self.valid is not None:
                    #log above all metrics for oot
                    mlflow.log_metric("oot_roc_auc", round(valid_roc_auc,4))
                    mlflow.log_metric("oot_accuracy", round(valid_class_report['accuracy'],4))
                    mlflow.log_metric("oot_precision", round(valid_class_report['1']['precision'],4))
                    mlflow.log_metric("oot_recall", round(valid_class_report['1']['recall'],4))
                    mlflow.log_metric("oot_f1_score", round(valid_class_report['macro avg']['f1-score'],4))
                    mlflow.log_metric('oot_d3_capture', round(results[model_type][3]['gain'][2],4))
                
                # Exporting PSI and CSI report
                psi.to_csv('psi_report.csv', index=False)
                mlflow.log_artifact("psi_report.csv", artifact_path="drift_report")
                os.remove('psi_report.csv')
                csi.to_csv('csi_report.csv', index=False)
                mlflow.log_artifact("csi_report.csv", artifact_path="drift_report")
                os.remove('csi_report.csv')
                # log feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance['Feature'], importance['Percentage'])
                ax.set_xlabel("Feature Importance")
                ax.set_ylabel("Feature Name")
                ax.set_title("Model Feature Importance")
                plt.tight_layout()
                plt.gca().invert_yaxis()
                mlflow.log_figure(fig, f"{model_type}_feature_importance.png")
                plt.close(fig)

                # log model summary
                mlflow.log_artifact(local_path=f'{model_type}_with_{self.n}_trials_Model_Summary.xlsx', artifact_path="evaluation_results")
                self.log_explainerdashboard_artifacts(model)
                # log model
                mlflow.sklearn.log_model(model, "model", signature = infer_signature(self.train[self.s_feature].head(), self.train[self.target].head()),
                input_example = self.train[self.s_feature].head(), registered_model_name = f'Best {model_type} model for {self.channel}')
            # Upload model summary to blob storage
            upload_excel_file(f'{model_type}_with_{self.n}_trials_Model_Summary.xlsx', self.report_path, self.vidhi_con)
            print('Model run and evaluation is completed')
        # Staging the best model based on highest D3 capture rate on Test/ OOT set
        client = MlflowClient()
        if self.valid is None:
            METRIC_NAME = 'd3_diff_train_test'
            runs = client.search_runs(
            experiment_ids=client.get_experiment_by_name(self.experiment_name).experiment_id,
            order_by=[f"metrics.{METRIC_NAME} DESC"],
            max_results=1
            )
        else:
            METRIC_NAME = 'd3_diff_train_test'
            runs = client.search_runs(
            experiment_ids=client.get_experiment_by_name(self.experiment_name).experiment_id,
            order_by=[f"metrics.{METRIC_NAME} DESC"],
            max_results=1
            )
        best_run = runs[0]
        best_metric_value = best_run.data.metrics[METRIC_NAME]
        
        print(f"Best Run ID: {best_run.info.run_id} with {METRIC_NAME}: {best_metric_value}")

        # --- 1. Find the logged model version ---
        # Get the latest model version for the run
        model_versions = client.search_model_versions(f"run_id='{best_run.info.run_id}'")
        latest_version = model_versions[0].version
        REGISTERED_MODEL_NAME = model_versions[0].name #'BestClassifer_Lapsation'
        PROMOTE_TO = 'Staging'

        # --- 2. Promote Model in Registry ---
        # Transition the best version to the Staging stage
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest_version,
            stage=PROMOTE_TO
        )

        mlflow.set_tag("explainer_url", "http://localhost:8050/")

        # Logging best model details in the yaml config file
        if REGISTERED_MODEL_NAME.find('lightgbm')>0:
            best_model_type = 'lightgbm'
        elif REGISTERED_MODEL_NAME.find('xgboost')>0:
            best_model_type = 'xgboost'
        elif REGISTERED_MODEL_NAME.find('randomforest')>0:
            best_model_type = 'randomforest'
        elif REGISTERED_MODEL_NAME.find('logistic')>0:
            best_model_type = 'logistic'

        # Getting the experiment details for further use
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = experiment.experiment_id
        experiment_path = experiment.artifact_location

        # Writing the best model details to Config file
        self.config_file['best_model_name'] = REGISTERED_MODEL_NAME
        self.config_file['latest_version'] = latest_version
        self.config_file['best_model_type'] = best_model_type
        self.config_file['experiment_id'] = experiment_id
        self.config_file['experiment_path'] = experiment_path
        self.config_file['best_run_id'] = best_run.info.run_id
        self.config_file['stage'] = PROMOTE_TO
        config_file_yaml = 'config.yaml'
        with open(config_file_yaml, 'w') as file:
            yaml.dump(self.config_file, file, default_flow_style=False, sort_keys=False)
        # Upload the config file
        upload_excel_file(config_file_yaml, self.config_file_path, self.vidhi_con)
        
        print(f"Model {REGISTERED_MODEL_NAME} version {latest_version} transitioned to {PROMOTE_TO}.")
        return results

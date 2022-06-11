import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime
import dateutil.relativedelta
import pickle

@task
def get_paths(date = None):

    if date == None:
        date = str(datetime.now().date())
    
    date = datetime.strptime(date, "%Y-%m-%d")
    date1 = date - dateutil.relativedelta.relativedelta(months=2)
    date2 = date - dateutil.relativedelta.relativedelta(months=1)

    train_path = './data/fhv_tripdata_'+str(date1.date().year)+'-'+str(date1.date().month).zfill(2)+'.parquet'
    val_path = './data/fhv_tripdata_'+str(date2.date().year)+'-'+str(date2.date().month).zfill(2)+'.parquet'

    return train_path, val_path


@task
def read_data(path):

    logger = get_run_logger()

    try:
        df = pd.read_parquet(path)
    except:
        print("Check file Path or if such a file exists")

    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main_deployment(date = None):

    train_path, val_path = get_paths(date).result()

    print(train_path)
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    ## Saving the model and dictionary vectorizer
    pickle.dump(lr, open("./artifacts/model-{date}.pkl".format(date = date), "wb"))
    pickle.dump(dv, open("./artifacts/dv-{date}.pkl".format(date = date), "wb"))

#main_deployment('2021-08-15')

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main_deployment,
    name="cron_schedule_deployment_new",
    schedule=CronSchedule(cron="0 9 15 * *", timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml_cron"]
)
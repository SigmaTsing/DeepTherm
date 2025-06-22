import statsmodels.api as sm
import numpy as np
import pandas as pd
import calendar
import patsy as pt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict_poisson(df, ranges, return_train = False):
    train_dates = ranges['train']
    test_dates = ranges['test']

    # df = pd.read_excel('./data/Seville 2023 (003).xlsx', sheet_name='All Data')
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.day_name()
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['DayOfWeekNumeric'] = df['DayOfWeek'].map(day_mapping)
    date_difference_years=20

    time_vector = []
    # print("creating time vector ",int(train_dates[0].split('-')[0]), int(train_dates[1].split('-')[0]) + 1)
    # print("creating test time vector ",int(test_dates[0].split('-')[0]), int(test_dates[1].split('-')[0]) + 1)
    for i in range(int(train_dates[0].split('-')[0]), int(train_dates[1].split('-')[0]) + 1):
        if calendar.isleap(i):
            if not time_vector:
                time_vector = [i for i in range(366)]
            else:
                time_vector.extend([i for i in range(366)])
        else:
            if not time_vector:
                time_vector = [i for i in range(365)]
            else:
                time_vector.extend([i for i in range(365)])
    time_vector = np.array(time_vector)
    test_time_vector = []
    for i in range(int(test_dates[0].split('-')[0]), int(test_dates[1].split('-')[0]) + 1):
        if calendar.isleap(i):
            if not test_time_vector:
                test_time_vector = [i for i in range(366)]
            else:
                test_time_vector.extend([i for i in range(366)])
        else:
            if not test_time_vector:
                test_time_vector = [i for i in range(365)]
            else:
                test_time_vector.extend([i for i in range(365)])
    test_time_vector = np.array(test_time_vector)
        
    df_train = df[(df['Date']>=pd.Timestamp(train_dates[0])) & (df['Date']<=pd.Timestamp(train_dates[1]))]
    df_test = df[(df['Date']>=pd.Timestamp(test_dates[0])) & (df['Date']<=pd.Timestamp(test_dates[1]))]
    df_train = df_train[['AllMort', 'DayOfWeekNumeric']]
    df_test = df_test[['AllMort', 'DayOfWeekNumeric']]
    # print("here",len(df_test))
    ns_train = pt.bs(time_vector, df=4*date_difference_years)
    ns_test = pt.bs(test_time_vector, df=4*date_difference_years)

    dow = pt.bs(df_train['DayOfWeekNumeric'].values, df=7)
    dow_test = pt.bs(df_test['DayOfWeekNumeric'].values, df=7)
    # print("len of dowtest, nstest = ", len(dow_test), len(ns_test))
    y_train = df_train['AllMort'].to_numpy().reshape(-1,1)
    y_test = df_test['AllMort'].to_numpy().reshape(-1,1)
    
    x_train = np.concatenate((dow, ns_train), axis=1)
    x_test = np.concatenate((dow_test, ns_test), axis = 1)
    x_train = sm.add_constant(x_train)
    x_test = sm.add_constant(x_test)

    poisson_training_results = sm.GLM(y_train, x_train, family=sm.families.Poisson(link=sm.families.links.Log())).fit()
    pred = poisson_training_results.predict(x_train)
    rmse = mean_squared_error(pred, y_train)
    # print(" Training RMSE erros = ", np.sqrt(rmse))
    predicted_death = poisson_training_results.predict(x_test)

    # res=predicted_death
    # rmse_pred = mean_squared_error(res, y_test)
    # print("Predicted RMSE =", np.sqrt(rmse_pred))
    
    if return_train:
        return pred, predicted_death[-len(df.loc[df['Date']>=ranges['test'][0]]):]
    else:
        return predicted_death[-len(df.loc[df['Date']>=ranges['test'][0]]):]


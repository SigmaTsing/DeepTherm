import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
from itertools import product
import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from poisson import predict_poisson
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from torch.utils.tensorboard import SummaryWriter
import sys

def train_test_split(df, ranges):
    train = df[(df['Date']>=pd.to_datetime(ranges["train"][0])) & (df['Date']<=pd.to_datetime(ranges["train"][1]))]
    test = df[(df['Date']>=pd.to_datetime(ranges["test"][0])) & (df['Date']<=pd.to_datetime(ranges["test"][1]))]
    return train, test

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_length=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, input_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-np.log(10000.0) / input_dim))

        if input_dim % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        if input_dim % 2 == 1:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x

class HeatTransformer(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, mlp_dim = 128, num_layers=2, num_heads=1, dropout=0.25):
        super(HeatTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, \
            nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        
        self.linear = nn.Linear(in_features = hidden_dim, out_features = mlp_dim)
        self.out_proj = nn.Linear(in_features = mlp_dim, out_features = 5)

    def forward(self, x):
        # x = self.embedding(x)[0]
        x = F.gelu(self.embedding(x))
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.linear(F.gelu(torch.mean(x, dim = 1)))
        x = self.out_proj(F.gelu(x))
        return x

class heatDataset(Dataset):
    def __init__(self, raw, mins, maxs, input_len = 14, pred_len = 5):
        values = raw[:, 1:-1]
        mins = mins.reshape((1, -1))
        maxs = maxs.reshape((1, -1))
        values[:, 1:] = np.divide(values[:, 1:] - mins, maxs - mins)
        
        values[:, 1:] += np.random.normal(0.1, 0.1, size = values[:, 1:].shape)
        
        X = sliding_window_view(values, (input_len, values.shape[1])).squeeze()[:-pred_len]
        y = sliding_window_view(values[:, 0], pred_len)[input_len:]
        self.X = torch.Tensor(X.astype(np.float32))
        self.y = torch.Tensor(y.astype(np.float32))
        assert len(self.X) == len(self.y)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train_model(model, train_loader, val_loader = None, learning_rate = 1e-4, num_epochs = 1000, val_freq = 20, device = 'cuda:3', year = 1997):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    recent_val_loss = 0
    best_val_loss = 0
    writer = SummaryWriter('./log')
    for epoch in (bar := tqdm(range(num_epochs))):
        train_loss = 0    
        for x, y in train_loader:
            pred = model(x.to(device))
            loss = loss_fn(pred, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        writer.add_scalar('Train', train_loss, global_step= epoch)
        
        bar.set_description('train: {:.4f}'.format(train_loss))
        bar.update()
        
    torch.save(model.state_dict(), f'./checkpoints/heat_{year}_{device}.dict')
    
def is_heatwave(sccs):
    return (np.min(sccs) >= 30 and np.max(sccs) < 40) or (len(set(sccs).intersection([61, 62]))>0 and len(set(sccs).intersection([30, 31, 32]))>0)
    
def rate_to_level(rate, thres = [.3, .15, .02]):
    if rate > thres[0]:
        return 3
    elif rate > thres[1]:
        return 2
    elif rate > thres[2]:
        return 1
    else:
        return 0
    
def mark_heatwave(data):
    heatwave_labels = [False, False]
    for i in range(2, len(data) - 2):
        heatwave_labels.append(is_heatwave(data['SSC'].iloc[i-2:i+1]) or is_heatwave(data['SSC'].iloc[i:i+3]) or is_heatwave(data['SSC'].iloc[i-1:i+2]))
    heatwave_labels.extend([False, False])
    data['is_heatwave'] = heatwave_labels
    return data
    
def preprocess_city(domain = 'MAD'):
    raw = pd.read_csv(f'./data/external/Meteo{domain}.csv', sep = ';', header = 2).drop(columns = ['ExpectedDeaths(MoMo)']).drop([0])
    print(raw)
    if '#Date' in raw.keys():
        raw = raw.rename(columns = {'#Date': 'Date'})
    raw = raw.drop_duplicates(subset = ['Date'])
    raw['Date'] = pd.to_datetime(raw['Date'], format = '%d-%b-%Y')
    raw = raw.sort_values(by = 'Date')
    start_date = raw.iloc[0]['Date']
    date_list = raw['Date'].tolist()
    
    for key in raw.keys()[1:]:
        mean = np.mean(raw.loc[raw[key]!=-99, key].to_numpy())
        raw.loc[raw[key]==-99, key] = mean
    
    to_append = []
    while start_date <= raw.iloc[-1]['Date']:
        start_date += pd.DateOffset(1)
        if start_date not in date_list:
            # to_append.append(start_date)
            pol = raw[(raw['Date']>=start_date-pd.DateOffset(7))&(raw['Date']<=start_date+pd.DateOffset(7))]
            filled = {}
            filled['Date'] = start_date
            for key in raw.keys()[1:]:
                filled[key] = np.mean(pol[key].to_numpy())
            to_append.append(filled)
    # print(to_append)
    raw = pd.concat((raw, pd.DataFrame(to_append))).sort_values(by = 'Date')
    
    ssc = pd.read_csv(f'./data/SSC_codes/{domain}.cal3', sep = ' ', names = ['area', 'Date', 'SSC']).drop(columns = ['area'])
    ssc['Date'] = pd.to_datetime(ssc['Date'], format = '%Y%m%d')
    
    merged = pd.merge(raw, ssc, on = 'Date', how = 'left').reindex(columns = ['Date', 'Deaths(MoMo)', 'SSC', 'TempMean', 'PresMean', 'WindSpeedMean', 'HumidityMean']).rename(columns={'Deaths(MoMo)': 'AllMort'})
    data = mark_heatwave(merged)
    
    for i in range(len(data)):
        if data.iloc[i]['Date'].month < 5 or data.iloc[i]['Date'].month > 9:
            data.is_heatwave.iloc[i] = False
    print(merged)
    data.to_csv(f'./data/{domain}_ine_preprocessed.csv', index = False)
    
if __name__ == '__main__':
    # domain = 'MAD'
    year_threshold = int(sys.argv[1])
    domain = sys.argv[2]
    suffix = sys.argv[3]
    device = sys.argv[4]
    print(domain, year_threshold, suffix, device)
    
    input_len = 14
    output_len = 5
    
    if not os.path.exists(f'./data/{domain}_preprocessed.csv'):
        preprocess_city(domain = domain)

    data = pd.read_csv(f'./data/{domain}_{suffix}_preprocessed.csv').drop(columns = ['SSC'])
    data['Date'] = pd.to_datetime(data['Date'])
    
    RANGES = {
        "train": [f"1995-01-01", f"{year_threshold - 1}-12-31"],
        "test": [f"{year_threshold}-01-01", f"{year_threshold}-12-31"]
    }
    
    raw_train, raw_test = train_test_split(data, RANGES)
    mins = np.min(raw_train.values[:, 2:-1], axis = 0)
    maxs = np.max(raw_train.values[:, 2:-1], axis = 0)
    train_loader = DataLoader(heatDataset(raw_train.values, mins, maxs, input_len, output_len), batch_size = 512)
    test_loader = DataLoader(heatDataset(raw_test.values, mins, maxs, input_len, output_len), batch_size = 512, shuffle= False)
    
    learning_rate = 1e-3
    num_epochs = 300 
    val_freq = 20
    
    model = HeatTransformer(input_dim = len(raw_train.keys()) - 2, hidden_dim= 32, mlp_dim= 32, num_layers= 2, num_heads= 2, dropout= 0).to(device)
    if os.path.exists(f'checkpoints/heat_{year_threshold - 1}_{device}.dict'):
        model.load_state_dict(torch.load(f'checkpoints/heat_{year_threshold - 1}_{device}.dict'))
    train_model(model, train_loader, None, learning_rate= learning_rate, num_epochs= num_epochs, val_freq = val_freq, device = device, year = year_threshold)

    all_preds = []
    for x, y in test_loader:
        all_preds.append(model(x.to(device)).detach().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis = 0)

    # print(len(all_preds), len(raw_train), len(raw_test))
    
    test_dates = raw_test['Date'].iloc[input_len:].to_list()
    test_grounds = raw_test['AllMort'].iloc[input_len:].to_list()

    # train_dataset = heatDataset(raw_train.values, mins, maxs)
    # train_X, train_y = train_dataset.X, train_dataset.y
    # train_X = train_X.reshape((train_X.shape[0], -1))
    
    # test_dataset = heatDataset(raw_test.values, mins, maxs)
    # test_X, test_y = test_dataset.X, test_dataset.y
    # test_X = test_X.reshape((test_X.shape[0], -1))
    
    # model = xgb.XGBRegressor(objective='reg:squarederror',
    #     n_estimators=1000,
    #     learning_rate=0.01,
    #     max_depth=6,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     early_stopping_rounds=50,
    # )
    
    # model.fit(train_X, train_y, eval_set = [(train_X, train_y)], verbose = False)
    # all_preds = model.predict(test_X)
    
    preds = np.zeros((len(raw_test) - input_len, output_len))
    for i in range(output_len):
        preds[i:preds.shape[0]-(output_len - 1)+i, i] = all_preds[:, i]
    preds = np.max(preds, axis = 1)
    
    poisson_ranges = {
        "train": [f"{year_threshold - 2}-01-01", f"{year_threshold - 1}-12-31"],
        "test": [f"{year_threshold}-01-01", f"{year_threshold}-12-31"]
    }
    poisson_pred = predict_poisson(data, poisson_ranges)[input_len:]
    
    heat_dates = raw_test.loc[raw_test['is_heatwave'], 'Date'].to_list()
    heat_marks = raw_test['is_heatwave'].iloc[input_len:].to_list()
    
    heat_start_dates = []
    heat_spans = []
    consec = False
    first_date = None
    for i in range(len(test_dates)):
        if not heat_marks[i] and consec:
            consec = False
            heat_spans.append((test_dates[i] - first_date).days)
        elif heat_marks[i] and not consec:
            consec = True
            heat_start_dates.append(test_dates[i])
            first_date = test_dates[i]
    
    assert len(heat_start_dates) == len(heat_spans)
    
    poisson_wave_labels = []
    pred_levels = []
    poisson_rates = []
    pred_rates= []
    baseline_levels_with_zero = []
    ground_levels = []
    for i in range(len(heat_start_dates)):
        ground_rates = []
        pred_rates = []
        baseline_levels = []
        for j in range(heat_spans[i]):
            date = heat_start_dates[i] + pd.Timedelta(days = j)
            test_ind = test_dates.index(date)
            ground_death = test_grounds[test_ind]
            poisson_death = poisson_pred[test_ind]
            pred_death = preds[test_ind]
            pred_rates.append((pred_death - poisson_death) / poisson_death)
            ground_rates.append((ground_death - poisson_death) / poisson_death)
            # baseline_levels.append(baseline_results.loc[baseline_results['Date'] == date, 'LEVEL'].item())
        ground_level = rate_to_level(np.max(ground_rates))
        # pred_level = rate_to_level(np.max(pred_rates), [.02, .015, .001])
        pred_level = rate_to_level(np.max(pred_rates))
        # baseline_level = np.max(baseline_levels)
        
        print('Date {} Range {} Poisson: Rate {:.3f} Level {} ## Baseline: Level 0 ## Pred: Rate {:.3f} Level {}'.format(heat_start_dates[i], heat_spans[i], np.max(ground_rates), ground_level, np.max(pred_rates), pred_level))
        poisson_wave_labels.append(ground_level)
        pred_levels.append(pred_level)
        poisson_rates.append(np.max(ground_rates))
        pred_rates.append(np.max(pred_rates))
        ground_levels.append(ground_level)
    
    
    
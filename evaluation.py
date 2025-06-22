import re
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np

def rate_to_level(rate, thres = [.3, .15, .02]):
    if rate > thres[0]:
        return 3
    elif rate > thres[1]:
        return 2
    elif rate > thres[2]:
        return 1
    else:
        return 0

    
ptn = re.compile('.*?Poisson:.*?Rate (-?\d+\.\d+).*?Baseline:.*?Pred:.*?Rate (-?\d+\.\d+).*')
city_name_to_code = {'MADRID': 'MAD', 'BILBAO': 'BIO', 'BARCELONA': 'BCN', 'MÁLAGA': 'AGP', 'ALICANTE': 'ALC', 'BADAJOZ': 'BJZ', 'CÓRDOBA': 'ODB', 'OURENSE': 'ORE', 'VALENCIA': 'VLC', 'SEVILLA': 'SVQ', 'TOLEDO': 'TOL', 'ZARAGOZA': 'ZAZ'}
codes = [i for i in city_name_to_code.values()]

scale = 20
grounds = []
preds = []
for code in codes:
# for code in ['BJZ']:
    # with open(f'./baseline_outputs/ine/{code}.log', 'r') as f:
    with open(f'./outputs/{code}.ine.log', 'r') as f:
        log = f.readlines()
    for line in log:
        res = ptn.findall(line)
        if len(res)>0:
            g, p = res[0]
            g = rate_to_level(float(g))
            p = rate_to_level(float(p)*scale)
            grounds.append(g)
            preds.append(p)

gs = [int(int(i) / 2) for i in grounds]
ps = [int(int(i) / 2) for i in preds]
print(confusion_matrix(gs, ps))
print(accuracy_score(gs, ps), precision_score(gs, ps), recall_score(gs, ps), f1_score(gs, ps))
gs = [int(int(i) / 3) for i in grounds]
ps = [int(int(i) / 3) for i in preds]
print(confusion_matrix(gs, ps))
print(accuracy_score(gs, ps), precision_score(gs, ps), recall_score(gs, ps), f1_score(gs, ps))

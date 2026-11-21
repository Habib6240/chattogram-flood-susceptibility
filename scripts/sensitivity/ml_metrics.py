import numpy as np, csv
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
EV=['2019','2021','2023','2026']
L=np.load('/home/claude/release/chattogram-flood-susceptibility-data-v1.0.0/06_oof_predictions/BUET_B07U_LOEO_OutOfFold_Predictions.npz')
FOLD=L['Spatial_Fold'].astype(int)
def loeo_metrics(pred):  # pred: dict event->array (nan where not predicted)
    out={}
    for e in EV:
        obs=L['Observed_'+e].astype(int);p=pred[e];v=obs>=0
        pooled=(AUC(obs[v],p[v]),AP(obs[v],p[v]))
        fa=[];fp=[]
        for f in range(1,6):
            k=v&(FOLD==f);fa.append(AUC(obs[k],p[k]));fp.append(AP(obs[k],p[k]))
        out[e]=dict(pooled_auc=pooled[0],pooled_ap=pooled[1],fold_auc=fa,fold_ap=fp)
    out['macro_pooled_auc']=np.mean([out[e]['pooled_auc'] for e in EV]);out['macro_pooled_ap']=np.mean([out[e]['pooled_ap'] for e in EV])
    out['macro_foldmean_auc']=np.mean([np.mean(out[e]['fold_auc']) for e in EV]);out['macro_foldmean_ap']=np.mean([np.mean(out[e]['fold_ap']) for e in EV])
    return out
if __name__=='__main__':
    for k,name in [('RF','RF'),('XGB','XGB')]:
        m=loeo_metrics({e:L[f'{k}_{e}'] for e in EV});print('archived',name,{a:round(b,6) for a,b in m.items() if a.startswith('macro')})

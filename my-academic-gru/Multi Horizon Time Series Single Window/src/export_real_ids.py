"""Recover real OULAD id_student for each row of X_sequences.npy and the test split,
and add it to model8_predictions.csv (written as model8_predictions_with_ids.csv)."""
import numpy as np, pandas as pd, os
from sklearn.model_selection import train_test_split
RAW = os.path.expanduser('~/mnt/raw/') if os.path.exists(os.path.expanduser('~/mnt/raw/')) else '../data/raw/'
N_WEEKS=17; SEED=42
si=pd.read_csv(RAW+'studentInfo.csv'); sa=pd.read_csv(RAW+'studentAssessment.csv'); a=pd.read_csv(RAW+'assessments.csv')
ass=sa.merge(a[['id_assessment','date','assessment_type','weight']],on='id_assessment',how='left').dropna(subset=['date'])
ass['week']=(ass['date']/7).astype(int)+1; ass=ass[ass['week']<=N_WEEKS]
aw=ass.groupby(['id_student','week'],as_index=False).agg(n=('id_assessment','count'))
valid=aw[aw['n']>0].groupby('id_student')['week'].nunique(); valid=valid[valid>=2].index
st=si[si['id_student'].isin(valid)][['id_student','code_module','code_presentation','final_result']].drop_duplicates()
cnt=st.groupby('id_student').size()
full=np.sort(cnt[cnt==1].index.values)
st=st.set_index('id_student').loc[full].reset_index()
st['label']=st['final_result'].apply(lambda x:0 if x in['Pass','Distinction'] else 1)
y=np.load('../data/processed/y_labels.npy')
print('rebuilt',len(full),'saved',len(y),'labels match:',(st['label'].values==y).all() if len(full)==len(y) else 'n/a')
idx=np.arange(len(y))
_,te=train_test_split(idx,test_size=0.15,random_state=SEED,stratify=y)
p=pd.read_csv('../results/metrics/model8_predictions.csv')
print('test',len(te),'preds',len(p),'actual_label match:',(p['actual_label'].values==y[te].astype(int)).all())
t=st.iloc[te].reset_index(drop=True)
p.insert(2,'id_student',t['id_student'].values); p.insert(3,'code_module',t['code_module'].values); p.insert(4,'code_presentation',t['code_presentation'].values)
p.to_csv('../results/metrics/model8_predictions_with_ids.csv',index=False); print('written')

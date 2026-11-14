import json
from pathlib import Path
import pandas as pd
from src.pipeline import ROOT,recommend

def test_validation_report():
    r=json.loads((ROOT/'reports'/'validation_report.json').read_text())
    assert r['quarantined_records']>=7
    assert r['valid_records']+r['quarantined_records']==r['total_records']

def test_time_split():
    a=pd.read_csv(ROOT/'data'/'processed'/'train.csv',parse_dates=['timestamp'])
    b=pd.read_csv(ROOT/'data'/'processed'/'test.csv',parse_dates=['timestamp'])
    assert a.groupby('user_id').timestamp.max().lt(b.groupby('user_id').timestamp.min()).all()

def test_recommendations_exclude_seen():
    train=pd.read_csv(ROOT/'data'/'processed'/'train.csv')
    for model in ['popularity','item_cf']:
        recs=recommend(1,model,5)
        assert len(recs)==5 and len(set(recs))==5
        assert not set(recs)&set(train.loc[train.user_id==1,'item_id'])

def test_metrics():
    df=pd.read_csv(ROOT/'reports'/'model_comparison.csv')
    assert set(df.model)=={'popularity','item_cf'}
    assert df[['precision_at_5','recall_at_5','ndcg_at_5','coverage','diversity']].apply(lambda x:x.between(0,1).all()).all()

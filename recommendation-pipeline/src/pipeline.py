"""Reproducible recommendation data pipeline. Run from repository root."""
from __future__ import annotations
import argparse, hashlib, json, logging, math, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models'
SEED = 529
K = 5
STAGES = ['ingestion','validation','preparation','transformation','features','training','evaluation']
for d in [DATA/'raw',DATA/'processed',DATA/'features',DATA/'errors',REPORTS,MODELS]: d.mkdir(parents=True,exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(REPORTS/'pipeline.log',mode='w')])
log=logging.getLogger('pipeline')

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def ingestion():
    rng=np.random.default_rng(SEED)
    genres=['Action','Comedy','Drama','Sci-Fi','Documentary','Romance']
    users=pd.DataFrame({'user_id':np.arange(1,101),'age':rng.integers(18,66,100),'region':rng.choice(['North','South','East','West'],100)})
    items=pd.DataFrame({'item_id':np.arange(1,81),'title':[f'Movie {i:03d}' for i in range(1,81)],'genre':[genres[(i-1)%6] for i in range(1,81)],'release_year':rng.integers(1990,2025,80)})
    rows=[]; start=pd.Timestamp('2025-01-01',tz='UTC')
    # Each user sees a sequence of unique items; preferences correlate with genre.
    for uid in users.user_id:
        preferred=int(uid%6)
        weights=np.array([4.0 if (i-1)%6==preferred else 1.0 for i in items.item_id],dtype=float)
        weights/=weights.sum()
        chosen=rng.choice(items.item_id.to_numpy(),size=36,replace=False,p=weights)
        times=np.sort(rng.integers(0,250*24*3600,size=len(chosen)))
        for iid,sec in zip(chosen,times):
            affinity=1.5 if (int(iid)-1)%6==preferred else 0
            rating=int(np.clip(np.rint(2.8+affinity+rng.normal(0,0.95)),1,5))
            rows.append((int(uid),int(iid),rating,(start+pd.Timedelta(seconds=int(sec))).isoformat()))
    interactions=pd.DataFrame(rows,columns=['user_id','item_id','rating','timestamp'])
    # Deliberately inject representative quality defects to exercise the quarantine layer.
    bad=pd.DataFrame([
        [1,1,9,'2025-02-01T00:00:00+00:00'],
        [1,999,4,'2025-02-02T00:00:00+00:00'],
        [999,1,4,'2025-02-03T00:00:00+00:00'],
        [1,2,4,'not-a-date'],
        [1,3,None,'2025-02-05T00:00:00+00:00'],
        ['bad',4,3,'2025-02-06T00:00:00+00:00'],
        list(interactions.iloc[0])
    ],columns=interactions.columns)
    interactions=pd.concat([interactions,bad],ignore_index=True)
    for name,df in [('users',users),('items',items),('interactions',interactions)]:df.to_csv(DATA/'raw'/f'{name}.csv',index=False)
    log.info('Ingested users=%d items=%d interactions=%d',len(users),len(items),len(interactions))

def validation():
    u=pd.read_csv(DATA/'raw/users.csv');i=pd.read_csv(DATA/'raw/items.csv');x=pd.read_csv(DATA/'raw/interactions.csv',dtype={'user_id':'string','item_id':'string','rating':'string','timestamp':'string'})
    reasons=[]; seen=set()
    for idx,row in x.iterrows():
        faults=[]
        for field in ['user_id','item_id','rating','timestamp']:
            if pd.isna(row[field]) or str(row[field]).strip()=='':faults.append(f'missing_{field}')
        uid=pd.to_numeric(row.user_id,errors='coerce');iid=pd.to_numeric(row.item_id,errors='coerce');rating=pd.to_numeric(row.rating,errors='coerce')
        if pd.isna(uid) or not float(uid).is_integer():faults.append('invalid_user_id_type')
        elif int(uid) not in set(u.user_id):faults.append('unknown_user')
        if pd.isna(iid) or not float(iid).is_integer():faults.append('invalid_item_id_type')
        elif int(iid) not in set(i.item_id):faults.append('unknown_item')
        if pd.isna(rating) or not float(rating).is_integer() or not 1<=rating<=5:faults.append('invalid_rating')
        timestamp=pd.to_datetime(row.timestamp,errors='coerce',utc=True)
        if pd.isna(timestamp):faults.append('invalid_timestamp')
        signature=tuple('' if pd.isna(v) else str(v) for v in row)
        if signature in seen:faults.append('duplicate_record')
        seen.add(signature)
        reasons.append(';'.join(dict.fromkeys(faults)))
    x['error_reason']=reasons
    bad=x[x.error_reason!=''].copy();good=x[x.error_reason==''].drop(columns='error_reason').copy()
    good['user_id']=pd.to_numeric(good.user_id).astype(int);good['item_id']=pd.to_numeric(good.item_id).astype(int);good['rating']=pd.to_numeric(good.rating).astype(int)
    good['timestamp']=pd.to_datetime(good.timestamp,utc=True)
    bad.to_csv(DATA/'errors'/'invalid_interactions.csv',index=False)
    good.to_csv(DATA/'processed'/'validated_interactions.csv',index=False)
    summary={'total_records':len(x),'valid_records':len(good),'quarantined_records':len(bad),'error_counts':{k:int(sum(k in r.split(';') for r in reasons)) for k in sorted(set(';'.join(reasons).split(';'))-{''})},'user_duplicate_ids':int(u.user_id.duplicated().sum()),'item_duplicate_ids':int(i.item_id.duplicated().sum()),'user_missing_cells':int(u.isna().sum().sum()),'item_missing_cells':int(i.isna().sum().sum())}
    save_json(REPORTS/'validation_report.json',summary)
    if summary['user_duplicate_ids'] or summary['item_duplicate_ids'] or summary['user_missing_cells'] or summary['item_missing_cells']:raise ValueError('Invalid user/item dimension table')
    log.info('Validated %d rows; quarantined %d',len(good),len(bad))

def preparation():
    x=pd.read_csv(DATA/'processed'/'validated_interactions.csv',parse_dates=['timestamp']).sort_values(['user_id','timestamp','item_id'])
    # Per-user chronological 80/20 split, to evaluate future preferences.
    x['position']=x.groupby('user_id').cumcount();x['n']=x.groupby('user_id').user_id.transform('size')
    train=x[x.position < np.floor(x.n*.8)].drop(columns=['position','n'])
    test=x[x.position >= np.floor(x.n*.8)].drop(columns=['position','n'])
    assert len(train)>0 and len(test)>0
    assert train.groupby('user_id').timestamp.max().lt(test.groupby('user_id').timestamp.min()).all()
    train.to_csv(DATA/'processed'/'train.csv',index=False);test.to_csv(DATA/'processed'/'test.csv',index=False)
    save_json(REPORTS/'split_metadata.json',{'strategy':'per-user chronological 80/20','train_rows':len(train),'test_rows':len(test),'seed':SEED,'cutoff_rule':'floor(0.8 * interactions per user)'})
    log.info('Split train=%d test=%d',len(train),len(test))

def transformation():
    x=pd.read_csv(DATA/'processed'/'train.csv')
    users=x.groupby('user_id').agg(activity_count=('rating','size'),average_rating=('rating','mean'),last_interaction=('timestamp','max')).reset_index()
    items=x.groupby('item_id').agg(popularity=('rating','size'),average_rating=('rating','mean')).reset_index()
    mat=x.pivot_table(index='user_id',columns='item_id',values='rating',fill_value=0)
    users.to_csv(DATA/'processed'/'user_activity.csv',index=False);items.to_csv(DATA/'processed'/'item_popularity.csv',index=False)
    mat.to_csv(DATA/'processed'/'interaction_matrix.csv')
    log.info('Transformed summaries and interaction matrix')

def features():
    train=pd.read_csv(DATA/'processed'/'train.csv',parse_dates=['timestamp']);items=pd.read_csv(DATA/'raw/items.csv')
    anchor=train.timestamp.max()
    user=train.groupby('user_id').agg(user_activity_count=('rating','size'),user_average_rating=('rating','mean'),last_timestamp=('timestamp','max')).reset_index()
    user['days_since_last_interaction']=(anchor-user.last_timestamp).dt.total_seconds()/86400
    fav=train.merge(items[['item_id','genre']],on='item_id').groupby(['user_id','genre']).size().reset_index(name='count').sort_values(['user_id','count','genre'],ascending=[True,False,True]).drop_duplicates('user_id')
    user=user.merge(fav[['user_id','genre']].rename(columns={'genre':'preferred_genre'}),on='user_id').drop(columns='last_timestamp')
    item=train.groupby('item_id').agg(item_popularity=('rating','size'),item_average_rating=('rating','mean')).reset_index()
    item=items.merge(item,on='item_id',how='left').fillna({'item_popularity':0,'item_average_rating':0})
    user.to_csv(DATA/'features'/'user_features.csv',index=False);item.to_csv(DATA/'features'/'item_features.csv',index=False)
    # Reusable for prediction: both training and serving load the same persisted feature tables.
    log.info('Created reusable user and item feature tables')

def train_models():
    train=pd.read_csv(DATA/'processed'/'train.csv');items=pd.read_csv(DATA/'raw/items.csv')
    matrix=train.pivot_table(index='user_id',columns='item_id',values='rating',fill_value=0)
    # User-item binary interaction similarities; item-item CF via cosine similarity.
    binary=(matrix.to_numpy()>0).astype(float)
    similarities=cosine_similarity(binary.T)
    np.fill_diagonal(similarities,0)
    np.savez_compressed(MODELS/'item_cf.npz',similarities=similarities,user_ids=matrix.index.to_numpy(),item_ids=matrix.columns.to_numpy(),ratings=matrix.to_numpy())
    popularity=train.groupby('item_id').agg(count=('rating','size'),mean_rating=('rating','mean'))
    popularity['score']=popularity['count']*(popularity['mean_rating']/5)
    popularity.sort_values('score',ascending=False).to_csv(MODELS/'popularity.csv')
    save_json(MODELS/'model_config.json',{'baseline':'weighted popularity: count * mean_rating/5','personalized':'item-item cosine similarity on binary training interactions, scored by similarity-weighted observed rating','k':K,'seed':SEED,'number_of_items':len(items)})
    log.info('Trained popularity baseline and item-item collaborative filtering')

def recommend(user_id,model,k=K):
    train=pd.read_csv(DATA/'processed'/'train.csv')
    items=pd.read_csv(DATA/'raw/items.csv')
    seen=set(train.loc[train.user_id==user_id,'item_id'])
    candidates=set(items.item_id)-seen
    if model=='popularity':
        popularity=pd.read_csv(MODELS/'popularity.csv');scores=dict(zip(popularity.item_id,popularity.score))
    elif model=='item_cf':
        with np.load(MODELS/'item_cf.npz') as d:
            ids=d['item_ids'];uids=d['user_ids'];mat=d['ratings'];sim=d['similarities']
            if user_id not in uids:return recommend(user_id,'popularity',k)
            vec=mat[np.where(uids==user_id)[0][0]];positive=vec>0
            # Penalize unobserved items only through lack of evidence, not as negative labels.
            scores_arr=(sim @ (vec*positive))/(sim @ positive.astype(float)+1e-9)
            scores=dict(zip(ids.astype(int),scores_arr))
    else:raise ValueError('Unknown model')
    return sorted(candidates,key=lambda item:(-scores.get(item,0),item))[:k]

def evaluation():
    test=pd.read_csv(DATA/'processed'/'test.csv');items=pd.read_csv(DATA/'raw/items.csv')
    genres=dict(zip(items.item_id,items.genre));eligible=sorted(test.user_id.unique())
    results=[];pred_rows=[]
    for model in ['popularity','item_cf']:
        metrics=[];recommended=set();diversities=[]
        for uid in eligible:
            actual=set(test.loc[(test.user_id==uid)&(test.rating>=4),'item_id'])
            if not actual:continue
            recs=recommend(uid,model,K);hits=np.array([int(r in actual) for r in recs])
            p=hits.sum()/K;r=hits.sum()/len(actual)
            dcg=sum(hit/math.log2(j+2) for j,hit in enumerate(hits))
            idcg=sum(1/math.log2(j+2) for j in range(min(len(actual),K)))
            metrics.append((p,r,dcg/idcg if idcg else 0))
            recommended.update(recs)
            if len(recs)>1:
                pairs=[int(genres[a]!=genres[b]) for idx,a in enumerate(recs) for b in recs[idx+1:]]
                diversities.append(float(np.mean(pairs)))
            pred_rows.extend({'model':model,'user_id':int(uid),'rank':j+1,'item_id':int(item),'relevant':int(item in actual)} for j,item in enumerate(recs))
        arr=np.array(metrics)
        results.append({'model':model,'precision_at_5':float(arr[:,0].mean()),'recall_at_5':float(arr[:,1].mean()),'ndcg_at_5':float(arr[:,2].mean()),'coverage':len(recommended)/len(items),'diversity':float(np.mean(diversities)),'evaluated_users':len(metrics)})
    df=pd.DataFrame(results);df.to_csv(REPORTS/'model_comparison.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(REPORTS/'recommendations.csv',index=False)
    save_json(REPORTS/'evaluation_results.json',results)
    # A simple file-based experiment tracker: parameters, metrics, artifacts, run metadata.
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for result in results:
        run={'run_id':run_id,'model':result['model'],'parameters':{'k':K,'seed':SEED,'split':'per-user chronological 80/20'},'metrics':{k:v for k,v in result.items() if k not in ['model']},'artifacts':['reports/model_comparison.csv','reports/recommendations.csv',('models/popularity.csv' if result['model']=='popularity' else 'models/item_cf.npz')]}
        save_json(REPORTS/f"experiment_{result['model']}.json",run)
    log.info('Evaluation:\n%s',df.to_string(index=False))

def workflow():
    records=[];status='success'
    try:
        for stage in STAGES:
            start=time.perf_counter();log.info('START %s',stage)
            try:
                globals()[stage if stage!='training' else 'train_models']()
                records.append({'stage':stage,'status':'success','elapsed_seconds':round(time.perf_counter()-start,3)})
            except Exception as exc:
                records.append({'stage':stage,'status':'failed','error':str(exc)})
                status='failed';raise
    finally:
        save_json(REPORTS/'workflow_run.json',{'status':status,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'stages':records})
    log.info('WORKFLOW SUCCESS')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=STAGES+['all'],default='all');args=parser.parse_args()
    if args.stage=='all':workflow()
    else:globals()[args.stage if args.stage!='training' else 'train_models']()

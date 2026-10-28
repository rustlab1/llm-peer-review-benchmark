from pathlib import Path
import argparse,json,csv,re,statistics,sys,hashlib
import numpy as np
from scipy.stats import spearmanr
from docx import Document

BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Check the current manuscript datasets and optional supplementary table.')
parser.add_argument('--supplement',type=Path,help='Path to Supplementary_information.docx for the Table 2 check')
args=parser.parse_args()
# When invoked from the preparation workspace, point at the assembled release.
if not (BASE/'data/manifest.json').exists():
    raise SystemExit('Copy data/, data_original/ and additional_judges/ from Supplementary Data 1 beside code/ before running this validator.')
DEST=BASE/'validation';DEST.mkdir(exist_ok=True)
ITEMS=[f'M{i}' for i in range(1,10)]+[f'm{i}' for i in range(1,6)]
errors=[];report={'study':'Current 15-model manuscript benchmark','validation':'Archived model scores and manuscript consistency; independent human scoring was not performed.'}
def read(p):return json.loads(p.read_text())
def scores(p):return {x['id']:float(x['score']) for x in read(p)['scores']}
records=[]
for tag,folder,mf,judge,rev in [('2004','data_original','manifest.json','judgements','reviews'),('2026_seeded','data','manifest.json','judgements','reviews'),('2026_controls','data','control_manifest.json','controls/judgements','controls/reviews')]:
    m=read(BASE/folder/mf)
    for entry in m:
        rid=entry['id'];js=[]
        for j in ['J1','J2']:
            p=BASE/folder/judge/f'{rid}__{j}.json'
            if not p.exists():errors.append(f'Missing {p.relative_to(BASE)}');continue
            s=scores(p)
            if set(s)!=set(ITEMS):errors.append(f'Incomplete key {p.name}')
            if not all(v in [0,.5,1] for v in s.values()):errors.append(f'Invalid score {p.name}')
            js.append(s)
        if len(js)!=2:continue
        # Experiment controls have their review text under controls/reviews.
        rp=BASE/folder/rev/f'{rid}.md'
        if not rp.exists():errors.append(f'Missing review {rp.relative_to(BASE)}')
        for it in ITEMS:
            a,b=js[0][it],js[1][it];records.append({'dataset':tag,'review_id':rid,'reviewer_model':entry['model'],'paper':entry['paper'],'error_id':it,'Fable_5':a,'Opus_5':b,'primary_mean_score':(a+b)/2,'primary_detected':int((a+b)/2>=.5),'either_judge_detected':int(max(a,b)>=.5)})
report['reviews_by_dataset']={t:len(set(r['review_id'] for r in records if r['dataset']==t)) for t in ['2004','2026_seeded','2026_controls']}
if report['reviews_by_dataset']!={'2004':180,'2026_seeded':180,'2026_controls':45}:errors.append('Unexpected review coverage')
with (DEST/'review_item_scores.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
models=sorted(set(r['reviewer_model'] for r in records));rr={}
v2=read(BASE/'data/results_v2.json')
for tag in ['2004','2026_seeded','2026_controls']:
    rr[tag]={}
    for m in models:
        rs=[r for r in records if r['dataset']==tag and r['reviewer_model']==m]
        tot={rid:sum(r['primary_mean_score'] for r in rs if r['review_id']==rid) for rid in set(r['review_id'] for r in rs)}
        rr[tag][m]={'mean_score':float(np.mean(list(tot.values()))),'primary_detection_percent':100*float(np.mean([r['primary_detected'] for r in rs])),'either_detection_percent':100*float(np.mean([r['either_judge_detected'] for r in rs])),'lowest_review_score':min(tot.values())}
        exp='exp1' if tag=='2004' else 'exp2';expected=v2[exp][m]
        key='spec_all' if tag=='2026_controls' else 'sens_all'
        obs=100-rr[tag][m]['primary_detection_percent'] if tag=='2026_controls' else rr[tag][m]['primary_detection_percent']
        if abs(round(obs,1)-expected[key])>.051:errors.append(f'Metric mismatch {tag}/{m}/{key}')
        if tag!='2026_controls' and abs(round(rr[tag][m]['mean_score'],2)-expected['total_mean'])>.011:errors.append(f'Mean mismatch {tag}/{m}')
report['metrics']=rr
report['control_flags_primary']=sum(r['primary_detected'] for r in records if r['dataset']=='2026_controls')
report['control_flags_either_judge']=sum(r['either_judge_detected'] for r in records if r['dataset']=='2026_controls')
if (report['control_flags_primary'],report['control_flags_either_judge'])!=(107,145):errors.append('Control counts differ from manuscript')
for rule in ['primary','either']:
    a=[rr['2026_seeded'][m][rule+'_detection_percent'] for m in models];b=[100-rr['2026_controls'][m][rule+'_detection_percent'] for m in models]
    rho,p=spearmanr(a,b);report[rule+'_sensitivity_specificity']={'rho':float(rho),'p':float(p)}
nohaw=[]
for m in models:
    sr=[r for r in records if r['dataset']=='2026_seeded' and r['reviewer_model']==m and r['error_id']!='m5']
    cr=[r for r in records if r['dataset']=='2026_controls' and r['reviewer_model']==m and r['error_id']!='m5']
    nohaw.append([100*np.mean([r['primary_detected'] for r in sr]),100*(1-np.mean([r['primary_detected'] for r in cr]))])
rho,p=spearmanr(np.array(nohaw)[:,0],np.array(nohaw)[:,1]);report['without_hawthorne']={'rho':float(rho),'p':float(p)}
report['all_seeded_reviews_above_historical_human_mean']=all(rr[tag][m]['lowest_review_score']>3.29 for tag in ['2004','2026_seeded'] for m in models)
kj=BASE/'additional_judges/kimi_k3'
for name,n in [('kimik3_main',405),('kimik3_reworded',90)]:
    files=list((kj/name/'judgements').glob('*__J3.json'));report[name+'_reviews']=len(files)
    if len(files)!=n:errors.append('Incomplete '+name)
    if any(set(scores(p))!=set(ITEMS) for p in files):errors.append('Incomplete items '+name)
report['kimi_dataset_max_absolute_mean_score_difference']={}
for tag in ['2004','2026_seeded','2026_controls']:
    dif=[]
    for m in models:
        ids=set(r['review_id'] for r in records if r['dataset']==tag and r['reviewer_model']==m)
        ks=[sum(scores(kj/'kimik3_main/judgements'/f'{rid}__J3.json').values()) for rid in ids]
        dif.append(float(np.mean(ks))-rr[tag][m]['mean_score'])
    report['kimi_dataset_max_absolute_mean_score_difference'][tag]=max(abs(x) for x in dif)
pred=BASE/'additional_judges/decision_comparison/predictions'
for name in ['gpt61sol_j5','jev_normal','decision1_normal']:
    ps=[p for p in (pred/name).glob('case_*__r1.json')]
    report[name+'_reviews']=len(ps)
    if len(ps)!=405:errors.append('Incomplete '+name)
# Export the three submission tables as CSV, then compare every numerical row in Table 2.
tables=BASE/'tables';tables.mkdir(exist_ok=True)
outroot=BASE.parents[1]
supp=args.supplement or outroot/'02_supplementary/Supplementary_information.docx'
if args.supplement and not supp.exists():
    raise SystemExit(f'Supplementary document not found: {supp}')
if supp.exists():
    d=Document(supp)
    for i,t in enumerate(d.tables,1):
        with (tables/f'Supplementary_Table_{i}.csv').open('w',newline='') as f:csv.writer(f).writerows([[c.text for c in r.cells] for r in t.rows])
    LABEL={v['model']:m for m,v in v2['exp1'].items()}
    exp=None;checked=0
    for row in d.tables[1].rows[2:]:
        ss=[c.text for c in row.cells]
        if ss[0] and ss[0] not in ('2004 test papers','2026 trials'):break  # 58-trial block: see code/further_trials/
        if ss[0]:exp='exp1' if ss[0]=='2004 test papers' else 'exp2'
        label=ss[1];m=LABEL[label];e=v2[exp][m]
        for col,key,ci,dp in [(2,'major_mean','major_ci_clustered',2),(3,'minor_mean','minor_ci_clustered',2),(4,'sens_all','sens_all_ci',1),(5,'spec_all','spec_all_ci',1)]:
            if col==5 and exp=='exp1':continue
            vals=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',ss[col])]
            want=[e[key],*e[ci]]
            if len(vals)!=3 or any(abs(a-b)>(.011 if dp==2 else .051) for a,b in zip(vals,want)):errors.append(f'Supplementary Table 2 mismatch {exp}/{label}/{key}: {vals} vs {want}')
            checked+=1
    report['supplementary_table_2_numeric_cells_checked']=checked
report['errors']=errors;report['status']='PASS' if not errors else 'FAIL'
(DEST/'release_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='metrics'},indent=2))
if errors:sys.exit(1)

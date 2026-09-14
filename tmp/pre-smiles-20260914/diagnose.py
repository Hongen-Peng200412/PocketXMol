"""一次验收诊断: 全量原模型离散键类别核对与 bf16 旧链重复误差."""
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from docking.smiles import read_smiles_graph
from legacy_assets import read_template

ROOT=Path('/storage/penghongen/AdaLigand/Ori_Data')
OUT=Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/diagnose-v1')


def main():
    OUT.mkdir(exist_ok=False)
    cache={};failures=[];count=0;max_center=0.
    for line in Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/cpu-v1/mapping.jsonl').open():
        r=json.loads(line);count+=1
        key=(r['prepared_smiles'],r['object_key'])
        if key not in cache:
            graph=read_smiles_graph(ROOT/'smiles_assets',r['prepared_smiles'])
            atoms,bonds,_,_=read_template(ROOT/'ligand_objects'/(r['object_key'].replace(':','_')+'.npz'))
            p=np.asarray(r['new_to_old']);n=len(p)
            old=np.zeros((n,n),np.int64)
            kinds=np.asarray([1,2,3,0,4])[bonds['type'].argmax(1)]
            old[bonds['atom_1'],bonds['atom_2']]=kinds
            old+=old.T
            new=np.zeros((n,n),np.int64)
            new[graph['bond_index'][0],graph['bond_index'][1]]=graph['bond_type'];new+=new.T
            differing=np.argwhere(np.triu(old[p][:,p]!=new,1))
            cache[key]=dict(differing_bonds=len(differing), old_new_types=dict(Counter(f'{old[p[i],p[j]]}->{new[i,j]}' for i,j in differing)), element_equal=bool(np.array_equal(atoms['element'][p],graph['element'])))
        info=cache[key]
        if info['differing_bonds'] or not info['element_equal']:
            failures.append(dict(pdb_id=r['pdb_id'],candidate_id=r['candidate_id'],split=r['split'],object_key=r['object_key'],prepared_smiles=r['prepared_smiles'],**info))
        max_center=max(max_center,r['center_max_abs_A'])
    report=dict(checked=count,failed=len(failures),failure_fraction=len(failures)/count,pairs=len(cache),failed_pairs=sum(bool(x['differing_bonds']) or not x['element_equal'] for x in cache.values()),max_center_abs_A=max_center,split_failures=dict(Counter(r['split'] for r in failures)),failures=failures)
    (OUT/'raw_model_graph_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='failures'}),flush=True)


if __name__=='__main__':main()

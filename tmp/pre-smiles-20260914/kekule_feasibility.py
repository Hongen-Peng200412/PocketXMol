"""一次只读资产诊断: SMILES 独立 Kekulé 键编码能否与旧原始模型图同构."""
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit import Chem
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from docking.smiles import read_smiles_graph
from legacy_assets import read_template,BOND_TYPES


def main():
    root=Path('/storage/penghongen/AdaLigand/Ori_Data')
    report_root=Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914')
    cache={};failures=[];count=0;old_aromatic_count=0
    for line in (report_root/'cpu-v1/mapping.jsonl').open():
        r=json.loads(line);key=(r['prepared_smiles'],r['object_key']);count+=1
        if key not in cache:
            graph=read_smiles_graph(root/'smiles_assets',r['prepared_smiles'])
            atoms,bonds,old,_=read_template(root/'ligand_objects'/(r['object_key'].replace(':','_')+'.npz'))
            kinds=bonds['type'].argmax(1)
            old_raw=Chem.RWMol()
            for properties in atoms:
                atom=Chem.Atom(int(properties['element']));atom.SetFormalCharge(int(properties['charge']));old_raw.AddAtom(atom)
            for bond,kind in zip(bonds,kinds):old_raw.AddBond(int(bond['atom_1']),int(bond['atom_2']),BOND_TYPES[int(kind)])
            old_raw=old_raw.GetMol();old_raw.UpdatePropertyCache(strict=False);Chem.GetSymmSSSR(old_raw)
            new=Chem.Mol(graph['mol']);Chem.Kekulize(new,clearAromaticFlags=True)
            mapping=old_raw.GetSubstructMatch(new,useChirality=False)
            cache[key]=dict(compatible=bool(mapping),old_has_aromatic=bool(np.any(kinds==4)),new_to_old=list(mapping))
        result=cache[key]
        old_aromatic_count+=result['old_has_aromatic']
        if not result['compatible']:failures.append(dict(pdb_id=r['pdb_id'],candidate_id=r['candidate_id'],split=r['split'],object_key=r['object_key'],prepared_smiles=r['prepared_smiles'],old_has_aromatic=result['old_has_aromatic']))
    report=dict(checked=count,failed=len(failures),failure_fraction=len(failures)/count,failed_pairs=sum(not x['compatible'] for x in cache.values()),old_aromatic_occurrences=old_aromatic_count,failures=failures,split_failures=dict(Counter(x['split'] for x in failures)))
    (report_root/'kekule_feasibility.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='failures'}),flush=True)


if __name__=='__main__':main()

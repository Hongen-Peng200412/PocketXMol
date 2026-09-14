"""一次任务: 用非测试实例比较旧模板和公共 SMILES 输入的真实 GPU 运算."""
import importlib.util
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from easydict import EasyDict
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from docking.dataset import OccurrenceDataset
from models.loss import get_loss_func
from models.maskfill import PMAsymDenoiser
from models.sample import sample_loop3, get_cfd_traj
from utils.misc import make_config, seed_all
from utils.sample_noise import get_sample_noiser
from utils.transforms import FeaturizeMol, get_transforms
from legacy_dataset import OccurrenceDataset as OldDataset

OUT = Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/gpu-v1')
MAPPING = Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/cpu-v1/mapping.jsonl')
OLD_ROOT = Path('/storage/penghongen/PocketXMol/data')
LIMITS = {'fp32': dict(atol=1e-4, rtol=1e-4, gradient_relative_l2=1e-3), 'bf16': dict(atol=.02, rtol=.02, gradient_relative_l2=.02)}


def differences(left, right):
    a, b = left.detach().float(), right.detach().float()
    return dict(max_abs=float((a-b).abs().max()) if a.numel() else 0., rms=float((a-b).square().mean().sqrt()) if a.numel() else 0., relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)))


def compare_model(model, loss_fn, old_batch, new_batch, atom_map, edge_map, precision):
    initial = deepcopy(model.state_dict())
    results = []
    for batch in [old_batch, old_batch, new_batch]:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.99,.999), weight_decay=.001, eps=1e-8)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
            outputs = model(batch.clone())
            losses = loss_fn(batch, outputs)
            loss = losses['mixed/total']
        loss.backward()
        gradients = {name: p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        results.append(dict(outputs={k:v.detach().cpu() for k,v in outputs.items()}, losses={k:float(v) for k,v in losses.items()}, gradients=gradients, parameters={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
    old, repeat, new = results
    output_errors = {}
    repeat_errors = {}
    failures = []
    for key, value in old['outputs'].items():
        indices = edge_map if 'halfedge' in key else atom_map
        aligned = value[indices]
        output_errors[key] = differences(aligned, new['outputs'][key])
        repeat_errors[key] = differences(value, repeat['outputs'][key])
        if not torch.allclose(aligned.float(), new['outputs'][key].float(), atol=LIMITS[precision]['atol'], rtol=LIMITS[precision]['rtol']):
            failures.append('output:'+key)
    gradient_delta = gradient_norm = 0.
    update_delta = update_norm = update_max = 0.
    for name, value in old['gradients'].items():
        gradient_delta += float((value-new['gradients'][name]).double().square().sum())
        gradient_norm += float(value.double().square().sum())
    for name, value in old['parameters'].items():
        d = (value-new['parameters'][name]).double()
        update_delta += float(d.square().sum())
        update_norm += float(value.double().square().sum())
        update_max = max(update_max, float(d.abs().max()))
    gradient_relative = (gradient_delta/max(gradient_norm,1e-30))**.5
    if gradient_relative > LIMITS[precision]['gradient_relative_l2']:
        failures.append('gradient_relative_l2')
    model.load_state_dict(initial)
    return dict(precision=precision, outputs=output_errors, old_repeat=repeat_errors, losses_old=old['losses'], losses_new=new['losses'], gradient_relative_l2=gradient_relative, adamw_parameter_relative_l2=(update_delta/max(update_norm,1e-30))**.5, adamw_parameter_max_abs=update_max, failures=failures)


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    (OUT/'limits.json').write_text(json.dumps(LIMITS,indent=2))
    started=time.time()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    config=make_config(str(PROJECT/'configs/docking/B-C-T0-RA.yml'))
    config.data.dataset.manifest_root=str(OLD_ROOT/'smiles-v1/frozen')
    config.data.dataset.smiles_root='/storage/penghongen/AdaLigand/Ori_Data/smiles_assets'
    config.data.dataset.smiles_coords_root='/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/SMILE_coords/v1'
    featurizer=FeaturizeMol(config.transforms.featurizer)
    transform=Compose([featurizer,get_transforms(EasyDict(name='dock',settings={'free':1},free_no_geometry=True),mode='test')])
    model=PMAsymDenoiser(config.model,featurizer.num_node_types,featurizer.num_edge_types,pocket_in_dim=25).cuda().eval()
    checkpoint=torch.load('/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/checkpoints/step=21600.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict({k[6:]:v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')},strict=True)
    del checkpoint
    loss_fn=get_loss_func(config.loss).cuda()
    noiser=get_sample_noiser(config.noise,featurizer.num_node_types,featurizer.num_edge_types,mode='train').noiser_dict['dock']
    noiser.sample_level=lambda step,batch:{'pos':torch.full((batch.num_nodes,),.4,device=batch.node_pos.device)}
    mappings={}
    for line in MAPPING.open():
        record=json.loads(line)
        if record['split']!='test':
            mappings[(record['split'],record['pdb_id'],record['candidate_id'])]=record
    # 按规模分层取12个不同SMILES, 同时包含三个非测试划分; 不根据模型误差挑选.
    chosen=[];seen=set()
    all_values=list(mappings.values())
    for split in ['validation','calibration','train']:
        unique={r['prepared_smiles']:r for r in all_values if r['split']==split}
        values=sorted(unique.values(),key=lambda r:(len(r['new_to_old']),r['pdb_id'],r['candidate_id']))
        for index in np.linspace(0,len(values)-1,4,dtype=int):
            record=values[index]
            if record['prepared_smiles'] not in seen:
                chosen.append(record);seen.add(record['prepared_smiles'])
    (OUT/'samples.json').write_text(json.dumps(chosen,indent=2))
    results=[]
    for selected in chosen:
        for protocol in ['C0','C5','E']:
            cfg=deepcopy(config.data.dataset);cfg.pocket_mode='envelope' if protocol=='E' else 'center'
            new_set=OccurrenceDataset(cfg,selected['split'],transform,'RA',protocol,False)
            old_cfg=deepcopy(cfg);old_cfg.manifest_root=str(OLD_ROOT)
            old_set=OldDataset(old_cfg,selected['split'],transform,'RA',protocol,False)
            index=next(i for i,r in enumerate(new_set.records) if r['pdb_id']==selected['pdb_id'] and r['candidate_id']==selected['candidate_id'])
            old_data=old_set[index];new_data=new_set[index]
            p=torch.tensor(selected['new_to_old'],dtype=torch.long)
            matrix=torch.zeros((len(p),len(p)),dtype=torch.long)
            ei=old_data.halfedge_index
            matrix[ei[0],ei[1]]=torch.arange(ei.shape[1]);matrix[ei[1],ei[0]]=torch.arange(ei.shape[1])
            q=matrix[p[new_data.halfedge_index[0]],p[new_data.halfedge_index[1]]]
            assert torch.equal(old_data.node_type[p],new_data.node_type)
            assert torch.equal(old_data.halfedge_type[q],new_data.halfedge_type)
            center_error=differences(old_data.pocket_center,new_data.pocket_center)
            torch.testing.assert_close(old_data.pocket_pos,new_data.pocket_pos,atol=1e-4,rtol=1e-5)
            torch.testing.assert_close(old_data.node_pos[p],new_data.node_pos,atol=1e-4,rtol=1e-5)
            assert torch.equal(old_data.pocket_atom_feature,new_data.pocket_atom_feature)
            assert torch.equal(old_data.pocket_nucleic_feature,new_data.pocket_nucleic_feature)
            assert torch.equal(old_data.pocket_knn_edge_index,new_data.pocket_knn_edge_index)
            seed_all(7801)
            epsilon=torch.randn_like(old_data.node_pos)
            with patch('torch.randn_like',lambda value,**kw:epsilon.to(value)):
                old_data=noiser(old_data)
            with patch('torch.randn_like',lambda value,**kw:epsilon[p].to(value)):
                new_data=noiser(new_data)
            torch.testing.assert_close(old_data.pos_in[p],new_data.pos_in,atol=1e-4,rtol=1e-5)
            batches=[Batch.from_data_list([d],follow_batch=featurizer.follow_batch+['pocket_pos'],exclude_keys=featurizer.exclude_keys+transform.transforms[-1].exclude_keys).cuda() for d in [old_data,new_data]]
            for precision in ['fp32','bf16']:
                report=compare_model(model,loss_fn,*batches,p,q,precision)
                report.update(pdb_id=selected['pdb_id'],candidate_id=selected['candidate_id'],split=selected['split'],protocol=protocol,atoms=len(p),center_error=center_error)
                results.append(report)
                (OUT/'gpu_report.json').write_text(json.dumps(dict(results=results,elapsed_seconds=time.time()-started),indent=2))
                print(json.dumps({k:v for k,v in report.items() if k not in ['outputs','old_repeat','losses_old','losses_new']}),flush=True)
    print('GPU_FORWARD_DONE',len(results),flush=True)


if __name__=='__main__':
    main()

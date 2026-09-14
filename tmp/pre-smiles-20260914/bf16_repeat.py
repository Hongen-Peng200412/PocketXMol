"""一次诊断: 固定同一非测试输入, 比较原生 bf16 重复误差与只在诊断中升精度归约."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
from easydict import EasyDict
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from legacy_dataset import OccurrenceDataset as OldDataset
from docking.dataset import OccurrenceDataset
from models.loss import get_loss_func
from models.maskfill import PMAsymDenoiser
import models.graph_context as graph_context
from utils.misc import make_config,seed_all
from utils.sample_noise import get_sample_noiser
from utils.transforms import FeaturizeMol,get_transforms

OUT=Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/bf16-repeat-v1')


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    cfg=make_config('configs/docking/B-C-T0-RA.yml')
    cfg.data.dataset.manifest_root='/storage/penghongen/PocketXMol/data'
    feat=FeaturizeMol(cfg.transforms.featurizer)
    task=get_transforms(EasyDict(name='dock',settings={'free':1},free_no_geometry=True),mode='test')
    ds=OldDataset(cfg.data.dataset,'validation',Compose([feat,task]),'RA','C0',False)
    index=next(i for i,r in enumerate(ds.records) if r['pdb_id']=='6d03' and r['candidate_id']==18)
    old=ds[index]
    new_cfg=deepcopy(cfg.data.dataset)
    new_cfg.update(manifest_root='/storage/penghongen/PocketXMol/data/smiles-v1/frozen',smiles_root='/storage/penghongen/AdaLigand/Ori_Data/smiles_assets',smiles_coords_root='/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/SMILE_coords/v1')
    new=OccurrenceDataset(new_cfg,'validation',Compose([feat,task]),'RA','C0',False)[index]
    maps=json.loads(Path('/storage/penghongen/tmp/pxm_pre_smiles_20260914/gpu-v1/samples.json').read_text())
    p=torch.tensor(next(r['new_to_old'] for r in maps if r['pdb_id']=='6d03'))
    ei=old.halfedge_index;n=len(p)
    matrix=torch.zeros((n,n),dtype=torch.long);matrix[ei[0],ei[1]]=torch.arange(ei.shape[1]);matrix[ei[1],ei[0]]=torch.arange(ei.shape[1])
    q=matrix[p[new.halfedge_index[0]],p[new.halfedge_index[1]]]
    # 诊断固定同一 C 和物理噪声, 排除质心归约和随机噪声作为差异来源.
    new.node_pos=old.node_pos[p].clone();new.gt_node_pos=old.gt_node_pos[p].clone();new.pocket_pos=old.pocket_pos.clone();new.pocket_center=old.pocket_center.clone()
    noise=get_sample_noiser(cfg.noise,feat.num_node_types,feat.num_edge_types,mode='train').noiser_dict['dock']
    noise.sample_level=lambda step,batch:{'pos':torch.full((batch.num_nodes,),.4)}
    seed_all(7801);eps=torch.randn_like(old.node_pos)
    with patch('torch.randn_like',lambda x,**kw:eps):old=noise(old)
    with patch('torch.randn_like',lambda x,**kw:eps[p]):new=noise(new)
    batches=[Batch.from_data_list([d],follow_batch=feat.follow_batch+['pocket_pos'],exclude_keys=feat.exclude_keys+task.exclude_keys).cuda() for d in [old,new]]
    model=PMAsymDenoiser(cfg.model,feat.num_node_types,feat.num_edge_types,pocket_in_dim=25).cuda().eval()
    ckpt=torch.load('/storage/penghongen/PocketXMol/training/B-C-T0-RA-C0/checkpoints/step=21600.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict({k[6:]:v for k,v in ckpt['state_dict'].items() if k.startswith('model.')},strict=True);del ckpt
    loss_fn=get_loss_func(cfg.loss).cuda()
    native=graph_context.scatter_sum
    def fp32_sum(src,index,dim=-1,out=None,dim_size=None):
        return native(src.float(),index,dim=dim,out=out,dim_size=dim_size).to(src.dtype) if src.dtype==torch.bfloat16 else native(src,index,dim=dim,out=out,dim_size=dim_size)
    report=[]
    for mode in ['fp32','bf16_native','bf16_fp32_scatter_diagnostic_only']:
        captured=[]
        for batch in [batches[0],batches[0],batches[0],batches[1]]:
            model.zero_grad(set_to_none=True)
            with patch.object(graph_context,'scatter_sum',fp32_sum if 'diagnostic' in mode else native),torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode!='fp32'):
                outputs=model(batch.clone());losses=loss_fn(batch,outputs)
            losses['mixed/total'].backward()
            captured.append(dict(outputs={k:v.detach().float().cpu() for k,v in outputs.items()},losses={k:float(v) for k,v in losses.items()},grad=torch.cat([v.grad.detach().float().flatten().cpu() for v in model.parameters() if v.grad is not None])))
        base=captured[0]
        for number,other in enumerate(captured[1:],1):
            gradient_delta=(base['grad']-other['grad']).norm()
            item=dict(mode=mode,comparison='old_new_aligned' if number==3 else f'old_repeat_{number}',gradient_abs_l2=float(gradient_delta),gradient_reference_l2=float(base['grad'].norm()),gradient_relative_l2=float(gradient_delta/base['grad'].norm()),losses_base=base['losses'],losses_other=other['losses'],outputs={})
            for key,value in base['outputs'].items():
                left=value[q if 'halfedge' in key else p] if number==3 else value
                diff=left-other['outputs'][key]
                item['outputs'][key]=dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/left.norm().clamp_min(1e-12)),rms=float(diff.square().mean().sqrt()))
            report.append(item)
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()

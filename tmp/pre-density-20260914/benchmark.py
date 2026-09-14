"""一次任务的真实训练性能探测；不保存正式模型或W&B，不读取test。"""
import argparse,json,os,resource,signal,subprocess,threading,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.loss import get_loss_func
from docking.density import read_density_source
import docking.dataset as density_dataset
from utils.misc import make_config


class PilotDataset(Dataset):
 def __init__(self,source,indices,density):
  self.source,self.indices,self.density=source,indices,density
 def __len__(self): return len(self.indices)*1000
 def __getitem__(self,cursor):
  original_density_loader=density_dataset.load_density_input
  density_elapsed=0.
  def measured_density_loader(*args,**kwargs):
   nonlocal density_elapsed
   begin=time.perf_counter()
   result=original_density_loader(*args,**kwargs)
   density_elapsed+=time.perf_counter()-begin
   return result
  density_dataset.load_density_input=measured_density_loader
  index=self.indices[cursor%len(self.indices)]
  begin=time.perf_counter()
  try: data=self.source[index]
  finally: density_dataset.load_density_input=original_density_loader
  original_seconds=time.perf_counter()-begin
  data['pilot_source_seconds']=torch.tensor([original_seconds-density_elapsed])
  data['pilot_density_seconds']=torch.tensor([density_elapsed])
  if self.density:
   pdb_id=self.source.records[index]['pdb_id']
   _,_,spacing,origin,receptor=read_density_source(str(self.source.root),pdb_id)
   corner=origin+data.density_start_zyx.numpy().reshape(3)[::-1].astype(np.float32)*spacing
   local=(receptor-corner)/spacing
   occupied=int(((local>=0)&(local<48)).all(1).sum())
   if occupied==0: raise RuntimeError(f'{pdb_id}:真实裁块完整受体mask为空，保留成熟数值并先报告科学边界')
   data['pilot_receptor_atoms']=torch.tensor([occupied])
  return data


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--config',required=True);p.add_argument('--output',required=True)
 p.add_argument('--batch',type=int,default=24);p.add_argument('--workers',type=int,default=8)
 p.add_argument('--backend',default='flash');p.add_argument('--distance',type=int,default=1)
 p.add_argument('--checkpoint',type=int,default=1);p.add_argument('--seconds',type=int,default=180)
 p.add_argument('--updates',type=int,default=30);p.add_argument('--samples',type=int,default=0)
 p.add_argument('--prefetch',type=int,default=1);p.add_argument('--channels-last',type=int,default=0)
 p.add_argument('--readout-reference',type=int,default=0)
 p.add_argument('--sample-seed',type=int,default=2023)
 p.add_argument('--deadline',type=float,required=True)
 args=p.parse_args()
 output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
 report={'scope':'preexperiment_not_formal','arguments':vars(args),'started_at':time.time(),'status':'running','updates':[],'metrics':[]}
 def timed_out(signum,frame): raise TimeoutError('预实验观察或整体预算到期')
 signal.signal(signal.SIGALRM,timed_out)
 signal.alarm(max(1,int(args.deadline-time.time())))
 def save(): (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 save()
 os.environ['OMP_NUM_THREADS']='1';torch.set_num_threads(1)
 torch.manual_seed(2023);np.random.seed(2023)
 torch.backends.cudnn.benchmark=True
 stop=threading.Event()
 process=psutil.Process()
 def monitor():
  while not stop.is_set():
   try:
    children=process.children(recursive=True)
    rss=sum(x.memory_info().rss for x in [process,*children] if x.is_running())
    cpu=sum(sum(x.cpu_times()[:2]) for x in [process,*children] if x.is_running())
    query=subprocess.check_output(['nvidia-smi','--id=GPU-adbf8fc8-5a4a-87e3-853b-c9cadcbdf74b','--query-gpu=utilization.gpu,utilization.memory,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    io=[x.io_counters() for x in [process,*children] if x.is_running()]
    report['metrics'].append(dict(wall=time.time(),rss_bytes=rss,cpu_seconds=cpu,gpu=query,read_bytes=sum(x.read_bytes for x in io),read_chars=sum(x.read_chars for x in io),write_bytes=sum(x.write_bytes for x in io)))
   except (psutil.Error,subprocess.SubprocessError): pass
   stop.wait(1)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  report['memory_available_bytes']=psutil.virtual_memory().available
  report['cpu_affinity_count']=len(os.sched_getaffinity(0))
  report['cudnn_benchmark']=torch.backends.cudnn.benchmark
  report['cudnn_allow_tf32']=torch.backends.cudnn.allow_tf32
  report['float32_matmul_precision']=torch.get_float32_matmul_precision()
  memory_limit=Path('/sys/fs/cgroup/memory/slurm/uid_1351/job_379402/memory.limit_in_bytes')
  if memory_limit.exists(): report['job_memory_limit_bytes']=int(memory_limit.read_text())
  config=make_config(args.config)
  density='density' in config.model
  if density: config.model.density.update(attention_backend=args.backend,distance_bias=bool(args.distance),checkpoint=bool(args.checkpoint))
  config.train.update(batch_size=args.batch,num_workers=0,persistent_workers=False)
  module=DataModule(config);module.setup('fit')
  source=module.train_loader.dataset
  if args.samples:
   indices=np.linspace(0,len(source.records)-1,args.samples,dtype=int).tolist()
   report['sampling_scope']='固定子集循环，仅计算及缓存热态基线'
  else:
   sample_count=args.updates*72+args.workers*args.batch*max(args.prefetch,1)+72
   indices=np.random.default_rng(args.sample_seed).integers(0,len(source.records),size=sample_count).tolist()
   report['sampling_scope']='完整训练清单按实例均匀有放回，固定随机种子；跨配置共用相同前缀'
  report['sample_ids']=[(source.records[i]['pdb_id'],source.records[i]['candidate_id']) for i in indices]
  dataset=PilotDataset(source,indices,density)
  loader=DataLoader(dataset,batch_size=args.batch,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,follow_batch=module.train_loader.follow_batch,exclude_keys=module.train_loader.exclude_keys,**({'prefetch_factor':args.prefetch} if args.workers else {}))
  model=PMAsymDenoiser(config.model,**module.get_in_dims())
  if args.readout_reference:
   from reference_density_readout import DensityReadout as ReferenceReadout
   with torch.random.fork_rng(devices=[]):
    for index,reader in enumerate(model.denoiser.density_readers):
     reference=ReferenceReadout(reader.output.out_features,config.model.density)
     reference.load_state_dict(reader.state_dict())
     model.denoiser.density_readers[index]=reference
  official=torch.load(config.train.initial_checkpoint,map_location='cpu',weights_only=False)
  state={k[6:]:v for k,v in official['state_dict'].items() if k.startswith('model.')}
  missing=model.load_state_dict(state,strict=False)
  invalid=[k for k in missing.missing_keys if not k.startswith(('nucleic_embedder.','density_encoder.','denoiser.density_readers.'))]
  assert not invalid and not missing.unexpected_keys,(invalid,missing.unexpected_keys)
  report['official_model_tensors']=len(state)
  del state,official
  model=model.cuda().train()
  if args.channels_last and density: model.density_encoder.to(memory_format=torch.channels_last_3d)
  loss_func=get_loss_func(config.loss)
  opt=config.train.optimizer
  optimizer=torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay,betas=(opt.beta1,opt.beta2),eps=opt.eps)
  accumulation=72//args.batch
  assert args.batch*accumulation==72
  report.update(mode=config.model.get('density',{}).get('mode','D0'),global_batch=72,accumulation=accumulation,torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in model.parameters()),file_descriptor_limits=resource.getrlimit(resource.RLIMIT_NOFILE))
  iterator=iter(loader)
  first_forward=None
  torch.cuda.reset_peak_memory_stats()
  for update in range(args.updates):
   if time.time()>=args.deadline or (first_forward is not None and time.perf_counter()-first_forward>=args.seconds): break
   times=dict(update=update,wall_start=time.time(),io_wait_seconds=0.,transfer_seconds=0.,forward_seconds=0.,backward_seconds=0.,optimizer_seconds=0.,source_cpu_seconds=0.,density_cpu_seconds=0.,atoms=0,pocket_atoms=0,loss=0.)
   update_start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
   for micro in range(accumulation):
    before=time.perf_counter();batch=next(iterator);times['io_wait_seconds']+=time.perf_counter()-before
    times['source_cpu_seconds']+=float(batch.pilot_source_seconds.sum());times['density_cpu_seconds']+=float(batch.pilot_density_seconds.sum())
    times['atoms']+=len(batch.node_type);times['pocket_atoms']+=len(batch.pocket_pos)
    before=time.perf_counter();batch=batch.to('cuda',non_blocking=True)
    if args.channels_last and density: batch.density_input=batch.density_input.contiguous(memory_format=torch.channels_last_3d)
    torch.cuda.synchronize();times['transfer_seconds']+=time.perf_counter()-before
    before=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
     outputs=model(batch);loss=loss_func(batch,outputs)['mixed/total']
    torch.cuda.synchronize();times['forward_seconds']+=time.perf_counter()-before
    if first_forward is None:
     first_forward=time.perf_counter();report['first_forward_wall']=time.time();print('FIRST_REAL_FORWARD',json.dumps(dict(mode=report['mode'],loss=float(loss),batch=batch.num_graphs)),flush=True);save()
     signal.alarm(max(1,min(300,int(args.deadline-time.time()))))
    assert torch.isfinite(loss),float(loss)
    times['loss']+=float(loss)/accumulation
    before=time.perf_counter();(loss/accumulation).backward();torch.cuda.synchronize();times['backward_seconds']+=time.perf_counter()-before
    del outputs,loss,batch
   # 每次完整更新核查总体梯度；仍保留原训练clip规则。
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.get('gradient_clip_val') or float('inf'),error_if_nonfinite=True)
   if density and update==0:
    report['density_gradients']={}
    for prefix in ('density_encoder.','denoiser.density_readers.'):
     inactive=[name for name,param in model.named_parameters() if name.startswith(prefix) and name.endswith('.beta') and not args.distance]
     selected=[(name,param) for name,param in model.named_parameters() if name.startswith(prefix) and name not in inactive]
     missing=[name for name,param in selected if param.grad is None]
     assert not missing,missing
     report['density_gradients'][prefix]={'active_tensors':len(selected),'inactive_parameters':inactive,'l2_norm':float(torch.stack([param.grad.float().square().sum() for _,param in selected]).sum().sqrt()),'all_finite':all(bool(torch.isfinite(param.grad).all()) for _,param in selected)}
   before=time.perf_counter();optimizer.step();torch.cuda.synchronize();times['optimizer_seconds']=time.perf_counter()-before
   times['total_seconds']=time.perf_counter()-update_start;times['grad_norm']=float(norm)
   times['wall_end']=time.time()
   times['peak_allocated_bytes']=torch.cuda.max_memory_allocated();times['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
   report['updates'].append(times);print('UPDATE',json.dumps(times),flush=True);save()
  assert report['updates'],'no_complete_update'
  report['parameters_finite']=all(bool(torch.isfinite(p).all()) for p in model.parameters())
  assert report['parameters_finite']
  report['status']='complete'
 except BaseException as error:
  report['status']='failed';report['error']=repr(error);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
 finally:
  stop.set();thread.join(timeout=3);report['finished_at']=time.time();save()
 if report['status']!='complete': raise SystemExit(1)


if __name__=='__main__': main()

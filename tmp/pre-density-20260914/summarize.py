"""汇总固定真实实例基准；冷首更新和后续更新分开，原始JSON保持不变。"""
import json,statistics
from pathlib import Path
snapshot=sorted(Path(__file__).parent.glob('evidence*.json'))[-1]
for path,data in json.loads(snapshot.read_text()).items():
 updates=data['updates'];warm=updates[1:]
 if not warm: print(path,data['status'],data.get('error'));continue
 total=sum(x['total_seconds'] for x in warm)
 metrics=data['metrics'];begin=warm[0].get('wall_start',data['first_forward_wall']+10)
 end=warm[-1].get('wall_end',data.get('finished_at',float('inf')))
 active=[m for m in metrics if begin<=m['wall']<=end]
 gpu=[int(m['gpu'].split(',')[0]) for m in active]
 cpu=(active[-1]['cpu_seconds']-active[0]['cpu_seconds'])/(active[-1]['wall']-active[0]['wall']) if len(active)>1 else None
 result=dict(name=Path(path).parent.name,status=data['status'],updates=len(updates),cold_seconds=updates[0]['total_seconds'],warm_mean=total/len(warm),warm_median=statistics.median(x['total_seconds'] for x in warm),compute_mean=statistics.mean(sum(x[k] for k in ('forward_seconds','backward_seconds','optimizer_seconds')) for x in warm),wait_percent=100*sum(x['io_wait_seconds'] for x in warm)/total,gpu_mean=statistics.mean(gpu) if gpu else None,cpu_cores=cpu,rss_gb=max(m['rss_bytes'] for m in metrics)/1e9,allocated_gb=max(x['peak_allocated_bytes'] for x in updates)/1e9)
 result['consumed_prefix_sha256']=data.get('consumed_sample_prefix_sha256')
 tail=warm[len(warm)//2:]
 tail_total=sum(x['total_seconds'] for x in tail)
 result['latter_half_mean']=tail_total/len(tail)
 result['latter_half_wait_percent']=100*sum(x['io_wait_seconds'] for x in tail)/tail_total
 if len(active)>1 and 'read_bytes' in active[0]:
  result['window_read_bytes']=active[-1]['read_bytes']-active[0]['read_bytes']
  result['window_read_chars']=active[-1]['read_chars']-active[0]['read_chars']
 print(json.dumps(result,ensure_ascii=False))

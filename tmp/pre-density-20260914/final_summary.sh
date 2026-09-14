#!/usr/bin/env bash
# 只读汇总原始JSON，计数器窗口包含同进程预取；不清缓存或启动GPU。
set -euo pipefail
'/storage/penghongen/PocketXMol/runtime/venv/bin/python' - <<'PY'
import glob,hashlib,json,statistics
result={}
for path in sorted(glob.glob('/storage/penghongen/tmp/pocketxmol_density_20260914/benchmarks/*/result.json')):
 raw=open(path,'rb').read();data=json.loads(raw);updates=data['updates'];warm=updates[1:]
 if int(path.split('/')[-2].split('_')[0])<9:continue
 item=dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),status=data['status'],arguments=data['arguments'],updates=len(updates),error=data.get('error'),parameters_finite=data.get('parameters_finite'),density_gradients=data.get('density_gradients'),release_project_root=data.get('release_project_root'),slurm_job_id=data.get('slurm_job_id'))
 ids=data.get('sample_ids',[])[:len(updates)*72]
 item['consumed_sample_prefix_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 item['metrics_count']=len(data['metrics'])
 if not warm:result[path]=item;continue
 metrics=data['metrics'];total=sum(x['total_seconds'] for x in warm)
 item['prepare_seconds']=updates[0]['wall_start']-data['started_at']
 item['process_to_first_update_seconds']=updates[0]['wall_end']-data['started_at']
 def window(begin,end):
  rows=[m for m in metrics if begin<=m['wall']<=end]
  if len(rows)<2:return {}
  first,last=rows[0],rows[-1];seconds=last['wall']-first['wall']
  return dict(samples=len(rows),seconds=seconds,read_bytes=last['read_bytes']-first['read_bytes'],read_chars=last['read_chars']-first['read_chars'],cpu_cores=(last['cpu_seconds']-first['cpu_seconds'])/seconds,gpu_mean=statistics.mean(int(m['gpu'].split(',')[0]) for m in rows))
 tail=warm[len(warm)//2:];tail_total=sum(x['total_seconds'] for x in tail)
 item.update(cold_seconds=updates[0]['total_seconds'],cold_io_wait_seconds=updates[0]['io_wait_seconds'],cold_density_worker_wall_seconds=updates[0]['density_cpu_seconds'],cold_graph_worker_wall_seconds=updates[0]['source_cpu_seconds'],cold_window=window(updates[0]['wall_start'],updates[0]['wall_end']),warm_mean=total/len(warm),warm_median=statistics.median(x['total_seconds'] for x in warm),warm_compute_mean=statistics.mean(sum(x[k] for k in ('forward_seconds','backward_seconds','optimizer_seconds')) for x in warm),warm_io_percent=100*sum(x['io_wait_seconds'] for x in warm)/total,warm_density_worker_wall_mean=statistics.mean(x['density_cpu_seconds'] for x in warm),warm_graph_worker_wall_mean=statistics.mean(x['source_cpu_seconds'] for x in warm),warm_window=window(warm[0]['wall_start'],warm[-1]['wall_end']),latter_half_mean=tail_total/len(tail),latter_half_io_percent=100*sum(x['io_wait_seconds'] for x in tail)/tail_total,peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in updates),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in updates),peak_rss_bytes=max(m['rss_bytes'] for m in metrics),madvise_verified_samples=sum(x.get('madvise_verified_samples',0) for x in updates))
 result[path]=item
print(json.dumps(result,ensure_ascii=False))
PY

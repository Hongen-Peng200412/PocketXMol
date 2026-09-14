"""批量读出对照修改前逐分子实现，检查所有输入和参数梯度。"""
import json
import torch
from models.density_readout import DensityReadout
from reference_density_readout import DensityReadout as Reference

torch.set_num_threads(1)
report=[]
for mode in ('D1','D4'):
 for biased in (False,True):
  for amp in (False,True):
   torch.manual_seed(91)
   actual_precision='medium' if amp else 'highest'
   torch.set_float32_matmul_precision(actual_precision)
   config=dict(mode=mode,distance_bias=biased,attention_backend='flash' if amp else 'sdpa')
   actual=DensityReadout(320,config).cuda()
   reference=Reference(320,config).cuda();reference.load_state_dict(actual.state_dict())
   side,channels=(6,256) if mode=='D1' else (48,48)
   feature=torch.randn(2,channels,side,side,side,device='cuda',requires_grad=True)
   hidden=torch.randn(5,320,device='cuda',requires_grad=True)
   positions=torch.tensor([[1.2,2.3,3.4],[-2.4,.1,.1],[-5.,.1,.1],[10.2,9.8,4.3],[51.1,3.,3.]],device='cuda',requires_grad=True)
   batch=torch.tensor([0,0,0,1,1],device='cuda');origin=torch.zeros(2,3,device='cuda');basis=torch.eye(3,device='cuda').repeat(2,1,1)
   reference.backend='sdpa'
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): original=reference(hidden,positions,batch,feature,origin,basis)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): observed=actual(hidden,positions,batch,feature,origin,basis)
   reference.backend='reference'
   torch.set_float32_matmul_precision('highest')
   with torch.autocast('cuda',enabled=False): expected=reference(hidden,positions,batch,feature,origin,basis)
   torch.testing.assert_close(observed,expected,rtol=.03 if amp else 1e-4,atol=.001 if amp else 2e-6)
   weight=torch.randn_like(expected)
   first=torch.autograd.grad((expected*weight).sum(),[hidden,positions,feature,*reference.parameters()],retain_graph=True,allow_unused=True)
   torch.set_float32_matmul_precision(actual_precision)
   original_grad=torch.autograd.grad((original*weight).sum(),[hidden,positions,feature,*reference.parameters()],retain_graph=True,allow_unused=True)
   second=torch.autograd.grad((observed*weight).sum(),[hidden,positions,feature,*actual.parameters()],allow_unused=True)
   errors=[];absolute_errors=[];zero_gradient_checks=[];gradients=[]
   for name,a,b,old in zip(['hidden','positions','feature',*[name for name,_ in actual.named_parameters()]],first,second,original_grad):
    if a is None: assert b is None;continue
    absolute=(a.float()-b.float()).square().mean().sqrt()
    scale=a.float().square().mean().sqrt()
    relative=absolute/scale.clamp_min(1e-8)
    gradients.append(dict(name=name,fp32_reference_rms=float(scale),new_difference_rms=float(absolute),original_difference_rms=float((a.float()-old.float()).square().mean().sqrt())))
    # key.bias对所有key加同一分数，数学上梯度为零；零附近采用绝对误差。
    tolerance=.04 if amp else .0002
    absolute_tolerance=1e-5 if amp else 1e-7
    if name=='key.bias':
     # BF16下两种归约顺序都会出现约1e-5伪梯度；分别检验它们接近理论零值。
     zero_tolerance=1e-4 if amp else 1e-7
     assert float(a.float().square().mean().sqrt())<zero_tolerance and float(b.float().square().mean().sqrt())<zero_tolerance,(name,mode,biased,amp,float(absolute),float(scale))
     zero_gradient_checks.append(dict(name=name,reference_rms=float(scale),batched_rms=float(b.float().square().mean().sqrt()),absolute_difference_rms=float(absolute),tolerance=zero_tolerance))
    else:
     assert absolute<=absolute_tolerance+tolerance*scale,(name,mode,biased,amp,float(absolute),float(scale))
    if name!='key.bias' and scale>1e-6: errors.append(float(relative))
    absolute_errors.append(float(absolute))
   alpha_index=3+[name for name,_ in actual.named_parameters()].index('alpha')
   report.append(dict(mode=mode,distance=biased,bf16=amp,reference_precision='complete_FP32_mathematics',output_max=float((expected-observed).abs().max()),original_output_max=float((expected-original).abs().max()),alpha_gradients=dict(original=float(original_grad[alpha_index]),fp32_reference=float(first[alpha_index]),batched=float(second[alpha_index])),nonzero_gradient_relative_rms_max=max(errors),gradient_absolute_rms_max=max(absolute_errors),theoretically_zero_gradients=zero_gradient_checks,all_gradient_comparisons=gradients))
   del actual,reference,feature,hidden,positions,expected,observed,original,first,second,original_grad
   torch.cuda.empty_cache()
print('BATCHED_READOUT_EQUIVALENCE',json.dumps(report),flush=True)

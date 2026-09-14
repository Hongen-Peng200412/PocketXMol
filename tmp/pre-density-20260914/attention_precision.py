"""只核对距离偏置候选计算的混合精度误差，不以单算子计时决定正式配置。"""
import json,torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from models.density_readout import density_attention
torch.manual_seed(78)
inputs=[torch.randn(2,4,n,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for n in (30,216,216)]
inputs += [torch.randn(2,n,3,device='cuda',requires_grad=True)*10 for n in (30,216)]
inputs += [torch.randn(4,device='cuda',requires_grad=True)]
q,k,v,x,p,beta=inputs
reference=density_attention(*(a.float() for a in inputs),True,'reference')
probe=torch.randn_like(reference)
ref_grads=torch.autograd.grad((reference*probe).sum(),inputs,retain_graph=True)
report=[]
for name in ('augmented_bf16','augmented_fp16','matrix_efficient','matrix_math'):
 try:
  if name.startswith('augmented'):
   args=[a.half() for a in inputs[:3]]+inputs[3:] if name.endswith('fp16') else inputs
   actual=density_attention(*args,True,'flash')
  else:
   bias=-F.softplus(beta)[None,:,None,None]*(x[:,:,None]-p[:,None]).square().sum(-1)[:,None]/100
   with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION if name=='matrix_efficient' else SDPBackend.MATH):
    actual=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,scale=.125)
  gradients=torch.autograd.grad((actual.float()*probe).sum(),inputs,retain_graph=True)
  errors=[float((g.float()-r.float()).square().mean().sqrt()/r.float().square().mean().sqrt().clamp_min(1e-7)) for g,r in zip(gradients,ref_grads)]
  report.append(dict(name=name,output_max_error=float((actual-reference).abs().max()),output_relative_rms=float((actual-reference).square().mean().sqrt()/reference.square().mean().sqrt()),gradient_relative_rms=errors))
 except Exception as error: report.append(dict(name=name,error=repr(error)))
print('ATTENTION_PRECISION',json.dumps(report),flush=True)

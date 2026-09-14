"""对照冻结Pocket_Plus源码的单次U-Net与三维RoPE，及GPU Flash的输出/梯度。"""
import json,sys,time
from pathlib import Path
from copy import deepcopy
import torch
sys.path.insert(0,str(Path(__file__).parent/'reference'))
from raunet import SimpleUnet
from attention_3d_rope import AttentionWith3DRoPE
from models.density_backbone import DensityEncoder,VolumeAttention
from models.density_readout import density_attention


def attention_name(key):
 if key.split('.')[0] in ('q','k','v'): return key.replace('.', '.0.',1)
 if key.startswith('back.'): return key.replace('back.','back.1.',1)
 if key.startswith('norm2.'): return key.replace('norm2.','transition1.norm.',1)
 if key.startswith(('w1.','w2.','w3.')): return 'transition1.'+key
 return key


torch.set_num_threads(1);torch.manual_seed(64)
report={'torch':torch.__version__,'cuda':torch.version.cuda,'checks':[]}
reference=AttentionWith3DRoPE(256,8,192).cuda()
actual=VolumeAttention('reference').cuda()
state=reference.state_dict()
actual.load_state_dict({key:state[attention_name(key)] for key in actual.state_dict()},strict=True)
x=torch.randn(2,256,3,3,3,device='cuda',requires_grad=True)
a=actual(x);b=reference(x)
torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
w=torch.randn_like(a)
ga=torch.autograd.grad((a*w).sum(),[x,*actual.parameters()]);gb=torch.autograd.grad((b*w).sum(),[x,*reference.parameters()])
# 原模块parameter注册顺序不同，按名字比较。
byname={name:grad for (name,_),grad in zip(reference.named_parameters(),gb[1:])}
torch.testing.assert_close(ga[0],gb[0],rtol=2e-4,atol=1e-5)
for (name,_),grad in zip(actual.named_parameters(),ga[1:]): torch.testing.assert_close(grad,byname[attention_name(name)],rtol=3e-4,atol=2e-5)
report['checks'].append('PocketPlus_RoPE_output_input_parameter_gradients')

actual.backend='flash'
with torch.autocast('cuda',dtype=torch.bfloat16): a=actual(x)
b=reference(x)
torch.testing.assert_close(a,b,rtol=.03,atol=.02)
w=torch.randn_like(a)
ga=torch.autograd.grad((a*w).sum(),[x,*actual.parameters()]);gb=torch.autograd.grad((b*w).sum(),[x,*reference.parameters()])
byname={name:grad for (name,_),grad in zip(reference.named_parameters(),gb[1:])}
errors=[]
for first,second in [(ga[0],gb[0]),*[(grad,byname[attention_name(name)]) for (name,_),grad in zip(actual.named_parameters(),ga[1:])]]:
 relative=(first-second).square().mean().sqrt()/second.square().mean().sqrt().clamp_min(1e-8)
 errors.append(float(relative));assert relative<.04,errors
report['checks'].append({'PocketPlus_RoPE_forced_Flash_bf16':'output_input_parameter_gradients','relative_gradient_rms_max':max(errors)})
del reference,actual,a,b,ga,gb,x

actual=DensityEncoder(dict(mode='D4',checkpoint=False,attention_backend='reference')).cuda().eval()
reference=SimpleUnet(56,48,gradient_checkpoint=False).cuda().eval()
mapped={}
for name,value in actual.state_dict().items():
 new=name
 if name.startswith('input_projection.'): new=name.replace('input_projection.','shortconvadd.',1)
 elif name.startswith('stem.'): new=name.replace('stem.','shortconv1.',1)
 elif name.startswith('down.'):
  _,idx,tail=name.split('.',2);new=f'downsample{int(idx)+1}.{tail}'
 elif name.startswith('attention.'):
  _,idx,tail=name.split('.',2);new=f'A_block.{idx}.{attention_name(tail)}'
 elif name.startswith('decoder3.'): new=name.replace('decoder3.','main1.',1)
 elif name.startswith('gates.'):
  _,idx,tail=name.split('.',2);new=f'attn{int(idx)+2}.{tail}'
 elif name.startswith('decoders.'):
  _,idx,tail=name.split('.',2);new=f'main{int(idx)+2}.{tail}'
 elif name.startswith('output_convs.'):
  _,idx,tail=name.split('.',2);new=f'conv_end_{(3,5,7)[int(idx)]}.{tail}'
 elif name.startswith('output.'): new=name.replace('output.','conv_end.',1)
 mapped[new]=value
incompatible=reference.load_state_dict(mapped,strict=False)
assert all(k.endswith('pos_encoding.inv_freq') for k in incompatible.missing_keys),incompatible
x=torch.randn(1,56,48,48,48,device='cuda')
with torch.no_grad():
 a=actual(x);b=reference(x,run_iters=1)
torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
report['checks'].append('PocketPlus_full_UNet_48cube_single_pass_output')
del reference,actual,a,b,x;torch.cuda.empty_cache()

for biased in (False,True):
 inputs=[torch.randn(2,4,n,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for n in (30,216,216)]
 inputs += [torch.randn(2,n,3,device='cuda',requires_grad=True)*10 for n in (30,216)]
 inputs += [torch.randn(4,device='cuda',requires_grad=True)]
 # Float32显式参考避免把低精度参考舍入当成高精度真值。
 with torch.autocast('cuda',enabled=False): b=density_attention(*(x.float() for x in inputs),biased,'reference')
 with torch.autocast('cuda',dtype=torch.bfloat16): a=density_attention(*inputs,biased,'flash')
 torch.testing.assert_close(a.float(),b,rtol=.03,atol=.008)
 w=torch.randn_like(a)
 ga=torch.autograd.grad((a*w).sum(),inputs,retain_graph=True,allow_unused=True)
 gb=torch.autograd.grad((b*w.float()).sum(),inputs,allow_unused=True)
 errors=[]
 for first,second in zip(ga,gb):
  if first is None: assert second is None;continue
  rms=(first.float()-second.float()).square().mean().sqrt()
  scale=second.float().square().mean().sqrt().clamp_min(1e-7)
  errors.append(float(rms/scale));assert rms/scale<.04,errors
 report['checks'].append({'distance':biased,'backend':'fp32_bias_math' if biased else 'forced_flash','relative_gradient_rms_errors':errors})
print('SOURCE_AND_FLASH_EQUIVALENCE',json.dumps(report),flush=True)

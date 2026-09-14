"""密度裁块、三维读出与高效attention的数值/梯度契约。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from docking.density import load_density_input,read_density_source
from models.density_backbone import VolumeAttention,DensityEncoder
from models.density_readout import DensityReadout,density_attention


def test_crop_geometry_and_full_receptor_mask(tmp_path):
    density=tmp_path/'density'/'example'
    receptor=tmp_path/'parse'/'example'
    density.mkdir(parents=True)
    receptor.mkdir(parents=True)
    grid=np.arange(52**3,dtype=np.float32).reshape(1,52,52,52)/10000
    for name in ('exp','sim'):
        np.save(density/f'{name}.npy',grid)
        np.savez(density/f'{name}.npz',voxel_size=np.array([1.,2.,3.],dtype=np.float32),origin=np.array([10.,20.,30.],dtype=np.float32))
    np.savez(receptor/'receptor_tokens.npz',coords=np.array([[15.,30.,45.]],dtype=np.float32),res_type=np.array([28]))
    result=load_density_input(tmp_path,'example',[11.,22.,33.],[11.,22.,33.])
    assert result['density_input'].shape==(1,56,48,48,48)
    np.testing.assert_array_equal(result['density_start_zyx'],[[0,0,0]])
    np.testing.assert_allclose(result['density_origin'],[[-1,-2,-3]])
    np.testing.assert_allclose(result['density_basis'][0],np.diag([1,2,3]))
    # UNK源原子确实参与smooth；exp_nonorm_smooth1为通道5。
    assert result['density_input'][0,5,5,5,5]==0
    assert result['density_input'][0,0,5,5,5]>0
    read_density_source.cache_clear()


@pytest.mark.parametrize('distance_bias',[False,True])
def test_attention_output_and_all_input_gradients(distance_bias):
    torch.manual_seed(41)
    inputs=[torch.randn(2,4,n,64,dtype=torch.float64,requires_grad=True) for n in (5,11,11)]
    inputs += [torch.randn(2,n,3,dtype=torch.float64,requires_grad=True) for n in (5,11)]
    inputs += [torch.randn(4,dtype=torch.float64,requires_grad=True)]
    reference=density_attention(*inputs,distance_bias,'reference')
    actual=density_attention(*inputs,distance_bias,'sdpa')
    torch.testing.assert_close(actual,reference,rtol=1e-10,atol=1e-10)
    probe=torch.randn_like(reference)
    expected_grad=torch.autograd.grad((reference*probe).sum(),inputs,retain_graph=True,allow_unused=True)
    actual_grad=torch.autograd.grad((actual*probe).sum(),inputs,allow_unused=True)
    for expected,observed in zip(expected_grad,actual_grad):
        if expected is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed,expected,rtol=1e-9,atol=1e-10)


def test_volume_attention_output_and_parameter_gradients():
    torch.manual_seed(42)
    reference=VolumeAttention('reference').double()
    optimized=deepcopy(reference)
    optimized.backend='sdpa'
    feature=torch.randn(1,256,2,3,2,dtype=torch.float64,requires_grad=True)
    expected=reference(feature)
    actual=optimized(feature)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    weight=torch.randn_like(actual)
    gradients1=torch.autograd.grad((expected*weight).sum(),[feature,*reference.parameters()])
    gradients2=torch.autograd.grad((actual*weight).sum(),[feature,*optimized.parameters()])
    for first,second in zip(gradients1,gradients2):
        torch.testing.assert_close(second,first,rtol=1e-8,atol=1e-9)


def test_d1_has_no_decoder_and_correct_coarse_centers():
    config=dict(mode='D1',attention_backend='sdpa',checkpoint=False,distance_bias=True)
    encoder=DensityEncoder(config)
    assert len(encoder.down)==3
    assert not hasattr(encoder,'decoder3')
    reader=DensityReadout(320,config)
    assert reader.voxel_xyz.shape==(216,3)
    torch.testing.assert_close(reader.voxel_xyz[0],torch.tensor([.5,.5,.5]))
    torch.testing.assert_close(reader.voxel_xyz[-1],torch.tensor([40.5,40.5,40.5]))


def test_d4_outside_intersection_empty_and_rigid_invariance():
    torch.manual_seed(43)
    reader=DensityReadout(320,dict(mode='D4',attention_backend='reference',checkpoint=False,distance_bias=True))
    feature=torch.randn(1,48,48,48,48)
    hidden=torch.randn(3,320)
    positions=torch.tensor([[0.2,0.3,0.4],[-2.5,0.1,0.1],[-5.,0.1,0.1]])
    batch=torch.zeros(3,dtype=torch.long)
    origin=torch.zeros(1,3)
    basis=torch.eye(3)[None]
    actual=reader(hidden,positions,batch,feature,origin,basis)
    assert actual[0].abs().sum()>0 and actual[1].abs().sum()>0
    assert torch.count_nonzero(actual[2])==0
    rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    shift=torch.tensor([3.,-2,1])
    rotated=reader(hidden,positions@rotation+shift,batch,feature,origin@rotation+shift,basis@rotation)
    torch.testing.assert_close(rotated,actual,rtol=1e-5,atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA自动混合精度几何验收')
def test_cuda_bf16_preserves_home_at_crop_intersection_boundary():
    reader=DensityReadout(320,dict(mode='D4',attention_backend='reference',checkpoint=False,distance_bias=True)).cuda()
    feature=torch.randn(1,48,48,48,48,device='cuda')
    hidden=torch.randn(2,320,device='cuda')
    # 第一个home=50，其-3邻居47仍在块内；第二个home=51，全部邻居在块外。
    positions=torch.tensor([[50.9999*.7,3.2*.7,3.2*.7],[51.0001*.7,3.2*.7,3.2*.7]],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=reader(hidden,positions,torch.zeros(2,dtype=torch.long,device='cuda'),feature,torch.zeros(1,3,device='cuda'),torch.eye(3,device='cuda')[None]*.7)
    assert result[0].abs().sum()>0
    assert torch.count_nonzero(result[1])==0

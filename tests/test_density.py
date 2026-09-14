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


@pytest.mark.parametrize('mode', ['D1', 'D4'])
def test_readout_preserves_individual_molecule_geometry(mode):
    """用不同角点、旋转和间距的两个分子, 核对拼批与分别读出的全部残差."""
    torch.manual_seed(84)
    reader = DensityReadout(320, dict(mode=mode, attention_backend='reference', distance_bias=True))
    channels, side = (256, 6) if mode == 'D1' else (48, 48)
    feature = torch.randn(2, channels, side, side, side)
    hidden = torch.randn(5, 320)
    batch = torch.tensor([0, 0, 0, 1, 1])
    origin = torch.tensor([[0., 0., 0.], [9., -4., 2.]])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    basis = torch.stack([torch.eye(3), torch.diag(torch.tensor([.9, 1.1, 1.2])) @ rotation])
    indices = torch.tensor([[.2, .3, .4], [-2.5, .1, .1], [-5., .1, .1], [10.2, 9.8, 4.3], [51.1, 3., 3.]])
    positions = origin[batch] + (indices[:, :, None] * basis[batch]).sum(-2)
    actual = reader(hidden, positions, batch, feature, origin, basis)
    expected = []
    for molecule in (0, 1):
        selected = batch == molecule
        expected.append(reader(hidden[selected], positions[selected], torch.zeros(int(selected.sum()), dtype=torch.long), feature[molecule:molecule + 1], origin[molecule:molecule + 1], basis[molecule:molecule + 1]))
    torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mode',['D1','D4'])
def test_batched_readout_keeps_variable_size_molecules_separate(mode):
    torch.manual_seed(92)
    reader=DensityReadout(320,dict(mode=mode,attention_backend='sdpa',checkpoint=False,distance_bias=True))
    side,channels=(6,256) if mode=='D1' else (48,48)
    feature=torch.randn(2,channels,side,side,side)
    hidden=torch.randn(5,320)
    positions=torch.randn(5,3)+2
    batch=torch.tensor([0,0,0,1,1])
    origin=torch.tensor([[0.,0.,0.],[-2.,1.,-.5]])
    basis=torch.eye(3).repeat(2,1,1)
    actual=reader(hidden,positions,batch,feature,origin,basis)
    expected=torch.cat([reader(hidden[batch==i],positions[batch==i],torch.zeros(int((batch==i).sum()),dtype=torch.long),feature[i:i+1],origin[i:i+1],basis[i:i+1]) for i in range(2)])
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA自动混合精度几何验收')
@pytest.mark.parametrize('matmul_precision',['highest','medium'])
def test_cuda_bf16_preserves_home_at_crop_intersection_boundary(matmul_precision):
    reader=DensityReadout(320,dict(mode='D4',attention_backend='reference',checkpoint=False,distance_bias=True)).cuda()
    feature=torch.randn(1,48,48,48,48,device='cuda')
    hidden=torch.randn(2,320,device='cuda')
    # 第一个home=50，其-3邻居47仍在块内；第二个home=51，全部邻居在块外。
    positions=torch.tensor([[50.9999*.7,3.2*.7,3.2*.7],[51.0001*.7,3.2*.7,3.2*.7]],device='cuda')
    previous_precision=torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(matmul_precision)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=reader(hidden,positions,torch.zeros(2,dtype=torch.long,device='cuda'),feature,torch.zeros(1,3,device='cuda'),torch.eye(3,device='cuda')[None]*.7)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    assert result[0].abs().sum()>0
    assert torch.count_nonzero(result[1])==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA自动混合精度距离输出及梯度验收')
@pytest.mark.parametrize('distance_bias',[False,True])
def test_cuda_bf16_attention_against_fp32_reference(distance_bias):
    if not distance_bias and not getattr(torch.backends.cuda, 'is_flash_attention_available', lambda: False)():
        pytest.skip('当前PyTorch未报告Flash可用；指定A800/Linux环境必须执行此检查')
    torch.manual_seed(64)
    inputs=[torch.randn(2,4,n,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for n in (30,216,216)]
    inputs += [torch.randn(2,n,3,device='cuda',requires_grad=True)*10 for n in (30,216)]
    inputs += [torch.randn(4,device='cuda',requires_grad=True)]
    reference=density_attention(*(value.float() for value in inputs),distance_bias,'reference')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=density_attention(*inputs,distance_bias,'flash')
    torch.testing.assert_close(actual.float(),reference,rtol=.03,atol=.008)
    weight=torch.randn_like(actual)
    expected=torch.autograd.grad((reference*weight.float()).sum(),inputs,retain_graph=True,allow_unused=True)
    observed=torch.autograd.grad((actual*weight).sum(),inputs,allow_unused=True)
    for first,second in zip(expected,observed):
        if first is None:
            assert second is None
        else:
            relative_rms=(first.float()-second.float()).square().mean().sqrt()/first.float().square().mean().sqrt().clamp_min(1e-7)
            assert relative_rms<.04

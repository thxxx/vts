import k_diffusion as K
import torch
import torchsde
from tqdm.auto import trange, tqdm

class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]

class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, cross_attention_inputs, cross_attention_masks, seconds_start, seconds_total, cfg_scale=6, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, audio_embed=None):
    """DPM-Solver++(3M) SDE."""

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(
            x,
            sigmas[i] * s_in,
            mask=None,
            input_ids=cross_attention_inputs,
            attention_mask=cross_attention_masks,
            seconds_start=seconds_start,
            seconds_total=seconds_total,
            cfg_dropout_prob=0.0,
            cfg_scale=cfg_scale,
            scale_phi=0.75
        )
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

def generation(
    model,
    ae,
    text_conditioner,
    text,
    steps,
    cfg_scale,
    duration=3.0,
    sample_rate=44100,
    batch_size=1,
    device='cuda',
    disable=False,
    train_duration=10.0,
    melody_prompt=None,
    audio_embed=None
):
    latent_channels = 64
    noise = torch.randn([batch_size, latent_channels, int(sample_rate*train_duration) // ae.downsampling_ratio], device=device, dtype=torch.float32)
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    
    input_ids, attention_mask = text_conditioner([text], device=device)
    seconds_start = torch.tensor([[0]], dtype=torch.float32)
    seconds_total = torch.tensor([[duration]], dtype=torch.float32)
    
    input_ids = input_ids.to(torch.float32)
    attention_mask = attention_mask.to(torch.float32)
    
    cross_attention_inputs, cross_attention_masks, global_cond = model.get_context(input_ids, attention_mask, seconds_start, seconds_total)

    model_dtype = next(model.parameters()).dtype
    noise = noise.type(model_dtype)
    
    sigmas = K.sampling.get_sigmas_polyexponential(100, 0.3, 500, rho=1.0, device=device)
    
    denoiser = K.external.VDenoiser(model)
    
    x = noise * sigmas[0]
    # print("noise : ", x)
    
    torch.set_printoptions(precision=10)  # 소수점 10자리까지 출력

    # with torch.cuda.amp.autocast():
    with torch.inference_mode():
        out = sample_dpmpp_3m_sde(
            denoiser, 
            x, 
            sigmas, 
            cross_attention_inputs,
            cross_attention_masks,
            seconds_start,
            seconds_total,
            disable=disable, 
            cfg_scale=cfg_scale, 
            callback=None,
            melody_prompt=melody_prompt,
            audio_embed=audio_embed
        )
        out = out.to(next(ae.parameters()).dtype)
        audio = ae.decode(out)
    
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak
    audio = audio.clamp(-1, 1)
    
    return audio
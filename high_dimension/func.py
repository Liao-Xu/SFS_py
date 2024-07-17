import numpy as np
from joblib import Parallel, delayed
import torch

def precompute_stats(sigma, alpha, device):
    inv_sigma = [torch.linalg.inv(torch.tensor(s, device=device)) for s in sigma]
    is_a = [torch.matmul(inv, a) for inv, a in zip(inv_sigma, alpha)]
    at_is_a = [torch.matmul(a.view(-1, 1).mT, ia.view(-1, 1)).squeeze() for a, ia in zip(alpha, is_a)]
    return list(zip(inv_sigma, is_a, at_is_a))

def b_func(y, p, kap, t, Ip, stats, alpha, sigma, theta, device):
    B = y.shape[1]  # Batch size
    b_de = torch.zeros((kap, B), device=device)
    b_nu = torch.zeros((kap, p, B), device=device)

    inv_sigmas, is_as, at_is_as = zip(*stats)
    inv_sigmas = torch.stack(inv_sigmas)
    is_as = torch.stack(is_as)
    at_is_as = torch.stack(at_is_as)
    alphas = torch.stack(alpha)
    sigmas = torch.stack([torch.tensor(s, device=device) for s in sigma])

    t_is = (1 - t) * inv_sigmas

    t_is_y = torch.einsum('ijk,ik->ij', t_is, alphas)[:, :, None] + y

    com_I = torch.linalg.inv(t * Ip + t_is)

    com_sig = torch.linalg.det(t * sigmas + (1 - t) * Ip)

    com_sig = torch.maximum(com_sig, torch.tensor(1e-15, device=device))

    t_is_y_expanded = t_is_y[:, :, :, None]

    t_is_y_mult = torch.einsum('ijk,iklj->ikl', torch.sqrt(torch.maximum(com_I, torch.tensor(0.0, device=device))), t_is_y_expanded)

    exp_input = torch.clip(
        (torch.sum(t_is_y_mult**2, axis=1) - torch.sum(y**2, axis=0)) / (2 - 2 * t) - 0.5 * at_is_as[:, None],
        None, 50
    )

    g = torch.exp(exp_input)

    b_de = g / torch.sqrt(com_sig[:, None])

    b_de_expanded = b_de[:, None, :]

    Ip_minus_inv_sigmas = Ip - inv_sigmas

    intermediate = torch.einsum('ijk,ikl->ijl', Ip_minus_inv_sigmas, com_I)

    term2 = torch.einsum('ijk,ikl->ijl', intermediate, t_is_y)

    b_nu = (is_as[:, :, None] + term2) * b_de_expanded
    b_nu = b_nu.permute(1, 0, 2)

    b_de_theta_product = torch.sum(b_de * theta[:, None], axis=0)

    result = torch.einsum('ijk,j->ik', b_nu, theta) / b_de_theta_product

    return result

def sample_gauss(p, kap, K, alpha, sigma, theta, stats, B, device):
    Ip = torch.eye(p, device=device)
    y = torch.zeros((p, K, B), device=device)
    s = 1 / K
    for k in range(K-1):
        eps = torch.from_numpy(np.random.multivariate_normal(np.zeros(p), np.eye(p), size=B)).T.to(device)
        y[:, k+1, :] = y[:, k, :] + s * b_func(y[:, k, :], p, kap, k*s, Ip, stats, alpha, sigma, theta, device) + torch.sqrt(torch.tensor(s, device=device)) * eps
    return y[:, K-1, :]

def SFS_gauss(N, p, kap, K, alpha, sigma, theta, B=1, parallel=False, device='cpu'):
    alpha = [torch.tensor(a, device=device) for a in alpha]
    theta = torch.tensor(theta, device=device)
    stats = precompute_stats(sigma, alpha, device)
    results = torch.zeros((p, N), device=device)

    if not parallel:
        for i in range(N // B):
            results[:, i*B:(i+1)*B] = sample_gauss(p, kap, K, alpha, sigma, theta, stats, B, device)
    else:
        results_batches = Parallel(n_jobs=parallel)(delayed(sample_gauss)(p, kap, K, alpha, sigma, theta, stats, B, device) for _ in range(N // B))
        results = torch.cat(results_batches, dim=1)
    return results.cpu().numpy()

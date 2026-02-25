import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import math

"""
NOTE: The initial implementation of this module was prototyped using an LLM. 
The code has since been refined and manually verified to ensure accurate 
transformation of points to the unit hypercube. The inverse transformation 
has also been validated across all tested cases.
"""

#Wrapper function that picks correct transformation
def transform(x, cfg, bounds, means=None, covariances=None, weights=None):
    if (cfg.exp.q_dist=='uniform') or (cfg.exp.target=='llm_prior'):
        return transform_uniform_to_unitcube(x, bounds)
    elif cfg.exp.q_dist=='mixture_gaussian':
        return transform2unitcube(x, means, covariances, weights)
    else:
        raise NotImplementedError("Selected cadidate sampling distribution q does not have transformation")
def inversetransform(x, cfg, bounds, means=None, covariances=None, weights=None):
    if (cfg.exp.q_dist=='uniform') or (cfg.exp.target=='llm_prior'):
        return inverse_transform_uniform_from_unitcube(x, bounds)
    elif cfg.exp.q_dist=='mixture_gaussian':
        return inverse_transform2unitcube(x, means, covariances, weights, lowerbound=bounds[0][0], upperbound=bounds[0][1])
    else:
        raise NotImplementedError("Selected cadidate sampling distribution q does not have transformation")
def transform_dataset(data, cfg, bounds, means=None, covariances=None, weights=None):
    if (cfg.exp.q_dist=='uniform') or (cfg.exp.target=='llm_prior'):
        return transform_uniform_dataset_to_unitcube(data, bounds)
    elif cfg.exp.q_dist=='mixture_gaussian':
        return transform_dataset_to_unitcube(data, means, covariances, weights)
    else:
        raise NotImplementedError("Selected cadidate sampling distribution q does not have transformation")


def transform_uniform_to_unitcube(x, bounds):
    x = x if torch.is_tensor(x) else torch.tensor(x)
    """
    Linearly transforms x from box bounds X to [-0.5, 0.5]^d.
    """
    a = torch.tensor([b[0] for b in bounds], dtype=x.dtype, device=x.device)
    b = torch.tensor([b[1] for b in bounds], dtype=x.dtype, device=x.device)
    return (x - a) / (b - a) - 0.5
def inverse_transform_uniform_from_unitcube(u, bounds):
    u = u if torch.is_tensor(u) else torch.tensor(u)
    """
    Inverts transformation from [-0.5, 0.5]^d back to original box X.
    """
    a = torch.tensor([b[0] for b in bounds], dtype=u.dtype, device=u.device)
    b = torch.tensor([b[1] for b in bounds], dtype=u.dtype, device=u.device)
    return (u + 0.5) * (b - a) + a


def transform_uniform_dataset_to_unitcube(data, bounds):
    num_alternatives, D, N = data.shape
    device = data.device
    # Reshape to (num_alternatives * N, D)
    data_reshaped = data.permute(2, 0, 1).reshape(-1, D)
    # Apply transform to each point
    transformed = torch.stack([
        transform_uniform_to_unitcube(x if torch.is_tensor(x) else torch.tensor(x).to(device), bounds) for x in data_reshaped
    ])
    # Reshape back to (N, num_alternatives, D), then permute to (num_alternatives, D, N)
    transformed = transformed.view(N, num_alternatives, D).permute(1, 2, 0)
    return transformed



def rosenblatt_transform_gmm(x_batch, means, covariances, weights):
    """
    Applies the Rosenblatt transform to a batch of points x_batch under a GMM.
    This version is vectorized and uses Cholesky decomposition for efficiency.

    Args:
        x_batch (Tensor): Input samples of shape (N, D), where N is the batch size.
        means (Tensor): GMM component means, shape (K, D).
        covariances (Tensor): GMM component covariances, shape (K, D, D).
        weights (Tensor): GMM component weights, shape (K,).

    Returns:
        Tensor: Transformed samples in the [-0.5, 0.5]^D hypercube, shape (N, D).
    """
    N, D = x_batch.shape
    K = means.shape[0]

    # Pre-compute Cholesky decompositions for all components
    cholesky_factors = torch.linalg.cholesky(covariances)
    log_2pi = math.log(2 * math.pi)

    u_batch = torch.empty_like(x_batch)

    for d in range(D):
        x_d_batch = x_batch[:, d]  # Current dimension for all samples, shape (N,)

        if d == 0:
            # Marginal distribution for the first dimension
            mean_d0 = means[:, 0]
            std_d0 = torch.sqrt(covariances[:, 0, 0])
            dist = Normal(mean_d0, std_d0)
            # component_cdfs shape: (N, K)
            component_cdfs = dist.cdf(x_d_batch.unsqueeze(1))
            # Responsibilities are just the prior weights
            responsibilities = weights.unsqueeze(0)
        else:
            # Conditional distribution for dimension d > 0
            x_prev_batch = x_batch[:, :d]  # Shape (N, d)

            # Extract parameters for all K components
            mu_1 = means[:, :d]              # (K, d)
            mu_2 = means[:, d]               # (K,)
            Sigma_12 = covariances[:, :d, d]   # (K, d)
            Sigma_22 = covariances[:, d, d]   # (K,)
            L_11 = cholesky_factors[:, :d, :d] # (K, d, d)

            # Instead of inverting Sigma_11, solve for the term A = Sigma_21 @ Sigma_11^-1
            # We solve (L_11 @ L_11^T) @ A^T = Sigma_12
            A_T = torch.cholesky_solve(Sigma_12.unsqueeze(2), L_11) # Shape: (K, d, 1)
            A = A_T.squeeze(2)  # Shape: (K, d)

            # Calculate conditional means for all N samples and K components
            diff = x_prev_batch.unsqueeze(1) - mu_1.unsqueeze(0) # Shape: (N, K, d)
            cond_mean_correction = torch.einsum('nkd,kd->nk', diff, A)
            cond_mean = mu_2.unsqueeze(0) + cond_mean_correction # Shape: (N, K)

            # Calculate conditional variances (same for all N samples)
            cond_var = Sigma_22 - torch.einsum('kd,kd->k', A, Sigma_12)
            cond_var = torch.clamp(cond_var, min=1e-9) # For numerical stability
            cond_std = torch.sqrt(cond_var) # Shape: (K,)

            dist = Normal(cond_mean, cond_std.unsqueeze(0))
            component_cdfs = dist.cdf(x_d_batch.unsqueeze(1)) # Shape: (N, K)
            #fix to improve numerical stability
            component_cdfs = torch.clamp(component_cdfs, 1e-7, 1 - 1e-7) 

            # --- Calculate responsibilities p(k | x_prev) in log space for stability ---
            log_det_L = torch.log(torch.diagonal(L_11, dim1=-2, dim2=-1))
            log_det_Sigma = 2 * torch.sum(log_det_L, dim=1)

            # Solve L_11 @ v = diff^T to find v = L_11^-1 @ diff^T
            # Then quad_term = v^T @ v
            diff_T = diff.permute(1, 2, 0) # Shape: (K, d, N)
            v = torch.linalg.solve_triangular(L_11, diff_T, upper=False) # Shape: (K, d, N)
            quad_term = torch.sum(v**2, dim=1).T # Sum over d -> (K, N) -> (N, K)

            log_prob_x_prev = -0.5 * (d * log_2pi + log_det_Sigma.unsqueeze(0) + quad_term)
            log_weights = torch.log(weights).unsqueeze(0)
            log_responsibilities = log_weights + log_prob_x_prev

            # Normalize using LogSumExp trick
            log_responsibilities_sum = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
            responsibilities = torch.exp(log_responsibilities - log_responsibilities_sum) # Shape: (N, K)
            #Fix to improve numerical stability
            responsibilities = torch.clamp(responsibilities, 1e-12, 1.0)
            responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True)

        # Calculate the CDF value for the current dimension
        u_d_batch = torch.sum(responsibilities * component_cdfs, dim=1)
        u_batch[:, d] = u_d_batch

    return u_batch - 0.5



def transform2unitcube(samples, means, covariances, weights):
    transformed = torch.stack([
        rosenblatt_transform_gmm(x if torch.is_tensor(x) else torch.tensor(x), means, covariances, weights) for x in samples
    ])
    return transformed

def transform_dataset_to_unitcube(data, means, covariances, weights):
    """
    Apply the Rosenblatt transformation to each data point in a tensor of shape (num_alternatives, D, N),
    transforming along the D dimension.
    Args:
        data (Tensor): Input data of shape (num_alternatives, D, N)
        means (Tensor): GMM component means, shape (K, D)
        covariances (Tensor): GMM component covariances, shape (K, D, D)
        weights (Tensor): GMM component weights, shape (K,)
    Returns:
        Tensor: Transformed data of shape (num_alternatives, D, N)
    """
    num_alternatives, D, N = data.shape
    device = data.device
    # Reshape to (num_alternatives * N, D)
    data_reshaped = data.permute(2, 0, 1).reshape(-1, D)
    transformed = rosenblatt_transform_gmm(data_reshaped, means, covariances, weights)
    # Reshape back to (N, num_alternatives, D), then permute to (num_alternatives, D, N)
    transformed = transformed.view(N, num_alternatives, D).permute(1, 2, 0)
    return transformed


def inverse_rosenblatt_transform_gmm(u_batch, means, covariances, weights, lowerbound, upperbound, tol=1e-5, max_iter=100):
    """
    Applies the inverse Rosenblatt transform to a batch of points u_batch.
    This version is vectorized and uses a parallelized bisection search.

    Args:
        u_batch (Tensor): Input samples from [-0.5, 0.5]^D, shape (N, D).
        means, covariances, weights: GMM parameters.
        lowerbound, upperbound: Search bounds for the bisection algorithm.
        tol, max_iter: Parameters for the bisection algorithm.

    Returns:
        Tensor: Transformed samples from the GMM distribution, shape (N, D).
    """
    u_batch = u_batch + 0.5  # Map from [-0.5, 0.5] to the [0, 1] hypercube
    N, D = u_batch.shape
    K = means.shape[0]
    device = u_batch.device
    
    # Pre-compute Cholesky factors and constants
    cholesky_factors = torch.linalg.cholesky(covariances)
    log_weights = torch.log(weights)
    log_2pi = math.log(2 * math.pi)

    x_batch = torch.empty_like(u_batch)

    # Helper function to compute the conditional CDF for a batch
    def vectorized_cdf_mix(xd_batch, x_prev_batch, d):
        if d == 0: # Marginal case
            dist = Normal(means[:, 0], torch.sqrt(covariances[:, 0, 0]))
            component_cdfs = dist.cdf(xd_batch.unsqueeze(1))
            responsibilities = weights.unsqueeze(0)
        else: # Conditional case (logic copied from forward transform)
            mu_1, mu_2 = means[:, :d], means[:, d]
            Sigma_12, Sigma_22 = covariances[:, :d, d], covariances[:, d, d]
            L_11 = cholesky_factors[:, :d, :d]
            
            A_T = torch.cholesky_solve(Sigma_12.unsqueeze(2), L_11)
            A = A_T.squeeze(2)
            
            diff = x_prev_batch.unsqueeze(1) - mu_1.unsqueeze(0)
            cond_mean = mu_2.unsqueeze(0) + torch.einsum('nkd,kd->nk', diff, A)
            cond_var = torch.clamp(Sigma_22 - torch.einsum('kd,kd->k', A, Sigma_12), min=1e-9)
            cond_std = torch.sqrt(cond_var)
            
            dist = Normal(cond_mean, cond_std.unsqueeze(0))
            component_cdfs = dist.cdf(xd_batch.unsqueeze(1))
            component_cdfs = torch.clamp(component_cdfs, 1e-7, 1 - 1e-7) #Fix to improve numerical stability
            
            log_det_Sigma = 2 * torch.sum(torch.log(torch.diagonal(L_11, dim1=-2, dim2=-1)), dim=1)
            v = torch.linalg.solve_triangular(L_11, diff.permute(1, 2, 0), upper=False)
            quad_term = torch.sum(v**2, dim=1).T
            
            log_prob_x_prev = -0.5 * (d * log_2pi + log_det_Sigma.unsqueeze(0) + quad_term)
            log_responsibilities = log_weights.unsqueeze(0) + log_prob_x_prev
            log_responsibilities_sum = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
            responsibilities = torch.exp(log_responsibilities - log_responsibilities_sum)
            #Fix to improve numerical stability
            responsibilities = torch.clamp(responsibilities, 1e-12, 1.0)
            responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True)
            
        return torch.sum(responsibilities * component_cdfs, dim=1)

    for d in range(D):
        u_d_batch = u_batch[:, d]
        x_prev_batch = x_batch[:, :d]
        
        # Perform bisection search in parallel for all N samples
        lb_batch = torch.full((N,), lowerbound, device=device, dtype=torch.float32)
        ub_batch = torch.full((N,), upperbound, device=device, dtype=torch.float32)

        for _ in range(max_iter):
            mid_batch = (lb_batch + ub_batch) / 2
            val_batch = vectorized_cdf_mix(mid_batch, x_prev_batch, d)
            
            if torch.all(torch.abs(val_batch - u_d_batch) < tol):
                break
            
            mask = val_batch < u_d_batch
            lb_batch[mask] = mid_batch[mask]
            ub_batch[~mask] = mid_batch[~mask]
        
        x_batch[:, d] = (lb_batch + ub_batch) / 2

        
    return x_batch


def inverse_transform2unitcube(u_samples, means, covariances, weights, lowerbound, upperbound):
    return inverse_rosenblatt_transform_gmm(u_samples if torch.is_tensor(u_samples) else torch.from_numpy(u_samples), means, covariances, weights,lowerbound=lowerbound, upperbound=upperbound)
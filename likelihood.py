# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (c) 2025, Petrus Mikkola, University of Helsinki


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import functools 
from scipy.integrate import solve_ivp
from torchdiffeq import odeint

### This implemetation assumes EDM type diffusion model ###
#EMD assumes s(t)=1 and sigma(t)=t
#Assuming EMD, the diffusion coefficient reduces to sqrt(2sigma)=sqrt(2t), since:
#drift_coef(t) = s(t)sqrt(2sigma'(t)sigma(t))   #Karras et al., NeurIPS 2022, Eq. (34)
#Assuming EMD, the standard deviation of the perturbation kernel reduces to sigma=t, since:
#var_kernel(t) = s^2(t)sigma^2(t)   #Karras et al., NeurIPS 2022, Eq. (11)


def loglik(x, score_model, sigma_min, sigma_max, solver_method='implicit_adams', joint_model=True,
           clamp_log_lik=True, use_hutchinson=False, fresh_noise_hutchinson=False):

  device=x.device

  #solver_method=='dopri5': #'dopri5' is RK45 equivalent in torchdiffeq
  #solver_method=='implicit_adams': #'implicit_adams' can handle midly stiffness (RECOMMENDED: does not so likely lead to collapsed densitiess on high-density regions)

  if joint_model:
    base_score = score_model
    def marginalized_score(x, t):
        x_losers = torch.zeros_like(x, device=x.device, dtype=x.dtype) #less randomness --> integration more stable
        #x_losers = sigma_max*torch.randn(x.shape[0], x.shape[1]).to(device)
        return base_score(torch.cat([x, x_losers], dim=1), t)[:,:x.shape[1]]
  else:
     marginalized_score = score_model
     

  def marginal_prob_std(t):
   std = t if torch.is_tensor(t) else torch.tensor(t, device=device)
   return std

  def diffusion_coeff(t):  
     t = t if torch.is_tensor(t) else torch.tensor(t, device=device)
     diffusion = torch.sqrt(2.0*t)
     return diffusion
    
  marginal_prob_std_fn = functools.partial(marginal_prob_std)
  diffusion_coeff_fn = functools.partial(diffusion_coeff)
  
  def prior_likelihood(z):
    shape = z.shape
    D = shape[1]
    return -D / 2. * np.log(2 * np.pi * sigma_max ** 2) - torch.sum(z ** 2, dim=1) / (2 * sigma_max ** 2)

  def ode_likelihood(x, 
                    marginalized_score,
                    marginal_prob_std, 
                    diffusion_coeff,
                    batch_size, 
                    device,
                    sigma_min,
                    sigma_max):
    """Compute the likelihood with probability flow ODE.
    
    Args:
      x: Input data.
      score_model: A PyTorch model representing the score-based model.
      marginal_prob_std: A function that gives the standard deviation of the 
        perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the 
        forward SDE.
      batch_size: The batch size. Equals to the leading dimension of `x`.
      device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
      sigma_min: A `float` number. The smallest time step for numerical stability.

    Returns:
      log_lik: log p(x), where p is the density induced by diffusion model
    """

    if use_hutchinson:
      def divergence_eval(x, t, vs):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        divs = []
        for v in vs:
            if fresh_noise_hutchinson: v = torch.randn_like(x)
            with torch.enable_grad():
                x_ = x.detach().requires_grad_(True)
                score_e = torch.sum(marginalized_score(x_, t) * v)
                grad_score_e = torch.autograd.grad(score_e, x_, create_graph=False)[0]
            div = torch.sum(grad_score_e * v, dim=1)
            divs.append(div)
        return torch.stack(divs, dim=0).mean(dim=0)
    else:
      def divergence_eval(x, t, VS):
        """Exact divergence ∇·sθ(x,t) computed via two reverse-mode calls."""
        x = x.detach().requires_grad_(True)
        s = marginalized_score(x, t)
        D = s.shape[1]
        div = torch.zeros(x.shape[0], device=x.device)
        for d in range(D):
            grad_s_d = torch.autograd.grad(
                s[:, d].sum(), x, create_graph=False, retain_graph=True
            )[0]
            div += grad_s_d[:, d]
        return div

    class ODEFunc(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.num_hutchinson_samples = 16
          self.vs = None  #Hutchinson's vectors will be initialized at first call
      def forward(self, t, states):
          sample, logp = states
          t_tensor = t.expand(sample.shape[0]).to(device)
          if self.vs is None:
            self.vs = [torch.randn_like(sample) for _ in range(self.num_hutchinson_samples)]
          with torch.no_grad():
            score = marginalized_score(sample, t_tensor)
          g = diffusion_coeff(t)
          dx = -0.5 * g**2 * score #Assumes zero drift term
          div = divergence_eval(sample, t_tensor, self.vs)
          dlogp = -0.5 * g**2 * div
          return (dx, dlogp)

    logp0 = torch.zeros(x.shape[0], device=device) 
    ode_func = ODEFunc()
    t_span = torch.tensor([sigma_min, sigma_max], device=device)

    xT, delta_logp = odeint(
        ode_func,
        (x, logp0),
        t_span,
        rtol=1e-6,
        atol=1e-6,
        method=solver_method
    )
    xT = xT[-1]
    delta_logp = delta_logp[-1] #accumulated log-prob change

    prior_logp = prior_likelihood(xT)
    log_likelihood = prior_logp + delta_logp
    return log_likelihood
    
    #Log lik in bit per dim (not needed)
    #bpd = -(prior_logp + delta_logp) / np.log(2)
    #N = np.prod(shape[1:])
    #bpd = bpd / N + 8.
    #return z, bpd
  
  log_lik = ode_likelihood(x, marginalized_score, marginal_prob_std_fn,diffusion_coeff_fn, x.shape[0], device=device, sigma_min=sigma_min, sigma_max=sigma_max)
  if clamp_log_lik:
    log_lik = torch.clamp(log_lik,max=torch.quantile(log_lik,0.999))
  
  return log_lik

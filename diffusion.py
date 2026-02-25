# Modified work based on the original code:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Original work licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# This modified version is also licensed under the same Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# Modifications copyright (c) 2025, Petrus Mikkola, University of Helsinki

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EDM(nn.Module):

    def __init__(self, D, sigma_min = 0.002, sigma_max = 5.0, ALD_sigmas=None, L=10, joint_model=True):
        super(EDM, self).__init__()
        self.joint_model = joint_model
        self.D = D
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ALD_sigmas = ALD_sigmas
        self.L = L

    def score_function(self, x_tilde, x, sigma):
        """
        Args:
            x_tilde (torch.Tensor): The noisy sample x.
            x (torch.Tensor): The original data point x.
            sigma (torch.Tensor): The noise level
        Returns:
            torch.Tensor: The score ∇_x_tilde log p_sigma(x_tilde|x).
        """
        score = -(x_tilde - x) / sigma**2
        return score

    def sample_deterministic(self, model, n_samples, joint=0, temp=0, return_trajectory=False):

        #EDM sampler adapted from https://github.com/NVlabs/edm2/blob/main/generate_images.py
        #Assumes model.eval() mode
        
        if (self.joint_model) and (temp>0):
            print("Warning: Diffusion sampling with joint model is not guaranteed to work with tempered version.")

        device = next(model.parameters()).device

        num_steps = 32 #Heun steps
        rho = 7

        def denoise(x, sigma):
            score = model(x, sigma, joint=joint, temp=temp)
            return x + score * (sigma ** 2)

        x_init = self.sigma_max*torch.randn(n_samples, self.D).to(device) #modified: this corresponds to the marginal p_sigma_max(x)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_init.device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_cur = x_init
        trajectory = [x_cur]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

            # Euler step.
            d_cur = (x_cur - denoise(x_cur, t_cur)) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = (x_next - denoise(x_next, t_next)) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

            # Record trajectory.
            x_cur = x_next
            trajectory.append(x_cur)

        if return_trajectory:
            return torch.stack(trajectory) #torch.Size([num_steps, n_samples, D])
        else:
            return torch.stack(trajectory)[-1,:,:]
        


    def sample_stochastic(self, model, n_samples, joint=0, model_temp=0, score_temp=0):
        
        #EDM sampler adpated from https://github.com/NVlabs/edm2/blob/main/generate_images.py
        #Assumes model.eval() mode

        if (self.joint_model) and ((model_temp>0) or (score_temp>0)):
            print("Warning: Diffusion sampling with joint model is not guaranteed to work with tempered version.")

        device = next(model.parameters()).device
        dtype = torch.float32

        #Check Table 5 in Karras et al. (2021)
        num_steps = 256
        rho = 7
        S_churn = 30
        S_min = 0.01
        S_max = 1
        S_noise = 1.007

        def denoise(x, sigma):
            score = score_temp*model(x, sigma, joint=joint, temp=model_temp)
            return x + score * (sigma ** 2)

        x_init = self.sigma_max*torch.randn(n_samples, self.D).to(device) #modified: this corresponds to the marginal p_sigma_max(x)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        t_steps = (self.sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = x_init * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            if S_churn > 0 and S_min <= t_cur <= S_max:
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
            else:
                t_hat = t_cur
                x_hat = x_cur

            # Euler step.
            d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = (x_next - denoise(x_next, t_next)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
    
    
    def sample_annealed_langevin(self, model, n_samples, joint=0, score_scaled=True, tempfield=None, T=10, epsilon=0.1):

        if score_scaled and tempfield is None:
            raise NotImplementedError("Please provide tempering field object for score-scaled ALD sampling.")
        if (joint>0) and (score_scaled):
            raise NotImplementedError("Warning: ALD sampling with joint distribution is not guaranteed to work with tempered version.")
        
        device = next(model.parameters()).device

        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        #Deterministic sequence of sigmas (since diffusion model is trained using these noise scales 50% of time)
        sigmas = self.ALD_sigmas

        #Since EDM corresponds to VE-SDE formulation, the correct initial sample is N(0,sigma_max^2 I).
        dim = int(self.D/2) if (self.joint_model) else self.D
        x_init = sigma_max*torch.randn(n_samples, dim).to(device) #Update: Using uniform here does not impact on the sampling much
        #Perform ALD across noise levels
        x = x_init
        for sigma in sigmas: #assume that sigmas are from high noise to low noise
            alpha_base = epsilon*(sigma**2)/(sigma_max**2) #note that this line is incorrect in Song et al., 2019
            for step in range(T):
                tau_attr = getattr(tempfield, "tau", None)
                if callable(tau_attr):
                    tau = tau_attr(x[:, :dim]).unsqueeze(1)
                else:
                    tau = tempfield #If contant tempring field
                if (self.joint_model):
                    x_model_input = torch.zeros(n_samples, self.D).to(device)
                    x_model_input[:,:int(dim)] = x
                    x_model_input[:,int(dim):] = sigma*torch.randn(n_samples, int(dim)).to(device) #force irrelevant x_loser to be noise (i.e. masking)
                    score = model(x_model_input, sigma*torch.ones((n_samples,1)), joint=0, temp=0)[:,:int(dim)]
                    #print(score.norm(dim=1)) #scores in two moons are about tens time smaller than in e.g. 4D mixute gaussians or onemoon, that is why step size in ALD should be higher in twomoons
                else:
                    score = model(x, sigma*torch.ones((x.shape[0],1)), joint=0, temp=0)
                alpha = alpha_base / tau
                noise = torch.randn_like(x)
                x = x + alpha * tau * score + torch.sqrt(2.0*alpha) * noise
        
        return x
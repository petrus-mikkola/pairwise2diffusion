import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

class r(torch.nn.Module):

    """r(x,x') model for the log density ratio: log p(x) - log p(x')"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        #Unnormalized log belief density (f_theta in Appendix C.4)
        self.logp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, X_prime):
        #X is shape (N, D)
        #X_prime is shape (M, D)
        logp_X = self.logp(X) #shape (N, 1)
        logp_X_prime = self.logp(X_prime) #shape (M, 1)
        if self.training: #During training, we need densities ratios between rows of X and X_prime
            #Stabilize optimization by fixing free offset in utility function
            both = torch.cat([logp_X, logp_X_prime], dim=0)   # (N+M, 1)
            offset = both.mean(dim=0, keepdim=True)           # (1, 1)
            logp_X = logp_X - offset
            logp_X_prime = logp_X_prime - offset
            return logp_X - logp_X_prime
        else: #For tau computations, we need all pairwise density ratios
            # By transposing the second term, its shape becomes (1, M).
            # The operation becomes (N, 1) - (1, M).
            # PyTorch broadcasts this to create an (N, M) output matrix where:
            # output[i, j] = logp_X[i] - logp_X_prime[j]
            return logp_X - logp_X_prime.T
        
 
class TemperingField:

    def __init__(self, D, diffusion, MWD, minibatch, loglik, s, MWD_sample_size, r_hidden_dim, quantile=0.99):
        self.diffusion = diffusion
        self.D = D
        self.s = s
        self.r_model = r(input_dim=D, hidden_dim=r_hidden_dim)
        self.MWD_model = MWD
        self.minibatch = minibatch
        self.loglik = loglik
        self.MWD_sample_size=MWD_sample_size
        self.quantile = quantile
        self.tau_quantile = None
        self.tau_mean = None

    def train_r(self, dataset, batchsize, lr, maxiter, weight_decay=1e-3):
        
        optimizer = torch.optim.Adam(self.r_model.parameters(), lr=lr, weight_decay=weight_decay) #with AdamW one needs significantly higher weigh decay

        preference_loss_fn = torch.nn.BCEWithLogitsLoss()
        # --- Training Loop ---
        self.r_model.train()
        for it in tqdm(range(maxiter),disable=True):
            optimizer.zero_grad()
            batch = self.minibatch(dataset,batchsize) #(k, D, batch_size)
            winners =  batch.clone()[0,:,:].transpose(0,1) #(batch_size,D)
            losers =  batch.clone()[1,:,:].transpose(0,1) #(batch_size,D)
            logits = self.r_model(winners,losers) / self.s #BT model noise fix
            labels_batch = torch.ones(batchsize).unsqueeze(1)
            loss = preference_loss_fn(logits, labels_batch)
            #Alternatively, one can explicitly write the BT-model likelihood
            #BTmodel_lik = (1.0 / (1.0 + torch.exp(-logits / self.s))).mean()
            #loss = neqloglik = F.softplus(-logits / self.s).mean()
            loss.backward()
            optimizer.step()

    def sample_MWD(self):
        self.X = self.diffusion.sample_deterministic(self.MWD_model,n_samples=self.MWD_sample_size,joint=0).detach()
        self.log_p_w_X = self.loglik(self.X[:,:self.D], lambda x, t: self.MWD_model(x, t, joint=0, temp=0), sigma_min=self.diffusion.sigma_min, sigma_max=self.diffusion.sigma_max).detach()
        self.X = self.X[:,:self.D]
    
    def initialize(self,dataset,batchsize,lr,maxiter,weight_decay):
        self.train_r(dataset,batchsize,lr,maxiter,weight_decay)
        self.sample_MWD()
        self.tau_quantile = torch.quantile(self.tau(self.X,clamp=False),self.quantile)
        self.tau_mean = torch.mean(self.tau(self.X,clamp=True))
    
    def tau(self, x, clamp=True):
        self.r_model.eval()
        with torch.no_grad():
            log_r = (self.r_model(x, self.X) / self.s)

        eps=1e-8
        sig_pos = torch.sigmoid(-log_r) #In the paper, it reads r(x,x') = p(x')/p(x) not p(x)/p(x')! So we need swap signs here
        sig_neg = torch.sigmoid(log_r)
        #Importance weights. Clipping provides robustness to likelihood computations 
        w = (1.0 / (self.log_p_w_X.exp() + eps)).reshape(1,self.MWD_sample_size) 
        w = torch.clamp(w, min=w.quantile(0.01), max = w.quantile(0.9))
        #Analytical tempering field formula under the Bradley-Terry model (Theorem 3.1)
        nominator   = torch.mean(sig_neg * w, dim=-1) #average over MC-samples with importance weights
        denominator = torch.mean((sig_pos * sig_neg) * w, dim=-1) #average over MC-samples with importance weights
        taux = (self.s * (nominator / (denominator + eps))) #even large jitter does not have much impact here

        taux = torch.nan_to_num(taux, nan=1.0, posinf=self.s*1e6, neginf=1.0) #handle nans/infs from pathological batches
        taux = torch.clamp(taux, min=1.0) #theoretical lower bound
        if clamp:
            taux = torch.clamp(taux, max=self.tau_quantile) #upper bound for numerical stability
        return taux
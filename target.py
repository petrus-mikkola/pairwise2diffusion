import torch
#import normflows as nf
from normflows.distributions.target import Target
from torch.distributions import MultivariateNormal, Distribution, Categorical, Normal, Uniform
import numpy as np
import math

def set_up_problem(target_name,D,tightbounds=False):
    normalize = None
    if target_name == "onemoon":
        D = 2
        bounds = ((-3, 3),) * D
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = OneMoon()
    if target_name == "onegaussian":
        loose_bounds = ((-7, 7),) * D
        tight_bounds = ((-5, 5),) * D
        bounds = tight_bounds if tightbounds else loose_bounds
        uniform = torch.distributions.uniform.Uniform(torch.tensor([b[0] for b in list(bounds)], dtype=torch.float), torch.tensor([b[1] for b in list(bounds)], dtype=torch.float))
        target = OneGaussian(D)
    if target_name == "stargaussian":
        loose_bounds = ((-3, 10),) * D
        tight_bounds = ((0, 7),) * D
        bounds = tight_bounds if tightbounds else loose_bounds
        uniform = torch.distributions.uniform.Uniform(torch.tensor([b[0] for b in list(bounds)], dtype=torch.float), torch.tensor([b[1] for b in list(bounds)], dtype=torch.float))
        target = StarGaussian(D)
    if target_name == "mixturegaussians":
        loose_bounds = ((-6, 6),) * D
        tight_bounds = ((-4, 4),) * D
        bounds = tight_bounds if tightbounds else loose_bounds
        uniform = torch.distributions.uniform.Uniform(torch.tensor([b[0] for b in list(bounds)], dtype=torch.float), torch.tensor([b[1] for b in list(bounds)], dtype=torch.float))
        target = MixtureGaussians(D)
    if target_name == "twomoons":
        D = 2
        bounds = ((-3, 3),) * D
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = TwoMoons()
    if target_name == "ring":
        D = 2
        bounds = ((-3, 3),) * D
        uniform = torch.distributions.uniform.Uniform(bounds[0][0]*torch.ones(2), bounds[1][1]*torch.ones(2))
        target = RingMixture(n_rings=1)
    if target_name == "llm_prior":
        D=8
        original_bounds = ((0.5, 15),(1,52),(0.846154,141.909091),(0.333333,34.066667),(3,35682),(0.692308,1243.333333),(32.54, 41.95,),(-124.35,-114.31))
        def normalize(X,reverse=False):
            X = torch.as_tensor(X, dtype=torch.float32) #not necessarily compatible with prefflow
            if len(X.shape)==3: #assume shape (K,D,N)
                size = (1, D, 1)
            if len(X.shape)==2: #assume shape (N,D)
                size = (1, D)
            mins = torch.tensor([original_bounds[d_][0] for d_ in range(D)]).view(size)
            maxs = torch.tensor([original_bounds[d_][1] for d_ in range(D)]).view(size)
            if not reverse:
                return 2 * ((X - mins) / (maxs - mins)) - 1
            else:
                return ((X + 1) * (maxs - mins) / 2) + mins     
        #new bounds = ((-1,1),) * D
        bounds = original_bounds
        uniform = None
        target = None
    return target, bounds, uniform, D, normalize 




class OneMoon(Target):
    """
    Unimodal two-dimensional distribution
    """

    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0
        self.lognormconstant = torch.tensor([1.1163528836769938]).log()

    def log_prob(self, z):
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((z[:, 0] + 2) / 0.3) ** 2
        )
        return log_prob - self.lognormconstant


class TwoMoons(Target):
    """
    Bi-modal two-dimensional distribution
    """

    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = -0.8043
        self.lognormconstant = torch.tensor([2.2352]).log()

        #bounds = (-3,3)*2
        #sample = uniform.sample(torch.tensor([100000000])).to(device)
        #print(36*target.log_prob(sample).exp().mean()) # = 2.2352! not 1.0!

    def log_prob(self, z):
        """
        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob - self.lognormconstant


class StarGaussian:
    """
    Star-shape distribution
    """
    def __init__(self,d):
        self.d = d
        self.mean = 3 * torch.ones(d)
        self.sigma2 = 1
        self.rho = 0.9  # Ensure rho < 1 for positive semidefiniteness
        self.initialize()
        
    def generate_covariance_matrix(self, sigma2, rho, sign_pattern):
        base_cov = torch.eye(self.d) * sigma2
        # Add off-diagonal correlations based on sign_pattern
        for i in range(self.d):
            for j in range(i + 1, self.d):
                base_cov[i, j] = sign_pattern[i] * sign_pattern[j] * rho * sigma2
                base_cov[j, i] = base_cov[i, j]
        return base_cov
    
    def initialize(self):
        sign_pattern1 = np.ones(self.d)
        sign_pattern2 = np.array([(-1)**i for i in range(self.d)])
        covariance1 = self.generate_covariance_matrix(self.sigma2, self.rho, sign_pattern1)
        covariance2 = self.generate_covariance_matrix(self.sigma2, self.rho, sign_pattern2)
        self.normaldist1 = MultivariateNormal(self.mean,covariance1)
        self.normaldist2 = MultivariateNormal(self.mean,covariance2)

    def log_prob(self, z):
        log_prob = torch.log(self.normaldist1.log_prob(z).exp()/2 + self.normaldist2.log_prob(z).exp()/2)
        return log_prob
    
    def sample(self, num_samples=10**6):
        component_samples = torch.distributions.Categorical(torch.tensor([0.5, 0.5])).sample((num_samples,))
        samples1 = self.normaldist1.sample((num_samples,))
        samples2 = self.normaldist2.sample((num_samples,))
        samples = torch.zeros_like(samples1)
        mask1 = component_samples == 0
        mask2 = component_samples == 1
        samples[mask1] = samples1[mask1]
        samples[mask2] = samples2[mask2]
        return samples


class MixtureGaussians:
    """
    Four squeezed Gaussian components pointing towards origin.
    Most of the 2D projections (crossplots) shows the star-shaped pattern, and the rest cut-star-shaped patter
    """
    def __init__(self, d):
        assert d >= 2, "Dimension must be at least 2"
        self.d = d
        self.radius = 3.0
        self.sigma2_major = 1.0
        self.sigma2_minor = 0.1
        self.num_components = 4
        self.initialize()

    def generate_mean_directions(self):
        # Base direction (here [1, 1, 1, ..., 1])
        base = torch.ones(self.d)
        dirs = [base, -base]
        # Flip every other sign to get two more symmetric variants
        alt = torch.tensor([(-1.0)**i for i in range(self.d)], dtype=torch.float32) #assumes standard precision
        dirs += [alt, -alt]
        return [self.radius * v / torch.norm(v) for v in dirs]

    def construct_orthogonal_basis(self, v):
        v = v / torch.norm(v)
        basis = [v]
        for i in range(self.d):
            e = torch.zeros(self.d)
            e[i] = 1.0
            for b in basis:
                e -= torch.dot(e, b) * b
            norm = torch.norm(e)
            if norm > 1e-6:
                basis.append(e / norm)
            if len(basis) == self.d:
                break
        return torch.stack(basis)

    def generate_covariance_matrix(self, mean_direction):
        basis = self.construct_orthogonal_basis(mean_direction)
        diag = torch.cat([torch.tensor([self.sigma2_major]), 
                          self.sigma2_minor * torch.ones(self.d - 1)])
        cov = basis.T @ torch.diag(diag) @ basis
        return cov
    
    def get_means_and_covariances(self):
        """
        Return stacked means and covariance matrices for MixtureSameFamily.
        - means: (num_components, d)
        - covariances: (num_components, d, d)
        """
        means = torch.stack([comp.mean for comp in self.components], dim=0)
        covariances = torch.stack([comp.covariance_matrix for comp in self.components], dim=0)
        return means, covariances

    def initialize(self):
        self.means = self.generate_mean_directions()
        self.components = []
        for mean in self.means:
            cov = self.generate_covariance_matrix(mean)
            self.components.append(MultivariateNormal(mean, cov))

    def log_prob(self, z):
        log_probs = torch.stack([comp.log_prob(z) for comp in self.components], dim=0)
        return torch.logsumexp(log_probs, dim=0) - np.log(self.num_components)

    def sample(self, num_samples=10000):
        component_ids = Categorical(torch.ones(self.num_components) / self.num_components).sample((num_samples,))
        samples = torch.zeros((num_samples, self.d))
        for i, comp in enumerate(self.components):
            mask = component_ids == i
            n_i = mask.sum()
            samples[mask] = comp.sample((n_i,))
        return samples



class OneGaussian:
    
    def __init__(self,d):
        mean = torch.tensor([2.0*(-1)**(i+1) for i in range(d)])
        covariance = torch.full((d,d),d/15).fill_diagonal_(d/10)
        self.normaldist = MultivariateNormal(mean,covariance)

    def log_prob(self, z):
        log_prob = self.normaldist.log_prob(z)
        return log_prob
    
    def sample(self, num_samples=10**6):
        samples = self.normaldist.sample((num_samples,))
        return samples



class RingMixture(Distribution):
    #Reimplementation of RingMixture, to make it properly normalized over (-3,3)^2 domain
    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, n_rings=2, scale=None, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.n_rings = n_rings
        self.n_dims = 2
        self.scale = 1 / 4 / n_rings  # standard deviation of each ring
        self.radii = torch.tensor([2 * (i + 1) / n_rings for i in range(n_rings)])
        self.weights = torch.ones(n_rings) / n_rings

        self.cat = Categorical(probs=self.weights)
        self.normals = Normal(self.radii, self.scale)
        self.uniform_theta = Uniform(0.0, 2 * math.pi)

    def sample(self, num_samples=10**6):
        sample_shape=(num_samples,)
        shape = torch.Size(sample_shape)
        n = int(torch.tensor(shape).prod().item())

        ring_indices = self.cat.sample((n,))
        radii_sampled = self.normals.sample()[ring_indices]

        theta = self.uniform_theta.sample((n,))
        x = radii_sampled * torch.cos(theta)
        y = radii_sampled * torch.sin(theta)
        return torch.stack([x, y], dim=1).reshape(*shape, 2)

    def log_prob(self, z):
        r = torch.norm(z, dim=-1)
        r = r.clamp(min=1e-8)  # avoid log(0)

        components = []
        for i in range(self.n_rings):
            normal = Normal(self.radii[i], self.scale)
            log_p_r = normal.log_prob(r)
            log_p_theta = -math.log(2 * math.pi)
            log_jacobian = -torch.log(r)
            components.append(log_p_r + log_p_theta + log_jacobian)

        stacked = torch.stack(components, dim=-1)
        log_mix = torch.logsumexp(stacked, dim=-1) - math.log(self.n_rings)
        return log_mix
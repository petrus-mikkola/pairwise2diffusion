import os
import time
import copy

import torch
import numpy as np
import hydra
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

from plotter import Plotter
from target import set_up_problem
from misc import convert_to_ranking, convert_to_ranking_and_change_k
from diffusion import EDM
import phema #EDM type EMA
from likelihood import loglik
from model import EDMToyModel
from tempfield import TemperingField
from metrics import wasserstein_dist, statistics, mmtv
from transformation2uniform import transform, inversetransform, transform_dataset


#run e.g. by command: python main.py --config-name=experiment2d --multirun exp.target=onemoon,ring exp.seed=1,2
method="SCORE-TAU(X)"   #score-based method with full tempring field tau(x)
#method="SCORE-TAUSTAR" #score-based method with constant tempring field


@hydra.main(version_base=None, config_path="conf/experiment")
def main(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
 
    if not cfg.plot.showduringtraining:
        matplotlib.use('Agg')

    ### Device and Precision ###
    torch.set_default_dtype(torch.float64 if cfg.device.precision_double else torch.float32)
    #enable_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    #enable_mps = False #Do not enable as MPS framework when precision is float64
    #device = torch.device('mps' if torch.backends.mps.is_available() and enable_mps else 'cpu')
    device = torch.device(cfg.device.device)

    ### Random seeds ###
    import random
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    random.seed(cfg.exp.seed)

    ### Target belief density###
    target_name = cfg.exp.target
    D = cfg.exp.d
    target, bounds, uniform, D, normalize = set_up_problem(target_name,D,tightbounds = True if cfg.exp.q_dist=="uniform" else False)

    ### Data generation part 1 ###
    if target_name not in ["llm_prior"]:
        target_sample = target.sample(10000)
        target_mean = target_sample.mean(dim=0)
        target_std = target_sample.std(dim=0)
        if cfg.exp.q_dist=="mixture_gaussian":
            if target_name=="mixturegaussians":
                means, covariance_matrix = target.get_means_and_covariances()
                max_std = torch.sqrt(torch.stack([torch.diag(c) for c in covariance_matrix]).max())
                covariance_matrix = torch.stack([torch.eye(D) * max_std**2 for _ in range(len(means))]) #diagonal Gaussian for each component to make problem harder
            else:
                #Assume always two mixture compenents but with mixture probs (e.g. 0.5 + 0.5 = 1.0) one can make this practically single component when using same means
                means = torch.stack([target_mean, target_mean])
                covariance_matrix = target_std * torch.eye(D).unsqueeze(0).repeat(2, 1, 1) #That is, lambda std is twice higher than the target std
            #How much sampling distribution (lambda) variance is higher than the target?
            var_scale = 3
            covariance_matrix = var_scale * covariance_matrix
        elif cfg.exp.q_dist=="uniform":
            means=None
            covariance_matrix=None

    def sample_alternatives(n,k=2,distribution="uniform",means=None,covariance_matrix=None):
        if distribution=="uniform":
            return uniform.sample(torch.tensor([k*n])).to(device)
        elif distribution=="target":
            return target.sample(k*n).to(device)
        elif distribution=="mixture_gaussian":
            component_distribution = torch.distributions.MultivariateNormal(means, covariance_matrix)
            mixing_probs = torch.tensor(cfg.exp.mixture_probs)
            mixing_distribution = torch.distributions.Categorical(mixing_probs)
            target_gaussian = torch.distributions.MixtureSameFamily(mixing_distribution, component_distribution)
            return target_gaussian.sample((k*n,))
        
    def expert_feedback_ranking(alternatives,rum_noise_dist,s):
        k = alternatives.shape[0]
        if rum_noise_dist == "exponential":
            noise = torch.distributions.Exponential(s).sample((k,)).to(device)
        elif rum_noise_dist == "gumbel":
            noise = torch.distributions.Gumbel(torch.tensor([0.0]), torch.tensor([s])).sample((k,)).to(device).squeeze(1)
        elif rum_noise_dist == "normal":
            noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([s])).sample((k,)).to(device).squeeze(1)
        logprobs = target.log_prob(alternatives).to(device) + noise
        _, ranking_inds = torch.sort(logprobs, descending=True)
        return ranking_inds.view(k).to(device)

    def generate_dataset_ranking(n,k,distribution="uniform",means=None,covariance_matrix=None,rum_noise_dist="exponential",s=1.0):
        X = sample_alternatives(1,k,distribution,means,covariance_matrix)
        Y = expert_feedback_ranking(X,rum_noise_dist,s).view(1,k)
        X = X.unsqueeze(2) #add new dimension, which indicates sample index
        if n > 1:
            for i in range(0,n-1):
                alternatives = sample_alternatives(1,k,distribution,means,covariance_matrix)
                X = torch.cat((X,alternatives.unsqueeze(2)),2)
                Y = torch.cat((Y,expert_feedback_ranking(alternatives,rum_noise_dist,s).view(1,k)),0)
        Xdata = convert_to_ranking(X.numpy(),Y.numpy())
        #return X,Y #X.shape = (k,D,N) = (alternatives,space dimensions, number of rankings)
        return torch.from_numpy(Xdata).view(k,-1,n)

    ### Data generation part 2 ###
    n = cfg.data.n
    if target_name in ["llm_prior"]:
        Xdata1 = np.load("data/llm_prior/california_data_set_1_21-04-2024_dataX.npy") #207 rankings
        Ydata1 = np.load("data/llm_prior/california_data_set_1_21-04-2024_dataY.npy", allow_pickle=True)
        Xdata2 = np.load("data/llm_prior/california_data_set_2_22-04-2024_dataX.npy") #13 rankings
        Ydata2 = np.load("data/llm_prior/california_data_set_2_22-04-2024_dataY.npy", allow_pickle=True)
        if cfg.data.k==5: #Basic scenario, matches to k that was used in creating the dataset
            Xdata = convert_to_ranking(np.concatenate((Xdata1,Xdata2), axis=2),np.concatenate((Ydata1,Ydata2), axis=0))
        else:
            Xdata = convert_to_ranking_and_change_k(np.concatenate((Xdata1,Xdata2), axis=2),np.concatenate((Ydata1,Ydata2), axis=0),k=cfg.data.k)
        Xdata = normalize(torch.from_numpy(Xdata)).to(torch.float64 if cfg.device.precision_double else torch.float32)
        variable_names = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
        n = 220
        dataset = Xdata
        ranking = True
    else:
        ranking = True if cfg.data.k > 2 else False
        k = cfg.data.k
        dataset = generate_dataset_ranking(n,k,distribution=cfg.exp.q_dist,means=means,covariance_matrix=covariance_matrix,rum_noise_dist=cfg.exp.rum_noise_dist,s=getattr(cfg.exp, "s_true", cfg.exp.s)) #s=cfg.exp.s)
        if cfg.data.transform2unitcube:
            dataset = transform_dataset(dataset, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))
            target_std_transformed = dataset.permute(0, 2, 1).reshape(-1, D).std(dim=0).mean()

    def minibatch(dataset,batch_size,ranking=False):
        indices = torch.randperm(n)[:batch_size]
        batch = dataset[:,:,indices]
        return batch

    #Initialize diffusion model
    if target_name not in ["llm_prior"]:
        if cfg.data.transform2unitcube:
            sigma_data = target_std_transformed #e.g. 0.28 in twogaussian 4D
        else:
            sigma_data = target_std.mean()
    else:
        sigma_data = 0.333
    if cfg.method.name == "EDM":
        if cfg.model.name == "EDMtoy":
            model = EDMToyModel(in_dim=2*D,num_layers=cfg.model.num_layers,hidden_dim=cfg.model.hidden_dim,sigma_data=sigma_data,use_temp=False).to(device).train().requires_grad_(True)
        if cfg.model.name == "Dit":
            raise NotImplementedError
        sigma_min = cfg.method.sigma_min
        sigma_max = cfg.method.sigma_max
        L = cfg.method.langevin_L
        #Noise schedule corresponding to the EDM time step discretization would be a natural option, but cosine noise schedule gives better results (and mode mixing?)
        def cosine_noise_schedule(sigma_min, sigma_max, L):
            t = torch.linspace(0, 1, steps=L)
            sigmas = sigma_min + 0.5*(sigma_max - sigma_min)*(1 + torch.cos(t * np.pi))
            return sigmas
        ALD_sigmas = cosine_noise_schedule(sigma_min, sigma_max, L)
        diffusion = EDM(2*D,sigma_min,sigma_max,ALD_sigmas=ALD_sigmas,L=L)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        ema_std = cfg.method.ema_std

    #Initialize optimizer
    loss_hist = np.array([])
    batch_size = cfg.optimization.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimization.lr_ref, betas=(0.9, 0.99))

    #Convenience sampler method
    def sample_diffusion(ema,diffusion=diffusion,temp=0):
        with torch.no_grad():
            #diffusionsample = diffusion.sample_stochastic(ema,cfg.plot.nsamples,joint=0,model_temp=temp).cpu().numpy()
            diffusionsample = diffusion.sample_deterministic(ema,cfg.plot.nsamples,joint=0,temp=temp).cpu().numpy()
            diffusionsample = diffusionsample[:,:D] #keep only winner samples
            if target_name in ["llm_prior"]:
                diffusionsample = normalize(diffusionsample,reverse=True)
            return diffusionsample
    
    #Initial sampling
    if target_name not in ["llm_prior"]:
        targetsample = target.sample(cfg.plot.nsamples)
    initial_diffusionsample = sample_diffusion(ema)[:,:D] if target_name != "llm_prior" else (None,None)
    if cfg.data.transform2unitcube:
        initial_diffusionsample = inversetransform(initial_diffusionsample, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))

    #Plotting
    plotter = Plotter(D,bounds)
    if target_name in ["onemoon","twomoons","ring"]:
        xx,yy,zz = plotter.generate_grid(cfg)
        def set_axes():
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)


    #Main training loop: learn the joint distribution p(x_winner,x_loser)
    start = time.time()
    time_training = start
    for it in tqdm(range(cfg.optimization.max_iter),disable=not cfg.plot.progressbar_show):
        
        #Use EDM type learning rate decay schedule?
        if cfg.optimization.lr_iter is not None:
            optimizer.param_groups[0]['lr'] = cfg.optimization.lr_ref / np.sqrt(max(it / cfg.optimization.lr_iter, 1)) #Below Eq. (15) in Karras et al, (NeurIPS2024)
        optimizer.zero_grad()

        batch = minibatch(dataset,batch_size) #(k, D, batch_size)
        winners =  batch.clone()[0,:,:].transpose(0,1) #(batch_size,D)
        losers =  batch.clone()[1,:,:].transpose(0,1) #(batch_size,D)

        if cfg.method.name == "EDM":
        
            x0 = torch.cat([winners, losers], dim=1)
            if cfg.method.sigma_dist=="uniform":
                log_sigma_t = torch.rand((batch_size,)) * (torch.tensor(sigma_max).log() - torch.tensor(sigma_min).log()) + torch.tensor(sigma_min).log()
            if cfg.method.sigma_dist=="lognormal":
                log_sigma_t = cfg.method.P_mean + cfg.method.P_std*torch.randn((batch_size,)) #Karras et al., 2024, Appendix C
            if random.random() < cfg.method.phi: #phi of time: specify first sigmas based on geometric series
                idx = torch.randperm(L)[:batch_size] if L >= batch_size else torch.randint(0, L, (batch_size,))
                log_sigma_t = ALD_sigmas.log()[idx]
            sigma_t = torch.clamp(log_sigma_t.exp().reshape(-1, 1), min=sigma_min, max=sigma_max)
            tilde_x = x0 + sigma_t * torch.randn_like(x0)
            temp = torch.zeros(batch_size,1) #no tempering
            if random.randint(0, 1) == 0: #Coin flip: train marginal p(x_winner)
                joint = torch.zeros(batch_size,1) #0=consider marginal p(x_winner)
                score = diffusion.score_function(tilde_x[:,:D], x0[:,:D], sigma_t)
                tilde_x[:, D:] = sigma_t * torch.randn_like(tilde_x[:, D:]) #explicit masking with noise
                predicted_score = model(tilde_x, sigma_t, joint, temp)[:,:D]
            else: #Coin flip: train joint p(x_winner,x_loser)
                joint = torch.ones(batch_size,1) #1=consider joint p(x_winner,x_loser)
                score = diffusion.score_function(tilde_x, x0, sigma_t)
                predicted_score = model(tilde_x, sigma_t, joint, temp)
            
            #Score-matching loss and weighting
            weight = (sigma_t**2) #Karras et al., 2024, Eq. (15)
            #weight = (sigma_t ** 2 + sigma_data ** 2) / (sigma_t * sigma_data) ** 2 #Karras et al., 2022, leads too high weights
            loss = torch.mean(weight * ((predicted_score - score) ** 2))
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                #Update EMA
                beta = phema.power_function_beta(std=ema_std, t_next=it+1, t_delta=1)
                for p_net, p_ema in zip(model.parameters(), ema.parameters()):
                    p_ema.lerp_(p_net.detach(), 1 - beta)

                loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())
            
        if (it + 1) % cfg.plot.show_iter == 0:
            end = time.time()
            mean_loss = np.mean(loss_hist[-cfg.plot.show_iter:])  # Compute mean of last cfg.plot.show_iter iters
            print(f"{it+1}: loss {mean_loss.item():0.7f} time {(end - start):0.2f}")
            start = end

            if target_name in ["onemoon","twomoons","ring"]:
                plotter.plot_target(target,xx,yy,zz)
                with torch.no_grad():
                    sample_winner = diffusion.sample_deterministic(ema,cfg.plot.nsamples,joint=0,temp=0).cpu().numpy()[:,:D]
                if cfg.data.transform2unitcube:
                    sample_winner = inversetransform(sample_winner, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))
                plt.scatter(sample_winner[:, 0], sample_winner[:, 1], color='blue', s=10, zorder=5)
                #plot_score_field(sample_winner,ema,sigma=sigma_intermediate) #plot intermediate sigma score field
                plt.title('Samples from Learned Winner Density (blue) and Belief Density (heatmap)')
                set_axes()
                plt.show()
            if target_name in ["onegaussian","stargaussian","mixturegaussians","llm_prior"]:
                diffusionsample = sample_diffusion(ema,temp=0)[:,:D] #no tempering yet, so winner distribution
                if cfg.data.transform2unitcube:
                    diffusionsample = inversetransform(diffusionsample, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))
                plotter.plot_dist(diffusionsample)
    
    time_training = time_training - time.time()
    
    ## Compute tempering fields
    time_temperingfield = time.time()
    tempfield = TemperingField(D=D,diffusion=diffusion,MWD=ema,minibatch=minibatch,loglik=loglik,s=cfg.exp.s,MWD_sample_size=2000*D,r_hidden_dim=cfg.model.hidden_dim)
    tempfield.initialize(dataset,batchsize=cfg.optimization.tau_batch_size,lr=cfg.optimization.tau_lr,maxiter=cfg.optimization.tau_max_iter,weight_decay=cfg.optimization.tau_max_weight_decay)
    time_temperingfield = time_temperingfield - time.time()


    if method=="SCORE-TAUSTAR":
        #Optimal scalar tempering estimate (Proposition 3.2)
        X = diffusion.sample_deterministic(model,n_samples=10000,joint=0).detach() #this sample should be from the belief density, but we approximate it here with MWD
        tauX = tempfield.tau(X[:,:D])
        score = lambda x: ema(x, torch.full((x.shape[0], 1), sigma_min).view(x.shape[0], 1), joint=0, temp=0)[:,:D]
        scores = score(X).detach()
        score_norm_sq = (scores**2).sum(dim=1)
        omegaX = score_norm_sq / score_norm_sq.mean()
        tau_star = torch.mean(omegaX * tauX)
        #print("tau^star=" + str(tau_star.item()))
        tau_ALD = tau_star
    elif method=="SCORE-TAU(X)":
        tau_ALD = tempfield
    else:
        raise NotImplementedError

    optional = False #Should we learn diffusion model also for the belief density or are samples enough?
   
    if not optional:

        time_sampling = time.time()
        with torch.no_grad():
            epsilon = cfg.method.langevin_epsilon
            T = cfg.method.langevin_T
            diffusionsample = diffusion.sample_annealed_langevin(ema, cfg.plot.nsamples, joint=0, score_scaled=True, tempfield=tau_ALD, T=T, epsilon=epsilon)
            diffusionsample = diffusionsample[:,:D] #keep only winner samples
        time_sampling = time_sampling - time.time()

        if target_name in ["llm_prior"]:
            diffusionsample = normalize(diffusionsample,reverse=True)
        if cfg.data.transform2unitcube:
            diffusionsample = inversetransform(diffusionsample, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))

    else:
        
        ### [OPTIONAL] ### (not needed for the experiments)
        #Generate synthetic data from the MWD and the 'tempered' MWD
        nsamples = D*2048
        synthetic_data = {}
        with torch.no_grad():
            epsilon = cfg.method.langevin_epsilon
            T = cfg.method.langevin_T
            synthetic_data[1] = diffusion.sample_annealed_langevin(ema, int(nsamples), joint=0, score_scaled=True, tempfield=tempfield, T=T, epsilon=epsilon)
            synthetic_data[0] = diffusion.sample_annealed_langevin(ema, int(nsamples), joint=0, score_scaled=False, T=T, epsilon=epsilon)

        if cfg.method.name == "EDM":
            if cfg.model.name == "EDMtoy":
                model_winner = EDMToyModel(in_dim=D,num_layers=cfg.model.num_layers_marginal,hidden_dim=cfg.model.hidden_dim_marginal,sigma_data=sigma_data,use_temp=True).to(device).train().requires_grad_(True)
            diffusion_winner = EDM(D,sigma_min,sigma_max,ALD_sigmas=None,L=L,joint_model=False) #Langevin sampling no needed anymore, we can use reverse diffusion
            ema_winner = copy.deepcopy(model_winner).eval().requires_grad_(False)

        optimizer = torch.optim.Adam(model_winner.parameters(), lr=cfg.optimization.lr_ref_marginal, betas=(0.9, 0.99))
        batch_size = cfg.optimization.batch_size_marginal

        #Second training loop
        start = time.time()
        for it in tqdm(range(cfg.optimization.max_iter),disable=not cfg.plot.progressbar_show):
            
            #Use EDM type learning rate decay schedule?
            if cfg.optimization.lr_iter_marginal is not None:
                optimizer.param_groups[0]['lr'] = cfg.optimization.lr_ref_marginal / np.sqrt(max(it / cfg.optimization.lr_iter_marginal, 1)) #Below Eq. (15) in Karras et al, (NeurIPS2024)
            optimizer.zero_grad()

            temp = random.choice(list(synthetic_data.keys())) #sample uniformly on MWD and 'tempered' MWD
            batch = synthetic_data[temp][torch.randint(0, synthetic_data[temp].shape[0], (batch_size,)), :] #sample minibatch of points (with replacement) from given tempered dist
            temps = torch.full((batch_size, 1), temp)

            if cfg.method.name == "EDM":
            
                x0 = batch[:,:D] #Consider winners only
                if cfg.method.sigma_dist=="uniform":
                    log_sigma_t = torch.rand((batch_size,)) * (torch.tensor(sigma_max).log() - torch.tensor(sigma_min).log()) + torch.tensor(sigma_min).log()
                if cfg.method.sigma_dist=="lognormal":
                    log_sigma_t = cfg.method.P_mean + cfg.method.P_std*torch.randn((batch_size,)) #Karras et al., 2024, Appendix C
                sigma_t = torch.clamp(log_sigma_t.exp().reshape(-1, 1), min=sigma_min, max=sigma_max)
                tilde_x = x0 + sigma_t * torch.randn_like(x0)
                score = diffusion.score_function(tilde_x[:,:D], x0[:,:D], sigma_t)
                predicted_score = model_winner(tilde_x[:,:D], sigma_t, torch.zeros(batch_size,1), temps)

                loss = torch.mean((sigma_t**2) * ((predicted_score - score) ** 2))  #Karras et al., 2024, Eq. (15)
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_winner.parameters(), 1.0)
                optimizer.step()
                
                #Update EMA
                beta = phema.power_function_beta(std=ema_std, t_next=it+1, t_delta=1)
                for p_net, p_ema in zip(model_winner.parameters(), ema_winner.parameters()):
                    p_ema.lerp_(p_net.detach(), 1 - beta)
        
        with torch.no_grad():
            diffusionsample = sample_diffusion(ema_winner,diffusion_winner,temp=1)
        if cfg.data.transform2unitcube:
            diffusionsample = inversetransform(diffusionsample, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))

    
    ############ Reporting and plotting the results #################
            
    #Record the success of the marginalization (Appendix C.1)
    with torch.no_grad():
        sample_winner_joint = diffusion.sample_deterministic(ema,35000,joint=0.0).cpu().numpy()
        sample_winner_marginal = diffusion.sample_deterministic(ema,35000,joint=1.0).cpu().numpy()
    Wd_marginal = wasserstein_dist(sample_winner_joint[:, :D],sample_winner_marginal[:, :D])

    
    #Experiment name
    def experiment_name():
        terms = list(range(10))
        terms[0] = target_name
        terms[1] = cfg.exp.q_dist
        terms[2] = str(n)
        terms[3] = "maxiter" + str(cfg.optimization.max_iter)
        terms[4] = "bsize" + str(batch_size)
        terms[5] = "s" + str(cfg.exp.s)
        terms[6] = "seed" + str(cfg.exp.seed)
        if cfg.exp.exp_id is None:
            expname = str(D) + "D"
        else:
            expname = cfg.exp.exp_id + "_" + str(D) + "D"
        for t in terms:
            expname += "_" + str(t)
        return expname

    #Save optimized hyperparameters
    def save_hyperparameters_log():
        f = open(os.path.join(output_folder,"hyperparameters_"+ experiment_name() + ".txt"), "w")
        f.write("Hyperparameters \n")
        f.write("tau_mean: " + str(tempfield.tau(tempfield.X,clamp=False).mean())+"\n")
        f.write("tau_min: " + str(tempfield.tau(tempfield.X,clamp=False).min())+"\n")
        f.write("tau_max: " + str(tempfield.tau(tempfield.X,clamp=False).max())+"\n")
        f.write("tau_999quantile: " + str(tempfield.tau_quantile)+"\n")
        f.write("Wasserstein distance between marginal samples with/without joint=True: " + str(Wd_marginal)+"\n")
        f.write("Total time spent in training: " + str(time_training)+"\n")
        f.write("Total time spent in estimating tempering field: " + str(time_temperingfield)+"\n")
        f.write("Total time spent in sampling: " + str(time_sampling)+"\n")
        f.close()
    save_hyperparameters_log()
    
    #Plot loss trajectory
    plt.figure(figsize=(15, 15))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder,"loss_"+ experiment_name() + ".png"), dpi=150)
    plt.show()

    #Report results
    if target_name in ["onegaussian","stargaussian","mixturegaussians","ring","llm_prior"]:
        np.save(os.path.join(output_folder,"diffusionsamples_" + experiment_name() + ".npy"), diffusionsample)
    f = open(os.path.join(output_folder,"results_"+ experiment_name() + ".txt"), "w")
    f.write("Results after the learning has finished.\n")
    f.write("Last iteration loss: " + str(loss.to('cpu').detach().numpy()) +"\n")
    if target_name not in ["llm_prior"]:
        Wd_init = wasserstein_dist(initial_diffusionsample[:cfg.plot.wasserstein_nsamples,:],targetsample[:cfg.plot.wasserstein_nsamples,:])
        f.write("Initial Wasserstein distance between the target and the density model: " + str(Wd_init)+ "\n")
        Wd = wasserstein_dist(diffusionsample[:cfg.plot.wasserstein_nsamples,:],targetsample[:cfg.plot.wasserstein_nsamples,:])
        f.write("Final Wasserstein distance between the target and the density model: " + str(Wd)+ "\n")
        tv_init = mmtv(initial_diffusionsample,targetsample)
        tv = mmtv(diffusionsample,targetsample)
        f.write("Initial mean marginal total variation distance between the target and the density model: " + str(tv_init)+ "\n")
        f.write("Final mean marginal total variation distance between the target and the density model: " + str(tv)+ "\n")
        results = np.array([[Wd_init,Wd],[tv_init,tv]])
        np.save(os.path.join(output_folder,'results.npy'), results)
    if target_name in ["llm_prior"]:
        f.write(str(statistics(diffusionsample,variable_names))+"\n")
    f.close()

    #Plot samples of te density estimate
    plt.figure(figsize=(15, 15))
    if target_name in ["onemoon","twomoons","ring"]:
        plotter.plot_target(target,xx,yy,zz)
        plt.scatter(diffusionsample[:, 0], diffusionsample[:, 1], color='blue', s=10, zorder=5)
        set_axes()
        plt.savefig(os.path.join(output_folder,experiment_name() + ".png"), dpi=150) #pdf produces poor looking aliasing
        plt.show()
    if target_name in ["onegaussian","stargaussian","mixturegaussians"]:
        labels = None
        linewidth = 0.1
        plotter.plot_dist(diffusionsample,targetsample,save=True,path=os.path.join(output_folder,experiment_name() + "_targetdisplayed" + ".png"),linewidth=linewidth,labels=labels)
        plotter.plot_dist(diffusionsample,targetsample,save=True,path=os.path.join(output_folder,experiment_name() + "_targetdisplayed_nomarginal" + ".png"),linewidth=linewidth,marginal_plot_dist2=False,labels=labels)
        plotter.plot_dist(diffusionsample,None,save=True,path=os.path.join(output_folder,experiment_name() + ".png"),labels=labels)
        #Plot samples of the target density
        plotter.plot_dist(targetsample,None,save=True,path=os.path.join(output_folder,"target_" + str(D) + "D" + target_name + ".png"))
        #Plot MWD samples
        MWDsample = sample_diffusion(ema,temp=0)[:,:D]
        if cfg.data.transform2unitcube:
            MWDsample = inversetransform(MWDsample, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))
        plotter.plot_dist(MWDsample,None,save=True,path=os.path.join(output_folder,experiment_name() + "_MWDsamples" + ".png"),labels=labels)
    if target_name in ["llm_prior"]:
        plotter.plot_dist(diffusionsample,None,save=True,path=os.path.join(output_folder,experiment_name() + ".png"),labels=variable_names)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()
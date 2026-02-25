import torch
import numpy as np
from matplotlib import pyplot as plt
import corner
from scipy.stats import gaussian_kde
import seaborn as sns

def plot_score_field(sample,model,sigma):
    x_min, x_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
    y_min, y_max = sample[:, 1].min() - 1, sample[:, 1].max() + 1
    num_arrows = 20
    x_grid = np.linspace(x_min, x_max, num_arrows)
    y_grid = np.linspace(y_min, y_max, num_arrows)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Evaluate score model at grid points
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_eval = torch.tensor([[X[i, j], Y[i, j], 0, 0]]) #Last two coordinates does not matter, as we plot winner marginal
            sigma_input = (torch.full((x_eval.shape[0], 1), sigma)).view(x_eval.shape[0],1)
            temperature = torch.ones(x_eval.shape[0],1)
            y = torch.ones(x_eval.shape[0],1) #compute marginal p(x_winner)
            score = model(x_eval.float(), sigma_input, y, temperature).detach().numpy()
            U[i, j] = score[0, 0]  # first coordinate of score of x_winner
            V[i, j] = score[0, 1]  # second coordinate of score of x_winner
    plt.quiver(X, Y, U, V, color='cyan', angles='xy', scale=80, width=0.005, alpha=0.8, pivot="middle") #Note: good scale value depends on which noise level (sigma) vectorfield is plotted

class Plotter():
    
    """
    Class for handling plotting of preferential datapoints and plotting any distribution
    """

    def __init__(self, d, bounds):
        self.d = d  # Dimension of the distribution
        self.bounds = bounds
    
    def plot_data(self,batch):
        batchX = batch[0]
        batchY = batch[1]
        for i in range(0, batchX.shape[2]):
            if batchY[i]:
                thetaprime = batchX[0,:,i]
                thetaprimeprime = batchX[1,:,i]
            else:
                thetaprime = batchX[1,:,i]
                thetaprimeprime = batchX[0,:,i]
            plt.plot(thetaprime[0], thetaprime[1], marker='+', markersize=15, color='red', linestyle='None', label="Winner")  
            plt.plot(thetaprimeprime[0], thetaprimeprime[1], marker='_', markersize=8, color='orange', linestyle='None', label="Loser")


    def plot_ranking_data(self,batch):
        #batchX = batch[1]
        k,D,N = batch.shape
        markers = [str(mark) for mark in range(1,k+1)]
        for i in range(N):
            for j in range(k):
                x = batch[j,0,i]
                y = batch[j,1,i]
                color = "red" if j==0 else "blue"
                plt.text(x, y, markers[j], color=color, fontsize=25)
    
    def plot_moon(self,target,prefflow,data,cfg):
        xx, yy = torch.meshgrid(torch.linspace(self.bounds[0][0], self.bounds[0][1], cfg.plot.grid_size), torch.linspace(self.bounds[1][0], self.bounds[1][1], cfg.plot.grid_size))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.double().to(cfg.device.device) if cfg.device.precision_double else zz.float().to(cfg.device.device)
        log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
        prob_target = torch.exp(log_prob)
        if prefflow is None:
            plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
            return None
        prefflow.eval()
        log_prob = prefflow.log_prob(zz).to('cpu').view(*xx.shape)
        prefflow.train()
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0
        rectangle_vol = ((self.bounds[0][1]-self.bounds[0][0])/cfg.plot.grid_size)*((self.bounds[1][1]-self.bounds[1][0])/cfg.plot.grid_size)
        probmassinarea = round(100*rectangle_vol*torch.sum(prob).detach().numpy(),1)
        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.detach().numpy())
        plt.contour(xx, yy, prob_target.detach().numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        if data is not None:
            if cfg.data.k > 2:
                self.plot_ranking_data(data)
            else:
                self.plot_data(data) #plot whole data set
        plt.gca().set_aspect('equal', 'box')
        return probmassinarea 

    def plot_dist(self,dist1_samples,dist2_samples=None,save=False,path=None,nbins=500,nlevels=3,linewidth=0.3,marginal_plot_dist2=True,density_marginal=False,labels=None):
        
        if isinstance(dist1_samples, torch.Tensor):
            dist1_samples = dist1_samples.numpy()

        data = dist1_samples
        if dist2_samples is not None:
            if isinstance(dist2_samples, torch.Tensor):
                dist2_samples = dist2_samples.numpy()
            data2 = dist2_samples

        # Use corner to create the initial corner plot framework
        try:
            figure = corner.corner(data,bins=nbins,density=density_marginal) #bins affect only to marginal plots in the diagonal
        except:
            print("Error in plotting corner plot. Input data shape: " + str(data.shape))

        axes = np.array(figure.axes).reshape((self.d, self.d))
        custom_limits = list(self.bounds)

        for i in range(self.d):
            for j in range(i):
                ax = axes[i, j]
                ax.cla() 
                
                x = data[:, j]
                y = data[:, i]
                
                hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, density=True)
                
                X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
                
                ax.pcolormesh(X, Y, hist, shading='auto')

                if dist2_samples is not None:
                    x = data2[:, j]
                    y = data2[:, i]

                    hist2, xedges, yedges = np.histogram2d(x, y, bins=nbins, density=True)

                    x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
                    y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
                    X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
                    max_val = np.max(hist2)
                    levels = np.linspace(0, max_val, nlevels+2)[1:]  # This creates n levels excluding the lowest one
                    ax.contour(X, Y, hist2.T, cmap=plt.get_cmap('cool'), linewidths=linewidth, levels=levels)

        for i in range(self.d):
            for j in range(self.d):
                ax = axes[i, j]
                if i == j:  # Diagonal plots (histograms)
                    ax.set_xlim(custom_limits[i])
                    #ax.set_ylim() #TODO: check this
                    if (dist2_samples is not None) and (marginal_plot_dist2):
                        # Plot dist2's marginal distribution on the diagonal
                        x = data2[:, i]  # Data from dist2 for this dimension
                        # Create histogram of dist2 data on the same axis
                        n, bins, patches = ax.hist(x, bins=nbins, density=density_marginal, histtype='step', color='magenta', linewidth=1, alpha=0.75) #TODO: check this
                elif j < i:  # Lower triangle
                    ax.set_xlim(custom_limits[j])
                    ax.set_ylim(custom_limits[i])

        #manually set labels
        if labels is None:
            labels = [f"x{i}" for i in range(1, self.d + 1)]
        for i in range(self.d):
            for j in range(self.d):
                ax = axes[i, j]
                if j < i:  # Lower triangle
                    if j == 0:
                        ax.set_ylabel(labels[i])
                    if i == self.d - 1:
                        ax.set_xlabel(labels[j])

                if i == j:  # Diagonal
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)
        plt.tight_layout()

        if save:
            plt.savefig(path, dpi=150)
                
        plt.show()
                        

    def generate_grid(self,cfg):
        #Generate grid
        xx, yy = torch.meshgrid(
            torch.linspace(self.bounds[0][0], self.bounds[0][1], cfg.plot.grid_size),
            torch.linspace(self.bounds[1][0], self.bounds[1][1], cfg.plot.grid_size)
        )
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.double().to(cfg.device.device) if cfg.device.precision_double else zz.float().to(cfg.device.device)
        return xx,yy,zz

    def plot_target(self,target,xx,yy,zz,colorbar=True):
        #Plot target in heatmap
        log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
        prob_target = torch.exp(log_prob)
        contour = plt.contourf(xx.cpu().numpy(), yy.cpu().numpy(), prob_target.cpu().numpy(), levels=50, cmap='viridis')
        if colorbar:
            cbar = plt.colorbar(contour,label='Belief density')
            cbar.set_ticks([])  #Hide the numbers on the colorbar, Remove ticks entirely
    
    def kdeplot_density(self,sample,logliks,colorbar=True):
        x_np = sample.cpu().numpy()
        logliks_np = logliks.cpu().numpy()
        x1, x2 = x_np[:, 0], x_np[:, 1]
        plt.figure(figsize=(8, 6))
        kde = sns.kdeplot(x=x1,y=x2,weights=np.exp(logliks_np),cmap="viridis",fill=True)
        if colorbar:
            mappable = kde.collections[0]
            plt.colorbar(mappable, label="Density")
                
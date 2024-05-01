import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def split_traintest(n_t, frac=0.25, pad=3, split_time=False):
    """this returns deterministic split of train and test in time chunks
    
    Parameters
    ----------
    n_t : int
        number of timepoints to split
    frac : float (optional, default 0.25)
        fraction of points to put in test set
    pad : int (optional, default 3)
        number of timepoints to exclude from test set before and after training segment
    split_time : bool (optional, default False)
        split train and test into beginning and end of experiment
    Returns
    --------
    itrain: 2D int array
        times in train set, arranged in chunks
    
    itest: 2D int array
        times in test set, arranged in chunks
    """
    #usu want 10 segs, but might not have enough frames for that
    n_segs = int(min(10, n_t/4)) 
    n_len = int(np.floor(n_t/n_segs))
    inds_train = np.linspace(0, n_t - n_len - 5, n_segs).astype(int)
    if not split_time:
        l_train = int(np.floor(n_len * (1-frac)))
        inds_test = inds_train + l_train + pad
        l_test = np.diff(np.stack((inds_train, inds_train + l_train)).T.flatten()).min() - pad
    else:
        inds_test = inds_train[:int(np.floor(n_segs*frac))]
        inds_train = inds_train[int(np.floor(n_segs*frac)):]
        l_train = n_len - 10
        l_test = l_train
    itrain = (inds_train[:,np.newaxis] + np.arange(0, l_train, 1, int))
    itest = (inds_test[:,np.newaxis] + np.arange(0, l_test, 1, int))
    return itrain, itest

def gabor_wavelet(sigma, f, ph, n_pts=201, is_torch=False):
    x = np.linspace(0, 2*np.pi, n_pts+1)[:-1].astype('float32')
    cos = np.cos
    sin = np.sin
    exp = np.exp
    xc = x - x.mean()
    cosine = cos(ph + f * xc)
    gaussian = exp(-(xc**2) / (2*sigma**2))
    G = gaussian * cosine
    G /= (G**2).sum()**0.5
    return G

class Core(nn.Module):
    """ linear -> conv1d -> relu -> linear -> relu = latents for KPN model"""
    def __init__(self, n_in=28, n_kp=None, n_filt=10, kernel_size=201, 
                 n_layers=1, n_med=50, n_latents=50, 
                 identity=True, relu_wavelets=True, relu_latents=True):
        super().__init__()
        self.n_in = n_in
        self.n_kp = n_in if n_kp is None or identity else n_kp
        self.n_filt = (n_filt//2) * 2 # must be even for initialization
        self.relu_latents = relu_latents
        self.relu_wavelets = relu_wavelets
        self.n_layers = n_layers
        self.n_latents = n_latents
        self.features = nn.Sequential()

        # combine keypoints into n_kp features
        if identity:
            self.features.add_module('linear0', nn.Identity(self.n_in))
        else:
            self.features.add_module('linear0', nn.Sequential(nn.Linear(self.n_in, self.n_kp),
                                                              ))
        # initialize filters with gabors
        f = np.geomspace(1, 10, self.n_filt//2).astype('float32')
        gw0 = gabor_wavelet(1, f[:,np.newaxis], 0, n_pts=kernel_size)
        gw1 = gabor_wavelet(1, f[:,np.newaxis], np.pi/2, n_pts=kernel_size)
    
        # compute n_filt wavelet features of each one => n_filt * n_kp features
        self.features.add_module('wavelet0', nn.Conv1d(1, self.n_filt, kernel_size=kernel_size,
                                                    padding=kernel_size//2, bias=False))
        self.features[-1].weight.data = torch.from_numpy(np.vstack((gw0, gw1))).unsqueeze(1)
    
        for n in range(1, n_layers):
            n_in = self.n_kp * self.n_filt if n==1 else n_med
            self.features.add_module(f'linear{n}', nn.Sequential(nn.Linear(n_in, 
                                                                            n_med),
                                                                 ))

        # latent linear layer
        if self.n_latents > 0:
            n_med = n_med if n_layers > 1 else self.n_filt * self.n_kp
            self.features.add_module('latent', nn.Sequential(nn.Linear(n_med, n_latents),
                                                        ))
        
    def wavelets(self, x):
        """ compute wavelets of keypoints through linear + conv1d + relu layer """
        # x is (n_batches, time, features)
        out = self.features[0](x.reshape(-1, x.shape[-1]))
        out = out.reshape(x.shape[0], x.shape[1], -1).transpose(2,1)
        # out is now (n_batches, n_kp, time)
        out = out.reshape(-1, out.shape[-1]).unsqueeze(1)
        # out is now (n_batches * n_kp, 1, time)
        out = self.features[1](out)
        # out is now (n_batches * n_kp, n_filt, time)
        out = out.reshape(-1, self.n_kp * self.n_filt, out.shape[-1]).transpose(2,1)
        out = out.reshape(-1, self.n_kp * self.n_filt)
        
        if self.relu_wavelets:
            out = F.relu(out)
        
        # if n_layers > 1, go through more linear layers
        for n in range(1, self.n_layers):
            out = self.features[n+1](out)
            out = F.relu(out)
        return out
                                              
    def forward(self, x=None, wavelets=None):
        """ x is (n_batches, time, features)
            sample_inds is (sub_time) over batches
        """
        if wavelets is None:
            wavelets = self.wavelets(x)
        wavelets = wavelets.reshape(-1, wavelets.shape[-1])
        
        # latent layer
        if self.n_latents > 0:
            latents = self.features[-1](wavelets)
            latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
            if self.relu_latents:
                latents = F.relu(latents)
            latents = latents.reshape(-1, latents.shape[-1])
            return latents
        else:
            return wavelets

class Readout(nn.Module):
    """ linear layer from latents to neural PCs or neurons """
    def __init__(self, n_latents=256, n_layers=1, 
                n_med=128, n_out=128):
        super().__init__()
        self.linear = nn.Sequential()
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.linear.add_module("linear", nn.Linear(n_latents, n_out))
        self.bias.requires_grad = False

    def forward(self, latents):
        return self.linear(latents) + self.bias
        
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        m.bias.data.fill_(0.01)

class PredictionNetwork(nn.Module):
    """ predict from n_in to n_out """
    def __init__(self, n_in=1, n_kp=None, n_filt=10, kernel_size=201, n_core_layers=1,
                 n_latents=20, n_out_layers=1, n_out=128, n_med=50,
                 identity=True, relu_wavelets=True, relu_latents=True):
        super().__init__()
        self.core = Core(n_in=n_in, n_kp=n_kp, n_filt=n_filt, kernel_size=kernel_size, 
                         n_layers=n_core_layers, n_med=n_med, n_latents=n_latents, 
                         identity=identity, relu_wavelets=relu_wavelets, relu_latents=relu_latents)
        self.readout = Readout(n_latents=n_latents if n_latents > 0 else n_filt*n_kp, 
                               n_layers=n_out_layers, n_out=n_out)
        self.apply(init_weights)

    def forward(self, x, sample_inds=None):
        latents = self.core(x)
        if sample_inds is not None:
            latents = latents[sample_inds]
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
        y_pred = self.readout(latents)
        return y_pred, latents
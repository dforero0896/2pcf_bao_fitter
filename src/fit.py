#!/usr/bin/env python
import zeus
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.special import spherical_jn
from scipy.interpolate import interp1d
import pymultinest as pmn
import hankl
import os
from mpi4py import MPI

class BAO_Fitter():
    def __init__(self, plin_fn, plin_nw_fn, pnw_temp_fn=None, pratio=1, a_smooth=1,
                k_norm=8e-3, smin=60, smax=150, sbin_size=5, kmin = 2.5e-3, kmax = 600, num_lnk_bin=2048,
                prior_params = ((0.8, 1.2), (0, 25), (0,30)), prior_types=('flat', 'flat', 'flat'), 
                nwalkers = 10, tolerance = 0.01, live_points = 1000, parameters = ['a', 'B', 'Snl'], 
                backend = 'multinest'):

        print(f"==> Initializing fitter.", flush=True)
        self.backend = backend
        
        self.parameters = parameters
        self.prior_params = prior_params
        self.nparams = len(prior_params)
        self.prior_types = prior_types
        self.nwalkers = nwalkers or 3*self.nparams
        self.live_points = live_points
        self.tolerance=tolerance
        self.kmin, self.kmax = kmin, kmax
        self.num_lnk_bin = num_lnk_bin
        self.lnk = np.linspace(np.log(kmin), np.log(kmax), self.num_lnk_bin)
        self.k = np.exp(self.lnk)
        self.plin_nw = self.get_pk_array(plin_nw_fn)
        self.plin = self.get_pk_array(plin_fn)
        if pnw_temp_fn is not None:
            self.pnw_temp = self.get_pk_array(pnw_temp_fn)
            self.pratio = self.pnw_temp / self.plin_nw
        else:
            self.pratio = pratio # Can be constant (for galaxies) or a function of k (passed as array of matching k binning)
        self.a = a_smooth
        self.asq = self.a**2
        self.ksq = self.k**2
        self.exp_ksq_asq = np.exp(-self.ksq * self.asq)
        self.j0 = lambda z: spherical_jn(0, z)
        self.smin=smin
        self.smax=smax
        self.sbin_size = sbin_size
        self.s_edges = np.arange(self.smin, self.smax+self.sbin_size, self.sbin_size)
        self.s = 0.5 * (self.s_edges[1:] + self.s_edges[:-1])
        self.fit_range_mask = None
        self.nuisance_s = self.s[:, None]**(-np.arange(3)[None, ::-1]) # s**-2, s**-1, s**-0 dimensions nbins, nfeatures
        self.k_norm = k_norm
        
        pnw_norm = np.mean(self.plin[self.k < self.k_norm] / self.plin_nw[self.k < self.k_norm])
        self.plin_nw*=pnw_norm



    def get_xi_fit_range(self, xi_obs, s_obs, match_bins=True):
        if match_bins:
            self.fit_range_mask = (s_obs > self.smin) & (s_obs < self.smax)
            return np.squeeze(xi_obs[self.fit_range_mask])
        else:    
            xi_interp = interp1d(s_obs, s_obs**2*xi_obs, kind='cubic', bounds_error=True, fill_value=0)
            return np.squeeze(xi_interp(self.s)/self.s**2)
        
        

    def get_pk_array(self, filename, usecols=(0,1)):
        data = pd.read_csv(filename, delim_whitespace=True, usecols=usecols, names=['k', 'pk'])
        log_pk_interp = interp1d(np.log(data['k'].values), np.log(data['pk'].values), kind='cubic', bounds_error=False, fill_value='extrapolate')
        return np.exp(log_pk_interp(np.log(self.k)))



    def compute_covariance(self, mocks, fit_range_mask=None, usecols=(1,)):
        if isinstance(mocks, str):
            mock_list = mocks
            mocks = []
            with open(mock_list, 'r') as f:
                for line in f:
                    fname = line.rstrip("\n")
                    if fname =='': continue
                    mocks.append(fname)
        
        if self.fit_range_mask is None:
            s = pd.read_csv(mocks[0], delim_whitespace=True, engine='c', names = ['s'], usecols = (0,)).values
            self.fit_range_mask = np.squeeze((s > self.smin) & (s < self.smax))

        xis = np.squeeze(np.array([pd.read_csv(f, delim_whitespace=True, engine='c', names = ['xi0'], usecols = usecols).values[self.fit_range_mask] for f in mocks]))
        

        N_mocks, N_bins = xis.shape

        error = xis - xis.mean(axis=0)[None, :]

        sample_cov = error.T.dot(error)

        cov_unbiased = sample_cov / (N_mocks - N_bins - 2)

        

        self.covariance = cov_unbiased
        self.inv_covariance = np.linalg.inv(self.covariance)



    def template_xi(self, s_arr, Sigma_nl):
        template_pk = ((self.plin - self.plin_nw) * np.exp(- 0.5 * self.ksq * Sigma_nl**2) + self.plin_nw) * self.pratio
        s_long, xi_long = hankl.P2xi(self.k, template_pk, 0, ext=3, lowring=True)
        xi_template_interp  = interp1d(s_long, xi_long, kind='cubic', bounds_error=False, fill_value='extrapolate')
        xi = np.real(xi_template_interp(s_arr))



        return xi
    

    def get_chisq(self, xi_obs, alpha, B, Sigma_nl, s_arr):

        # Use leastsq to solve for nuisance parameters
        xi_model = self.template_xi(s_arr, Sigma_nl)
        xi_model_interpolator = interp1d(s_arr, xi_model, kind='cubic', bounds_error=True, fill_value=0) # Define interpolator with whole s range
        xi_model_shift = xi_model_interpolator(self.s * alpha) # Interpolate only with fitting range
        A_obs = xi_obs[self.fit_range_mask] - xi_model_shift * B**2
        
        design_matrix = self.nuisance_s.T.dot(self.nuisance_s) 
        
        vector = self.nuisance_s.T.dot(A_obs)
        a_params = np.linalg.solve(design_matrix, vector)

        error = A_obs - a_params.dot(self.nuisance_s.T)

        chisq = error.T.dot(self.inv_covariance.dot(error))

        return chisq

    def best_fit(self, xi_obs, alpha, B, Sigma_nl, s_arr):

        # Use leastsq to solve for nuisance parameters
        xi_model = self.template_xi(s_arr, Sigma_nl)
        xi_model_interpolator = interp1d(s_arr, xi_model, kind='cubic', bounds_error=False, fill_value=np.nan)
        xi_model_shift = xi_model_interpolator(self.s * alpha)
        A_obs = xi_obs[self.fit_range_mask] - xi_model_shift * B**2
        
        design_matrix = self.nuisance_s.T.dot(self.nuisance_s)
        
        vector = self.nuisance_s.T.dot(A_obs)
        a_params = np.linalg.solve(design_matrix, vector)
        #xi_model_interpolator = interp1d(s_arr, xi_model, kind='cubic', bounds_error=False, fill_value=np.nan)


        best = B**2 * xi_model_interpolator(alpha * s_arr) + a_params.dot((s_arr[:, None]**(-np.arange(3)[None,::-1])).T)



        mask = np.isnan(best)
        


        return s_arr[~mask], best[~mask] #why not s_arr*alpha?



    def logprior(self, theta):

        lp = 0
        for i in range(self.nparams):
            if self.prior_types[i] == 'flat':
                lp -= 0 if self.prior_params[i][0] < theta[i] < self.prior_params[i][1] else np.inf
            elif self.prior_types[i] == 'gauss':
                mu, sigma = self.prior_params[i]
                lp -= 0.5 * ((theta[i] - mu) / sigma)**2
            else:
                lp = -np.inf

            
        return lp 
    
    def loglike(self, theta, data, s):

        alpha, B, Sigma_nl = theta

        chisq = self.get_chisq(data, alpha, B, Sigma_nl, s)

        return - 0.5 * chisq

    def logpost(self, theta, data, s):
        
        # Use Bayes' rule
        lp =  self.logprior(theta)

        if ~np.isfinite(lp):            
            return -np.inf
                
        return lp + self.loglike(theta, data, s)


    def fit(self, data, s, nsteps = 100, outbase="output/test_", seed=42, resume=True):
        
        self.nsteps = nsteps
        
        if self.backend =='zeus':
            param_init = []
            for i in range(self.nparams):
                if self.prior_types[i] == 'flat':
                    min, max = self.prior_params[i]
                    param_init.append((max-min) * np.random.random(self.nwalkers) + min)
                elif self.prior_types[i] == 'gauss':
                    mu, sigma = self.prior_params[i]
                    param_init.append(np.random.normal(loc = mu, scale = sigma, size=self.nwalkers))
                else:
                    raise NotImplementedError

            start = np.array(param_init).T
            if not os.path.isfile(outbase+"zeus_chains.txt") or ~resume:
                np.random.seed(seed)
                with Pool() as pool:
                    sampler = zeus.EnsembleSampler(self.nwalkers, self.nparams, self.logpost, args=[data, s], vectorize=False) # Initialise the sampler
                    sampler.run_mcmc(start, nsteps) # Run sampling
                sampler.summary # Print summary diagnostics
                return sampler
            else:
                print(f"==> Chain file found, skipping the fit.")

            
        elif self.backend == 'multinest':

            print(f"==> Using PyMultinest. Assuming params passed are flat priors.")

            
            def prior_pmn(cube, ndim, nparams):
                for i in range(self.nparams):
                    min, max = self.prior_params[i]
                    cube[i] = cube[i] * (max - min) + min

            def loglike_pmn(cube, ndim, nparams):
                alpha, B, Snl = cube[0], cube[1], cube[2]
                return self.loglike((alpha, B, Snl), data, s)

            
            
            pmn.run(loglike_pmn, prior_pmn, self.nparams, outputfiles_basename=outbase, resume=resume, \
                verbose=True, n_live_points=self.live_points, evidence_tolerance=self.tolerance)

        else:
            raise NotImplementedError

    def analyze(self, outbase = "output/test_", sampler = None, burnin_factor = 0.5, resume=True):
        if self.backend == 'zeus':
            
            if not os.path.isfile(outbase+"zeus_chains.txt") or ~resume:
                if sampler is None:
                    raise ValueError
                print(f" ==> Getting chains. thinning the chain and removing burn-in with factor {burnin_factor}")
                chain = sampler.get_chain(flat=True, discard=burnin_factor, thin=10)
                np.savetxt(outbase+"zeus_chains.txt", chain)
            else:
                chain = np.loadtxt(outbase+"zeus_chains.txt")
            # Compute MAP estimate and 1-sigma quantiles
            olist = []
            bestlist = []
            for i in range(self.nparams):
                mcmc = np.percentile(chain[:, i], [16, 50, 84])
                olist.append(mcmc[1])
                bestlist.append(mcmc[1])
                q = np.diff(mcmc)
                olist.append(q[0])
                olist.append(q[1])
            self.write_line(outbase+"zeus_map.txt", np.array(olist))

            fig, axes = zeus.cornerplot(chain, labels=self.parameters, truth=[1, 0.75, 8], title_fmt='.4f')
            fig.savefig(f"{outbase}zeus_marginals_post.png") #, bbox_inches='tight')

            return np.array(bestlist)

        elif self.backend == 'multinest':
            print(f"==> No burnin used with Multinest backend.")

            res = pmn.Analyzer(outputfiles_basename=outbase, n_params=self.nparams)
            bestpar = res.get_best_fit()['parameters']
            stats = res.get_stats()['marginals']
            
            p = pmn.PlotMarginalModes(res)
            plt.figure(figsize=(5*self.nparams, 5*self.nparams))
            olist = []
            for i in range(self.nparams):
                median = stats[i]['median']
                onesigma = stats[i]['1sigma']
                low = median - onesigma[0]
                high = onesigma[1] - median
                olist.append(median)
                olist.append(stats[i]['sigma'])
                plt.subplot(self.nparams, self.nparams, self.nparams * i + i + 1)
                p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
                
                plt.title(f"${self.parameters[i]} = {stats[i]['median']:.4f}_{{-{low:.4f}}}^{{-{high:.4f}}}$")
                plt.ylabel("Probability")
                plt.xlabel(self.parameters[i])
                
                for j in range(i):
                    plt.subplot(self.nparams, self.nparams, self.nparams * j + i + 1)
                    #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
                    p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
                    plt.xlabel(self.parameters[i])
                    plt.ylabel(self.parameters[j])
            self.write_line(outbase+"multinest_map.txt", np.array(olist))
            plt.savefig(f"{outbase}multinest_marginals_post.png") #, bbox_inches='tight')
            return bestpar

        else: 
            raise NotImplementedError

    def fit_analyze(self, data, s, nsteps = 100, outbase="output/test_", seed=42, burnin_factor = 0.5, resume=True, plot_bf=True):
        
        if self.fit_range_mask is None:
            self.get_xi_fit_range(data,s)
        
        sampler = self.fit(data, s, nsteps = nsteps, outbase=outbase, seed=seed, resume=resume)
        best_parameters = self.analyze(outbase=outbase, sampler=sampler, burnin_factor=burnin_factor, resume=resume)
        

        if plot_bf:
            best_s, best_f = self.best_fit(data, best_parameters[0], best_parameters[1], best_parameters[2], s)
            fig, ax = plt.subplots(1, 1)
            ax.plot(s, s**2*data, label='Data')
            ax.plot(best_s, best_s**2*best_f, label='Best-fit')
            ax.legend(loc=0)
            ax.set_xlabel('$s$ [Mpc/$h$]', fontsize=13)
            ax.set_ylabel(r'$s^2\xi_0$', fontsize=13)
            plt.tight_layout()
            fig.savefig(f"{outbase}bestfit.png")
            fig.clear()
            plt.close(fig)

        return best_parameters


    def batch_fit_analyze(self, in_file_list, nsteps, outdir, seed=42, burnin_factor=0.5, resume=True, plot_bf=True):
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        my_infile_list = np.array_split(in_file_list, size)[rank]
        for i, f in enumerate(my_infile_list):
            data = pd.read_csv(f, delim_whitespace=True, usecols=(0,1), names = ['s', 'xi0'])
            basename = f"{os.path.basename(f)}"
            ind_outdir = f"{outdir}/{basename}"
            os.makedirs(ind_outdir, exist_ok=True)
            
            if len(ind_outdir) > 80 :
                print(f"WARNING: The length of the filename is probably too long for buffer, creating symbolic links instead.", flush=True)
                backup_odir = f"run/{basename}"
                os.makedirs("run", exist_ok=True)
                os.makedirs(backup_odir, exist_ok=True)
                os.symlink(ind_outdir, backup_odir)
                ind_outdir = backup_odir

                
            outbase = f"{ind_outdir}/{basename}_"

            best_params = self.fit_analyze(data = data['xi0'].values, s = data['s'].values, nsteps=nsteps, 
                                            outbase = outbase, seed = i + rank, burnin_factor=burnin_factor, 
                                            resume=resume, plot_bf=plot_bf)


        return

    def write_line(self, filename, array):
        fmt = " ".join(["%.6f"]*len(array))
        
        with open(filename, "w") as f:
            f.write(fmt%tuple(array))

if __name__ == '__main__':

    import argparse
    import os
    import glob
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('-plin', '--linear-pk', dest='plin', type=str, default='data/Albert_Plin.dat')
    parser.add_argument('-pnw', '--non-wiggle-pk', dest='pnw', type=str, default='data/Albert_Pnw.dat')
    parser.add_argument('-ptemp', '--template-pk', dest='ptemp', type=str, default='data/PKvoid_template_case1.pspec')
    parser.add_argument('-ifile', '--input-2pcf', dest='ifile', type=str, default='test/mocks_void/TwoPCF_CATALPTCICz0.466G960S1005638091.VOID.R-15.6-50.dat')
    parser.add_argument('-mocks', '--mocks', dest='mocks', type=str, default='test/mocks_void/')
    args = parser.parse_args()
    # Import data to fit
    if os.path.isdir(args.mocks):
        mocks = glob.glob(f"{args.mocks}/TwoPCF*")
    s_obs, xi_obs = np.loadtxt(args.ifile, usecols=(0,1), unpack=True)

    # Initialize fitter
    bao_fitter = BAO_Fitter(plin_fn = args.plin, plin_nw_fn=args.pnw, pnw_temp_fn = args.ptemp, 
                            prior_params = ((0.8, 1.2), (0, 25), (0,30)), 
                            prior_types=('flat', 'flat', 'flat'), 
                            backend = 'zeus')
    # Initialize (compute) covariance matrix if necessary, else set attributes with an appropriate one
    bao_fitter.compute_covariance(mocks)

    best_params = bao_fitter.fit_analyze(xi_obs, s_obs, nsteps=500, resume=True)
    # Zeus backend: 4m52s 500 steps
    # Zeus backend: 17m31s 2000 steps
    # Multinest backend 3m22s
    



    
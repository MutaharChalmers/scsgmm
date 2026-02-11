#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import sklearn.mixture as skm
from tqdm.auto import tqdm

class GaussianMixture(skm.GaussianMixture):
    """Subclass of `sklearn.mixture.GaussianMixture` with conditional sampling.
    """

    def __init__(self, n_components=1, covariance_type='full', tol=0.001,
                 reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        """Same arguments as the superclass."""
        super().__init__(n_components=n_components, 
                         covariance_type=covariance_type, tol=tol,
                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
                         init_params=init_params, weights_init=weights_init,
                         means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start=warm_start,
                         verbose=verbose, verbose_interval=verbose_interval)

    def fit(self, X, y=None):
        """Same arguments as superclass. Convert different shapes of
        self.covariances_ to single consistent shape."""
        super().fit(X, y)

        if self.covariance_type == 'full':
            self.covmats_ = self.covariances_
        elif self.covariance_type == 'tied':
            self.covmats_ = np.tile(self.covariances_, (self.n_components,1,1))
        elif self.covariance_type == 'diag':
            self.covmats_ = np.stack([np.diag(d) for d in self.covariances_])
        else: # 'spherical'
            self.covmats_ = np.stack([c*np.eye(self.means_.shape[1])
                                      for c in self.covariances_])
        return self

    def conditional_sample(self, x_cond, dims_cond, lam=1e-6, seed=None):
        """Conditional random sampling from the fitted model.

        Parameters
        ----------
        x_cond : (m, n) ndarray
            Array of m n-dimensional values to condition on.
        dims_cond : (n,) int ndarray
            Indices of the dimensions which are conditioned on.
        seed : {None, int}, optional
            Seed for numpy.random.default_rng().

        Returns
        -------
        X : (m, d-n) ndarray
            Conditional samples. Total number of dimensions is d.
        """

        # Check that dimensions are consistent
        x_cond = np.atleast_2d(x_cond.T).T
        if x_cond.shape[1] != len(dims_cond):
            print(f'Dimensions of x_cond {x_cond.shape} must be consistent '
                  f'with dims_cond ({len(dims_cond)})')
            return None

        m, n = x_cond.shape

        # Set random number generator
        if seed is not None:
            random_state = np.random.default_rng(seed)
        else:
            random_state = np.random.default_rng()

        # Determine indices of dimensions to be sampled from
        dims_samp = np.setdiff1d(range(self.means_.shape[1]), dims_cond)

        # Subset covariance matrices into blocks
        components = np.arange(self.n_components)
        A = self.covmats_[np.ix_(components, dims_samp, dims_samp)]
        B = self.covmats_[np.ix_(components, dims_samp, dims_cond)]
        C = self.covmats_[np.ix_(components, dims_cond, dims_cond)]

        # Evaluate log-densities at x_cond for all components
        logpdfs = self._mvn_logpdf(x_cond, self.means_[:,dims_cond], C)

        # Convert to probabilities robustly, weight then normalise
        pdfs = np.exp(logpdfs - logpdfs.max(axis=1)[:,None]) * self.weights_
        ps = pdfs/pdfs.sum(axis=1)[:,None]

        # Sample components proportional to normalised pdfs and get indices
        ix = random_state.multinomial(1, ps).argmax(axis=1)

        # Conditional mean and covariance matrices based on Schur complement
        BCinv = np.linalg.solve(C, B.swapaxes(-1, -2)).swapaxes(-1, -2)
        ccovs = A - BCinv @ np.transpose(B, (0,2,1))
        deltas = (x_cond[:,None] - self.means_[:,dims_cond])[:,:,:,None]
        cmus = self.means_[:,dims_samp] + np.squeeze(BCinv @ deltas, -1)

        # Block conditional Cholesky decomposition by GMM component
        cLs = np.linalg.cholesky(ccovs)

        # Select conditional means using fancy indexing
        cmus = cmus[np.arange(m), ix]

        # Generate standard normal z-scores, transform and add to means
        z = random_state.normal(size=(m, dims_samp.size, 1))
        canoms = np.squeeze(cLs[ix] @ z, -1)
        return (cmus + canoms).reshape(m, dims_samp.size)

    def _mvn_logpdf(self, X, mu, cov):
        """Evaluation of k-component multivariate normal log-pdf for GMM.

        Evaluates log-density for all combinations of data points x,
        distribution means mu, and covariance matrices cov.

        Parameters
        ----------
        X : (m, n) ndarray
            Array of m n-dimensional values to evaluate.
        mu : (k, n) ndarray
            Array of k n-dimensional mean vectors.
        cov : (k, n, n) ndarray
            Array of k covariance matrices.

        Returns
        -------
        logpdf : (m, k) ndarray
            Array of log-pdfs.
        """

        # Dimension of MVN
        mu = np.atleast_2d(mu)
        n = mu.shape[1]
    
        # Mahalanobis distance by Cholesky decomposition of covariance matrices
        L = np.linalg.cholesky(cov)
        deltas = X[:,None] - mu
        y = np.squeeze(np.linalg.solve(L, deltas[:,:,:,None]), -1)
        maha2 = np.square(y).sum(axis=2)
    
        # Other terms in the log-pdf
        nlog_2pi = n*np.log(2*np.pi)
        log_det = 2*np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), axis=1)
    
        # Log-pdfs by component
        logpdf = -0.5*(nlog_2pi + log_det + maha2)
        return logpdf

class SCSGMM():
    """Sequential Conditional Sampling from Gaussian Mixture Models (SCS-GMM).
    
    Fit time series models using flexible parametric methods and simulate
    synthetic realisations with optional exogenous forcing.
    """

    def __init__(self, ordern=1, orderx=None, verbose=True):
        """Class constructor.

        Parameters
        ----------
        ordern : int, optional
            Order of model, i.e. longest time lag, for endogenous features.
            Defaults to 1.
        orderx : int, optional
            Order of model, i.e. longest time lag, for exogenous features.
            Should be less than or equal to ordern for current version.
            Defaults to None.
        verbose : bool, optional
            Show tqdm toolbar during fitting and simulation, or not.
            Defaults to True.
        """

        if orderx is not None and orderx > ordern:
            print(f'orderx ({orderx}) should be <= ordern ({ordern})')
            return None
        if ordern < 1:
            print(f'ordern ({ordern}) should be >= 1')
            return None

        self.ordern = ordern
        self.orderx = orderx
        self.verbose = verbose
        self.models = {}

    def fit(self, Xn, depn, Xx=None, depx=None, periods=None, cov_type='full',
            kmax=100):
        """Fit model.

        Parameters
        ----------
        Xn : ndarray
            Training data for endogenous features only. (m, n) 2D array
            of `m` samples and `n` features.
        depn : dict
            Dependency graph of endogenous features on other endogenous
            features. Structure is as follows:
                {(m1, n1): [n1, n2, ..., nj],
                 (m1, n2): [n1, n2, ..., nj],
                 ...,
                 (mi, nj): [n1, n2, ..., nj]}
            for periods `mi`  and endogenous features `nj`.
            Keys must cover all combinations of periods and endogenous
            features - modelled features must depend on something.
        Xx : ndarray, optional
            Exogenous forcing. Defaults to None.
        depx : dict, optional
            Dependency graph of endogenous features on exogenous
            features. Structure is as follows: 
                {(m1, n1): [x1, x2, ..., xk],
                 (m1, n2): [x1, x2, ..., xk],
                 ...,
                 (mi, nj): [x1, x2, ..., xk]}
            for periods `mi`, endogenous `nj` and exogenous `xk`.
            The keys of `depx` need not cover all combinations of periods
            and endogenous features. Defaults to None.
        periods : ndarray, optional
            PeriodID for each time step, so different models can be fit to
            subsets of the data. If None (default), all data used to fit a
            single model, otherwise must be the same length as Xn.shape[0].
        cov_type : str, optional
            Covariance type to use for Gaussian components. Must be one of the
            options for GaussianMixture: 'full', 'tied', 'diag', 'spherical'.
        kmax : int, optional
            Maximum number of mixture components to evaluate.
        """

        # Input validation
        if periods is None:
            periods = np.zeros(Xn.shape[0])
            self.uperiods = {0}
        elif periods.shape[0] == Xn.shape[0]:
            periods = np.array(periods)
            self.uperiods = set([int(p) for p in periods])
        else:
            print('`periods` must be the same length as `Xn.shape[0]`')
            return None

        if depn is None:
            print('Dependency dictionary `depn` must be specified')
            return None

        if Xx is not None:
            if self.orderx is None:
                print('`self.orderx` is None - cannot pass an exogenous forcing `Xx`')
                return None
            if Xx.shape[0] != Xn.shape[0]:
                print('`Xx` must have the same number of rows as `Xn`')
                return None
            if depx is None:
                print('Dependency dictionary `depx` must be specified')
                return None
            mx, nx = zip(*[(m, n) for m, n in depx.keys()])
            if not set(mx).issubset(self.uperiods):
                print(f'Periods `m` in `depx` must be a subset of {self.uperiods}')
                return None
            if not set(nx).issubset(set(range(Xx.shape[1]))):
                print('Variables `n` in `depx` must be a subset of '
                      f'{range(Xx.shape[1])}')
                return None

            self.dx = depx
            self.Nx = Xx.shape[1]
            Xx = np.array(Xx)
        else:
            self.dx = None
            self.Nx = None

        mn, nn = zip(*[(m, n) for m, n in depn.keys()])
        if set(mn) != self.uperiods:
            print(f'Periods `m` in `depn` must match {self.uperiods}')
            return None
        if set(nn) != set(range(Xn.shape[1])):
            print(f'Variables `n` in `depn` must match {range(Xn.shape[1])}')
            return None

        self.dn = depn
        self.Nn = Xn.shape[1]
        Xn = np.array(Xn)

        # Loop over periods
        self.bics = np.full((len(self.uperiods), self.Nn, kmax), np.inf)

        pbar = tqdm(total=len(self.uperiods)*self.Nn, disable=not self.verbose)
        for i, m in enumerate(self.uperiods):
            # Loop over endogenous variables to be modelled
            for j, n in enumerate(range(self.Nn)):
                # Endogenous variables
                lags = range(self.ordern)
                XN = ([np.roll(Xn, -lag, axis=0)[:,self.dn[m,n]] for lag in lags] +
                      [np.roll(Xn[:,[n]], -self.ordern, axis=0)])

                # Exogenous variables
                if Xx is None:
                    XX = []
                else:
                    lags = range(self.ordern-self.orderx, self.ordern+1)
                    XX = [np.roll(Xx, -lag, axis=0)[:,self.dx[m,n]]
                          for lag in lags if self.dx.get((m,n), None) is not None]

                # Select relevant records only
                mask = np.roll(periods, -self.ordern)[:-self.ordern] == m
                X = np.hstack(XX + XN)[:-self.ordern][mask]

                # Fit GMMs
                ks = range(kmax)
                for k in ks:
                    gmm = GaussianMixture(n_components=k+1,
                                          covariance_type=cov_type).fit(X)

                    # Near-singularity check on covmats
                    eigval_min = np.linalg.eigvalsh(gmm.covmats_).min()
                    if eigval_min > gmm.reg_covar:
                        self.bics[i, j, k] = gmm.bic(X)
                    else:
                        break
                        #self.bics[i, j, k] = np.inf
                k = self.bics.argmin(axis=2)[i,j] + 1
                self.models[m,n] = GaussianMixture(n_components=k,
                                                   covariance_type=cov_type
                                                  ).fit(X)
                pbar.update(1)
        pbar.close()

    def simulate(self, Nt, X0, Xx=None, periods=None, seed=42):
        """Simulate from fitted model.

        Parameters
        ----------
        Nt : int
            Number of time steps to simulate.
        X0 : ndarray
            Inital values to be used in the simulation. `X0` must be 3D with
            shape (# batches, model order, # endogenous features).
        Xx : ndarray, optional
            Exogenous forcing to be used in the simulation. `Xx` must be 3D
            with shape (# batches, time steps, # exogenous features).
        periods : ndarray, optional
            PeriodID for each time step, allowing different models to be
            used for subsets of the data. Must be length `Nt`.
            If None (default) all time steps modelled identically.
        seed : {int, `np.random.Generator`, `np.random.RandomState`}, optional
            Seed or random number generator state variable.

        Returns
        -------
        Y : ndarray
            Simulated data.
        """

        # Input validation
        if len(X0.shape) != 3:
            print('`X0` must be 3D')
            return None

        if X0.shape[1:] != (self.ordern, self.Nn):
            print(f'`X0.shape` {X0.shape} must be consistent with'
                   ' (# batches, model order, # of endogenous features)'
                  f' (..., {self.ordern}, {self.Nn})')
            return None
        else:
            X0 = np.array(X0)
        batches = X0.shape[0]

        if Xx is not None:
            if Xx.shape[1:] != (Nt, self.Nx):
                print(f'`Xx.shape` {Xx.shape} must be consistent with'
                      ' (# batches, time steps, # of exogenous features)'
                      f' (..., {Nt}, {self.Nx})')
                return None
            else:
                Xx = np.array(Xx)

        if periods is None:
            periods = np.zeros(Nt, dtype=int)
        elif periods.shape[0] == Nt:
            if set(periods).issubset(self.uperiods):
                periods = np.array(periods)
            else:
                print(f'`periods` has periodIDs not in {self.uperiods}')
                return None
        else:
            print(f'`periods` must be length Nt={Nt} or `None`')
            return None

        # Initialise random number generator
        rng = np.random.default_rng(seed)

        # Initialise output array
        Y = np.zeros(shape=(batches, Nt, self.Nn))
        Y[:,:self.ordern,:] = X0

        # Loop over time steps
        for i in tqdm(range(self.ordern, Nt), disable=not self.verbose):
            m = periods[i]
            # Loop over variables
            for n in range(self.Nn):
                # Define conditioning vector
                if Xx is None:
                    x_cond = np.hstack([Y[:,i-lag,self.dn[m,n]]
                                        for lag in range(self.ordern, 0, -1)])
                else:
                    x_cond_x = [Xx[:,i-lag,self.dx[m,n]]
                                for lag in range(self.orderx, -1, -1)
                                if self.dx.get((m,n), None) is not None]
                    x_cond_n = [Y[:,i-lag,self.dn[m,n]]
                                for lag in range(self.ordern, 0, -1)]
                    x_cond = np.hstack(x_cond_x + x_cond_n)

                # Generate conditional samples
                Y[:,i,n] = self.models[m,n].conditional_sample(x_cond=x_cond,
                                                               dims_cond=range(x_cond.shape[1]),
                                                               seed=rng)[:,0]
        return Y

    def save(self, outpath, model_name, overwrite=False):
        """Save model to disk.

        Parameters
        ----------
        outpath : str
            Outpath.
        model_name : str
            Model name.
        overwrite : bool
            Overwrite data if it exists. Default False.
        """

        # Create directory
        outpath = os.path.join(outpath, model_name)
        try:
            os.makedirs(outpath)
        except:
            if overwrite:
                print('Model file exists; overwriting...')
            else:
                print('Model file exists; aborting...')
                return None

        # Define metadata to recreate the SCS-GMM model and write to file
        meta = {'name': model_name,
                'ordern': self.ordern,
                'orderx': self.orderx,
                'verbose': self.verbose,
                'uperiods': list(self.uperiods),
                'Nn': self.Nn,
                'Nx': self.Nx}

        with open(os.path.join(outpath, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        # Loop over models and write each one to disk
        """for k, v in self.models.items():
            v.save(outpath, '_'.join(map(str, k)), overwrite=True, verbose=False)"""

        # Save dependency graphs after converting to JSONable formats
        dn_str = {'_'.join(map(str, k)): v.tolist() for k, v in self.dn.items()}
        with open(os.path.join(outpath, 'depn.json'), 'w') as f:
            json.dump(dn_str, f)
        if self.orderx is not None:
            dx_str = {'_'.join(map(str, k)): v.tolist() for k, v in self.dx.items()}
            with open(os.path.join(outpath, 'depx.json'), 'w') as f:
                json.dump(dx_str, f)

    def whiten(self, X):
        """ZCA/Mahalanobis whitening.

        Simulated stochastic principal components with a complex dependency
        structure can end up being non-orthogonal. When recombining stochastic
        PCs with their EOFs, the PCs must be orthogonalised. According to
        Kessey et al (2016) "Optimal Whitening and Decorrelation", the optimal
        whitening transformation to minimise the changes from the original data
        is the ZCA/Mahalanobis transformation with the whitening matrix being
        the inverse-square root of the covariance matrix.

        Parameters
        ----------
        X : ndarray
            Array to be whitened of shape (m, n) where m denotes records
            and n features.

        Returns
        -------
        Xw : ndarray
            Whitened version of input array.
        """

        S = np.cov(X.T)
        u, v = np.linalg.eigh(S)
        S_root = v * np.sqrt(np.clip(u, np.spacing(1), np.inf)) @ v.T
        W = np.linalg.inv(S_root)
        return (X @ W.T) * X.std(axis=0)

    def load(inpath, model_name):
        """Load model from disk.
    
        Parameters
        ----------
        inpath : str
            Inpath.
        model_name : str
            Model name.
    
        Returns
        -------
        scsgmm : scsgmm.SCSGMM
            Instance of SCS-GMM model.
        """
    
        # Load metadata
        inpath = os.path.join(inpath, model_name)
        with open(os.path.join(inpath, 'meta.json'), 'r') as f:
            meta = json.load(f)
    
        # Create object and define fitted characteristics
        scsgmm = SCSGMM(ordern=meta['ordern'], orderx=meta['orderx'],
                        verbose=meta['verbose'])
        scsgmm.uperiods = set(meta['uperiods'])
        scsgmm.Nn = meta['Nn']
        scsgmm.Nx = meta['Nx']
    
        # Load dependency matrices
        with open(os.path.join(inpath, 'depn.json'), 'r') as f:
            depn = json.load(f)
        scsgmm.dn = {tuple(map(int, k.split('_'))): np.array(v)
                     for k, v in depn.items()}
        if meta['Nx'] is not None:
            with open(os.path.join(inpath, 'depx.json'), 'r') as f:
                depx = json.load(f)
            scsgmm.dx = {tuple(map(int, k.split('_'))): np.array(v)
                         for k, v in depx.items()}
    
        # Load models
        """for model in os.listdir(inpath):
            if os.path.isdir(os.path.join(inpath, model)):
                m, n = tuple(map(int, model.split('_')))
                scsgmm.models[m, n] = kt.load(inpath, model)"""
    
        return scsgmm

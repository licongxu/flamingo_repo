"""Gaussian chi^2 likelihood for binned D_ell^{yy} with a multipole cut.

Keeps only the LAST ``keep_last`` of the 18 bins (the ell>100 bins) of both the
data vector and the covariance. The covariance is the same ``cov_full_*`` matrix
used by the full 18-bin fit, sliced to its trailing keep_last x keep_last block.
"""
from __future__ import annotations

import os
import time

import numpy as np
from cobaya.likelihood import Likelihood


class YYEllCutGaussianLikelihood(Likelihood):
    data_directory: str = "."
    data_file: str = "Dl_yy_binned_18.txt"
    cov_file: str = "cov_full_Dl_yy_binned_18.npy"
    keep_last: int = 9          # keep the last 9 of 18 bins (ell > 100)
    log_every: int = 20
    _n_eval: int = 0

    def initialize(self):
        D = np.loadtxt(os.path.join(self.data_directory, self.data_file))
        n_all = D.shape[0]
        k = int(self.keep_last)
        self.sl = slice(n_all - k, n_all)
        self.ell_data = D[self.sl, 0]
        self.Dl_obs = D[self.sl, 1]

        C = np.load(os.path.join(self.data_directory, self.cov_file))
        C = np.atleast_2d(C)[self.sl, self.sl]
        if C.shape != (k, k):
            raise ValueError(f"sliced cov shape {C.shape} != ({k},{k})")
        self.covmat = C
        self.inv_covmat = np.linalg.inv(C)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            raise ValueError("Sliced covariance not positive definite.")
        self.logdet_covmat = logdet
        self.log.info(
            "YYEllCutGaussianLikelihood: keep last %d bins, ell=%.0f..%.0f; data=%s cov=%s",
            k, self.ell_data[0], self.ell_data[-1], self.data_file, self.cov_file)
        super().initialize()

    def get_requirements(self):
        return {"Cl_sz": {}}

    def logp(self, **_):
        t0 = time.perf_counter()
        th = self.provider.get_Cl_sz()
        Dl_full = np.asarray(th["1h"], dtype=float) + np.asarray(th.get("2h", 0.0), dtype=float)
        Dl_th = Dl_full[self.sl]
        if Dl_th.shape != self.Dl_obs.shape:
            raise ValueError(f"theory slice {Dl_th.shape} != data {self.Dl_obs.shape}")
        resid = self.Dl_obs - Dl_th
        chi2 = float(resid @ self.inv_covmat @ resid)
        self._n_eval += 1
        if self.log_every and (self._n_eval % self.log_every == 0):
            self.log.info("YYEllCut chi2=%.2f loglike=%.2f dt=%.3fs",
                          chi2, -0.5 * chi2, time.perf_counter() - t0)
        return -0.5 * chi2

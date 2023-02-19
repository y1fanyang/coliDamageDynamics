# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.special as sc
import scipy.special._ufuncs as scu

from scipy.stats._distn_infrastructure import rv_continuous
from scipy.stats._constants import _XMIN, _EULER, _ZETA3, _XMAX, _LOGXMAX

class genbeta2_gen(rv_continuous):
    r"""A Generalized Beta (second kind) continuous random variable.

    %(before_notes)s

    See Also
    --------
    fisk : a special case of either `burr` or `burr12` with ``d=1``
    burr12 : Burr Type XII distribution, a special case of `genbeta2` with ``c=1`` 
    burr : Burr Type III distribution, a special case of `genbeta2` with ``d=1``
    betaprime: a specal case of a special case of `genbeta2` with ``s=1``, ``c=a`` and ``d=b``

    Notes
    -----
    The probability density function for `genbeta2` is:

    .. math::

        f(x, s, c, d) = \frac{s x^{s c-1}}{\beta(c, d)(1+x^s)^{c+d}}

    for :math:`x >= 0` and :math:`s, c, d > 0`, where
    :math:`\beta(c, d)` is the beta function (see `scipy.special.beta`).

    `genbeta2` takes ``s``, ``c``, ``d`` as shape parameters for :math:`s`
    , :math:`c`, :math:`d`.

    References
    ----------
    .. [1] McDonald, J. B. "Some Generalized Functions for the Size Distribution of Income", 
    Econometrica 52 (3): 647–63.
    
    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _pdf(self, x, s, c, d):
        return np.exp(self._logpdf(x, s, c, d))

    def _logpdf(self, x, s, c, d):
        return np.log(s) - np.log(sc.beta(c,d)) + sc.xlogy(c*s - 1, x) + sc.xlog1py(-d-c, x**s)

    #def _cdf(self, x, s, c, d):
    #    return sc.hyp2f1(c, c+d, 1+c, -x**s)*x**(c*s)/c/sc.beta(c,d)

    #def _logcdf(self, x, s, c, d):
    #    return c*s*np.log(x)+np.log(sc.hyp2f1(c, c+d, 1+c, -x**s))-np.log(c)-sc.betaln(c,d)
    
    def _cdf(self, x, s, c, d):
        saturation = np.exp(-sc.log1p(x**(-s)))
        return sc.betainc(c, d, saturation)

    def _logcdf(self, x, s, c, d):
        saturation = np.exp(-sc.log1p(x**(-s)))
        return np.log(sc.betainc(c, d, saturation))
        
    def _sf(self, x, s, c, d):
        return sc.betainc(d, c, np.exp(-sc.log1p(x**s)))

    def _logsf(self, x, s, c, d):
        return np.log(sc.betainc(d, c, np.exp(-sc.log1p(x**s))))
    
    def _ppf(self, q, s, c, d):
        saturation = sc.betaincinv(c,d, q)
        ## implementing e^(1/s*(log(Sat)-log(1-Sat))) with expm1 and log1p
        return 1+np.expm1((np.log(saturation) - sc.log1p(-saturation))/s)
    
    def _munp(self, n, s, c, d):
        ns = 1. * n / s
        return sc.beta(c + ns, d - ns)/sc.beta(c, d)

genbeta2 = genbeta2_gen(a=0.0, name='genbeta2')
"""Univariate Kalman filter and Gaussian log-likelihood."""

from __future__ import annotations

import numpy as np


def KF(X, sigma_epsilon, sigma_eta):
    T = len(X)
    s_eps2 = sigma_epsilon**2
    s_eta2 = sigma_eta**2

    mMstore = np.zeros(T)
    mM_Lstore = np.zeros(T)
    mPstore = np.zeros(T)
    mP_Lstore = np.zeros(T)
    mVstore = np.zeros(T)
    mFstore = np.zeros(T)

    m = 0.0
    P = 100000.0

    for i in range(T):
        m_L = m
        P_L = P + s_eta2
        aF = P_L + s_eps2
        v = X[i] - m_L
        K = P_L / aF
        P = P_L * (1.0 - K)
        m = m_L + K * v
        mMstore[i] = m
        mM_Lstore[i] = m_L
        mPstore[i] = P
        mP_Lstore[i] = P_L
        mVstore[i] = v
        mFstore[i] = aF

    return {
        "mFilter": mMstore,
        "Predict": mM_Lstore,
        "PFilter": mPstore,
        "PPredict": mP_Lstore,
        "mErrors": mVstore,
        "mErrVar": mFstore,
    }


def logL(mErr, mVar):
    mErr = np.asarray(mErr)
    mVar = np.asarray(mVar)
    return np.sum(-0.5 * np.log(mVar) - 0.5 * (mErr * mErr / mVar))


def logL_LL(sigma_epsilon, sigma_eta, Y):
    output = KF(Y, sigma_epsilon, sigma_eta)
    return -logL(output["mErrors"], output["mErrVar"])

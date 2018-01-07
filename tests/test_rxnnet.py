"""
"""

#from rxnnet import network, experiments


#net = network.Network()


#expts_xc = experiments.Experiments()
#expts_jc = experiments.Experiments()
#expts = expts_xc + expts_jc

#pred_xc = net.get_predict(expts_xc, tol_ss=1e-13)
#pred_jc = net.get_predict(expts_jc, tol_ss=1e-13)
#pred = net.get_predict(expts, tol_ss=1e-13)

from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)

import pytest
import numpy as np

from infotopo import predict
reload(predict)


#@pytest.fixture()
def preds():
    Cs = [(2,1), (3,1)]
    ts = [1,2,3]
    x0 = 0
    pids = ['k1','k2']

    def f_xc(p):
        k1, k2 = p   
        return np.array([(k1*C1+k2*C2)/(k1+k2) for C1, C2 in Cs])

    def Df_xc(p):
        k1, k2 = p
        return np.array([[(k2*(C1-C2))/(k1+k2)**2, (k1*(C2-C1))/(k1+k2)**2] 
                         for C1, C2 in Cs])

    pred_xc = predict.Predict(f=f_xc, Df=Df_xc, pids=pids, ydim=len(Cs))

    def f_jc(p):
        k1, k2 = p
        return np.array([(k1*k2)/(k1+k2)*(C1-C2) for C1, C2 in Cs])

    def Df_jc(p):
        (k1, k2), ksum = p, np.sum(p)
        return np.array([[k2**2*(C1-C2)/ksum**2, k1**2*(C1-C2)/ksum**2] 
                         for C1, C2 in Cs])

    pred_jc = predict.Predict(f=f_jc, Df=Df_jc, pids=pids, ydim=len(Cs))

    def f_xt(p):
        k1, k2 = p
        C1, C2 = Cs[0]
        return np.array([(k1*C1+k2*C2)/(k1+k2)*(1-np.exp(-(k1+k2)*t))+\
                         x0*np.exp(-(k1+k2)*t) for t in ts])

    def Df_xt(p):
        (k1, k2), ksum = p, np.sum(p)
        C1, C2 = Cs[0]
        return np.array([[np.exp(-ksum*t)*((C1*(k1**2*t+k2*(np.exp(ksum*t)+k1*t-1))+\
                            k2*C2*(ksum*t-np.exp(ksum*t)+1))/ksum**2-x0*t),
                          np.exp(-ksum*t)*((C2*(k2**2*t+k1*(np.exp(ksum*t)+k2*t-1))+\
                            k1*C1*(ksum*t-np.exp(ksum*t)+1))/ksum**2-x0*t)]
                         for t in ts])

    pred_xt = predict.Predict(f=f_xt, Df=Df_xt, pids=pids, ydim=len(ts))

    return pred_xc, pred_jc, pred_xt


def test_f(preds):
    pred_xc, pred_jc, pred_xt = preds
    p = [1,2]
    assert np.allclose(pred_xc(p), [4/3, 5/3])
    assert np.allclose(pred_jc(p), [2/3, 4/3])
    assert np.allclose(pred_xt(p), [1.2669505755095147, 
                                    1.33002833043111152,
                                    1.333168786927884427])


def test_Df(preds):
    pred_xc, pred_jc, pred_xt = preds
    p = [1,2]
    assert np.allclose(pred_xc.Df(p), [[2/9,-1/9],[4/9,-2/9]])
    assert np.allclose(pred_jc.Df(p), [[4/9,1/9],[8/9,2/9]])
    assert np.allclose(pred_xt.Df(p), [[0.2775411870754044,-0.0391964568019743],
                                       [0.22828139420962888,-0.10422568839814900],
                                       [0.222688437037660790,-0.110603759694310317]])


def test_spectrum(preds):
    pred_xc, pred_jc, pred_xt = preds
    p = [1,2]
    assert np.allclose(pred_xc.get_spectrum(p), [5/9,0])
    assert np.allclose(pred_jc.get_spectrum(p), [np.sqrt(85)/9,0])
    assert np.allclose(pred_xt.get_spectrum(p), [0.445993544186507,0.0667379352854524])


def test_get_dat(preds):
    #dat = pred.get_dat()
    pass



def test_fitting():
    pass


def test_sampling():
    pass




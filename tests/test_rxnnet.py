"""
"""

from rxnnet import network, experiments


net = network.Network()


expts_xc = experiments.Experiments()
expts_jc = experiments.Experiments()
expts = expts_xc + expts_jc

pred_xc = net.get_predict(expts_xc, tol_ss=1e-13)
pred_jc = net.get_predict(expts_jc, tol_ss=1e-13)
pred = net.get_predict(expts, tol_ss=1e-13)


def test_spectrum():
    spec_xc = pred_xc.get_spectrum()
    spec = pred.get_spectrum()


def test_get_dat():
    dat = pred.get_dat()


def test_fitting():
    pass


def test_sampling():
    pass




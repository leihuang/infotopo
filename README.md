# infotopo

A Python package for carrying out some analysis of "typical"<sup>1</sup> mathematical models, including:
* parameter estimation 
* sampling of parameter posterior distribution
* model reduction
* model comparison

The frameworks and methods here follow the treatments of **information geometry** and its extension **information topology**<sup>2</sup>, hence the name of the package. 

Note 1: The technical definition of "typical" here is any mathematical model whose predictions are differentiable with respect to parameters, which includes most models in physical sciences. 

Note 2: See http://doi.org/10.1103/PhysRevE.83.036701 and http://arxiv.org/abs/1409.6203. 

## Prerequisites


## Usage examples
```python
mod = Model()
expts = Experiments()

pred = mod.get_pred(expts)

s = pred.get_spectrum()

dat = pred.get_data()

res = residual.Residual(pred, dat)

fit = fitting.leverberg_marquardt(res, p0)

ens = sampling.sampling(res, p0, nstep=100)

ens.scatterplot()

gds = pred.get_geodesic()

gds.integrate()

gds.plot()

```
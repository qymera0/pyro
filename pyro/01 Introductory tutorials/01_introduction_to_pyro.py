# IMPORT LIBRARIES

import logging
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

# CONFIGURE CODE

smoke_test = ('CI' in os.environ)

assert pyro.__version__.startswith('1.8.0')

pyro.enable_validation(True)

logging.basicConfig(format='%(message)s', level=logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# OPEN DATA

dataUrl = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"

data = pd.read_csv(dataUrl, encoding='ISO-8859-1')

df = data[['cont_africa', 'rugged', 'rgdppc_2000']]

# DATAW RANGLING

df = df[np.isfinite(df.rgdppc_2000)]

df['rgdppc_2000'] = np.log(df['rgdppc_2000'])

# CONVERT NUMPY ARRAY TO TORCH TENSOR

train = torch.tensor(df.values, dtype=torch.float)

is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:,  2]

# EDA

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

african_nations = df[df["cont_africa"] == 1]

non_african_nations = df[df["cont_africa"] == 0]

sns.scatterplot(x=non_african_nations["rugged"],
                y=non_african_nations["rgdppc_2000"],
                ax=ax[0])

ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")

sns.scatterplot(x=african_nations["rugged"],
                y=african_nations["rgdppc_2000"],
                ax=ax[1])

ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

# MAXIMUM LIKELYHOOD LINEAR REGRESSION

## No distributions is assumed for any parameters

def simple_model(is_cont_africa, ruggedness, log_gdp=None):    
    
    # Parameters priors    
    
    a = pyro.param('a', lambda: torch.randn(()))
    b_a = pyro.param("bA", lambda: torch.randn(()))
    b_r = pyro.param("bR", lambda: torch.randn(()))
    b_ar = pyro.param("bAR", lambda: torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)
    
    # Model definition
    
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    
    with pyro.plate('data', len(ruggedness)):
        return pyro.sample('obs', dist.Normal(mean, sigma), obs=log_gdp)

pyro.render_model(simple_model, model_args=(is_cont_africa, ruggedness, log_gdp))    

# BAYESIAN REGRESSION

def model(is_model_africa, ruggedness, log_gdp=None):
    
    # Parameters priors
    
    a = pyro.param('a', dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    
    # Model definition
    
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    
    with pyro.plate('data', len(ruggedness)):
        return pyro.sample('obs', dist.Normal(mean, sigma), obs=log_gdp)
    
pyro.render_model(model, model_args=(is_cont_africa, ruggedness, log_gdp))    

# MEAN FIELD VARIATIONAL APPROXIMATION FOR BAYESIAN LINEAR REGRESSION

def custom_guide(is_cont_africa, ruggedness, log_gdp=None):
    
    a_loc = pyro.param('a_loc', lambda: torch.tensor(0.))
    a_scale = pyro.param('a_scale', lambda: torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', lambda: torch.tensor(1.),
                             constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', lambda: torch.randn(3))
    weights_scale = pyro.param('weights_scale', lambda: torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    
    return {"a": a, "b_a": b_a, "b_r": b_r, "b_ar": b_ar, "sigma": sigma}

pyro.render_model(custom_guide, model_args=(is_cont_africa, ruggedness, log_gdp))

auto_guide = pyro.infer.autoguide.AutoNormal(model)

# BAYESIAN REGRESSION VIA STOCHASTIC VARIATION INFERENCE (SVI)

## Algorithm configuration

adam = pyro.optim.Adam({'lr': 0.02})

elbo = pyro.infer.Trace_ELBO()

svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []

for step in range(1000 if not smoke_test else 2):
    loss = svi.step(is_cont_africa, ruggedness, log_gdp)
    losses.append(loss)
    if step % 100 == 0:
        logging.info('Elbo loss: {}'.format(loss))
        

plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss");
Text(0, 0.5, 'ELBO loss')

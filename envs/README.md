# Environments

We provide in this folder the two environments necessary to run the training code and optimizing the bias correction layers.
To install them using conda, run the following commands:

```bash
conda env create -f ./envs/FACIL.yml
conda env create -f ./envs/iCaRL.yml
```

Note that the iCaRL environment is installed on top of Python 2.7, to be able to run the original iCaRL code.
Do not alter the name of these environments, as they are reused later in demo scripts.

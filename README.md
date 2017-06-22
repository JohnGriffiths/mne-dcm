# mne-dcm
Dynamic Causal Modelling in MNE-python

# Overview

Dynamic Causal Modelling (DCM) is a framework for modelling (primarily) neuroimaging data signals using biophysically-based neural population models, with an emphasis on (small) networks and estimation of effective connectivity. DCM lives inside SPM, somewhat confusingly spread across multiple sub-toolboxes, and following the standard SPM convention of being cryptically coded and poorly (if at all) documented. This project aims to liberate DCM from these squalid matlab shackles, and (to mix metaphors somewhat) unleash unto it the beast that is the mne-python and broader nipy development community. A subsidiary aim is to make some of the better parts of the DCM optimization machinery available to the neurophysiological modelling and neuroinformatics platform The Virtual Brain (TVB).

There’s really two components to DCM – the neural modelling part, which (for M/EEG) are generally variants on the second-order linear filter-type model of Jansen & Rit, and the model estimation part, which use Variational Bayes E-M type algorithms. For this project we’ll focus on the neural models described in spm_dcm_erp.m, and the generic VB-inversion routine spm_nlsi_N.m, plus the related helper functions above and below these two. We will push hard, and aim to have a functioning MNE-Python ERP-DCM implementation by the end of the hackathon. The acid test shall be to be able to run the DCM ERP tutorial in the SPM manual.

There should be many interesting questions that come up along the way, to do with improvements and general integration with other nipy and generic python libraries. Can we make use of other python libraries such as scikit-learn for parts of this? Would the VB-inversion routine be useful for fitting non-DCM models in MNE? Etc. etc. I’m hoping to get lots of feedback and good ideas from the imaging community on these Qs.


# Plan


The main thing we need to do is port several matlab functions to python. Including:

[spm_fx_erp.m](https://github.com/neurodebian/spm12/blob/master/toolbox/dcm_meeg/spm_fx_erp.m) (Jansen-Rit-David model equations for ERP simulation)
[spm_int_L.m](https://github.com/neurodebian/spm12/blob/master/spm_int_L.m) (integrator with explicit Jacobian)
[spm_int_ode.m](https://github.com/neurodebian/spm12/blob/master/spm_int_ode.m) (generic ode integrator)
[spm_nlsi_N.m](https://github.com/neurodebian/spm12/blob/master/spm_nlsi_N.m) (nonlinear system identification - i.e. model fitting)

Each of these do have lots of other dependencies that will also need to be dealt with. 




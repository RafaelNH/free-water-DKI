# Free Water Elimination DKI Model

A first version of the DKI free water elimination model (fwDKI) is implemented.

I add also a first version of the tests. In addition to test the fwDKI model, I am also processing the standard DKI model and fwDTI model just to illustrate why the fwDKI is important.

At the moment, only a non-linear approach to fit the DKI model was implemented. This fit seems only stable to contamination with volume fraction lower than 0.6. This might be because of the given initial guess. Currently I am using the standard DKI model as the initial guess for the DT and KT parameters and the fwDTI model as the initial guess for the volume fraction f. For larger contamination volume fractions the initial guess will be farther from the ground truth solution. As the next step, I have to implement a better strategy to estimate a better first guess.

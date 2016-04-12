# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:56:43 2016

@author: Rafael Neto Henriques
"""

import numpy as np
import scipy.optimize as opt
import dipy.reconst.dti as dti
import dipy.reconst.dki as dki
import dipy.reconst.fwdti as fwdti
from dipy.reconst.dti import (decompose_tensor, from_lower_triangular,
                              lower_triangular)
from dipy.reconst.vec_val_sum import vec_val_vect


def _nls_err_func(params, design_matrix, data, Diso=3e-3,
                  f_transform=False):
    """ Error function for the non-linear least-squares fit of the tensor water
    elimination model.

    Parameters
    ----------
    params : array (23, )
        The six independent elements of the diffusion tensor followed by
        the fifth elements of the kurtosis tensor, the -log(S0) signal,
        and the volume fraction f of the water elimination compartment.
        If f_transform is true, volume fraction f has to be converted to
        ft = arcsin(2*f - 1) + pi/2
    design_matrix : array (22,)
        The dki design matrix
    data : array
        The voxel signal in all gradient directions
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    f_transform : bool, optional
        If true, the water volume fraction was converted to
        ft = arcsin(2*f - 1) + pi/2, insuring f estimates between 0 and 1.
        See fwdti.nls_fit_tensor
        Default: True
    """
    tensor = np.copy(params)

    if f_transform:
        f = 0.5 * (1 + np.sin(tensor[22] - np.pi/2))
    else:
        f = tensor[22]

    # This is the predicted signal given the params:
    y = (1-f) * np.exp(np.dot(design_matrix, tensor[:22])) + \
        f * np.exp(np.dot(design_matrix,
                          np.array([Diso, 0, Diso, 0, 0, Diso,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, tensor[21]])))

    return data - y


def nls_fit_tensor(design_matrix, design_matrix_dki, data, S0, params=None, Diso=3e-3,
                    f_transform=True, mdreg=2.7e-3):
    """
    Fit the water elimination DKI model using the non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    S0 : ndarray ([X, Y, Z])
        A first guess of the non-diffusion signal S0. 
    params : ndarray ([X, Y, Z, ...], 28), optional
        A first model parameters guess (3 eigenvalues, 3 coordinates
        of 3 eigenvalues, 15 elements of the kurtosis tensor and the volume
        fraction of the free water compartment). If the initial params are
        not given, for the diffusion and kurtosis tensor parameters, its 
        initial guess is obtain from the standard DKI model, while for the
        free water fraction its value is estimated using the fwDTI model.
        Default: None
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    f_transform : bool, optional
        If true, the water volume fractions is converted during the convergence
        procedure to ft = arcsin(2*f - 1) + pi/2, insuring f estimates between
        0 and 1.
        Default: True
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).

    Returns
    -------
    fw_params : ndarray (x, y, z, 28)
        Matrix containing in the dimention the free water model parameters in
        the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
            4) The volume fraction of the free water compartment
    S0 : ndarray (x, y, z)
        The models estimate of the non diffusion-weighted signal S0.
    """
    # preparing data and initializing parameters
    data = np.asarray(data)
    data_flat = np.reshape(data, (-1, data.shape[-1]))
    S0out = S0.copy()
    S0out = S0out.ravel()

    # Computing WLS DTI solution for MD regularization
    dtiparams = dti.wls_fit_tensor(design_matrix, data_flat)
    md = dti.mean_diffusivity(dtiparams[..., :3])
    cond = md > mdreg  # removal condition
    data_cond = data_flat[~cond, :]

    # Initializing fw_params according to selected initial guess
    if np.any(params) is None:
        params_out = np.zeros((len(data_flat), 28))
        dkiparams = dki.wls_fit_dki(design_matrix_dki, data_flat)
        fweparams, sd = fwdti.wls_fit_tensor(design_matrix, data_flat,
                                             S0=S0, Diso=Diso,
                                             mdreg=2.7e-3)
        params_out[:, 0:27] = dkiparams
        params_out[:, 27] = fweparams[:, 12]
    else:
        params_out = params.copy()
        params_out = np.reshape(params_out, (-1, params_out.shape[-1]))

    params_cond = params_out[~cond, :]
    S0_cond = S0out[~cond]

    for vox in range(data_cond.shape[0]):
        if np.all(data_cond[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        params = params_cond[vox]

        # converting evals and evecs to diffusion tensor elements
        evals = params[:3]
        evecs = params[3:12].reshape((3, 3))
        dt = lower_triangular(vec_val_vect(evecs, evals))
        kt = params[..., 12:27]
        s0 = S0_cond[vox]

        # f transformation if requested
        if f_transform:
            f = np.arcsin(2*params[27] - 1) + np.pi/2
        else:
            f = params[27]

        # Use the Levenberg-Marquardt algorithm wrapped in opt.leastsq
        start_params = np.concatenate((dt, kt, [-np.log(s0), f]), axis=0)
        this_tensor, status = opt.leastsq(_nls_err_func, start_params,
                                          args=(design_matrix_dki,
                                                data_cond[vox],
                                                Diso, f_transform))

        # Invert f transformation if this was requested
        if f_transform:
            this_tensor[22] = 0.5 * (1 + np.sin(this_tensor[22] - np.pi/2))

        # The parameters are the evals and the evecs:
        evals, evecs = decompose_tensor(from_lower_triangular(this_tensor[:6]))
        params_cond[vox, :3] = evals
        params_cond[vox, 3:12] = evecs.ravel()
        params_cond[vox, 12:27] = this_tensor[6:21]
        params_cond[vox, 27] = this_tensor[22]
        S0_cond[vox] = np.exp(-this_tensor[21])

    params_out[~cond, :] = params_cond
    params_out[cond, 27] = 1  # Only free water
    params_out = np.reshape(params_out, (data.shape[:-1]) + (28,))
    S0out[~cond] = S0_cond
    S0out[cond] = \
        np.mean(data_flat[cond, :] / \
                np.exp(np.dot(design_matrix[..., :6],
                              np.array([Diso, 0, Diso, 0, 0, Diso]))),
                -1)  # Only free water
    S0out = S0out.reshape(data.shape[:-1])
    return params_out, S0out
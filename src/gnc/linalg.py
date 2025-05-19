"""
Linear algebra functions. 

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen


This is an extension from PythonVehicleSimulator
Author on some of the function is @rambech
"""


import numpy as np


def Smtrx(a):
    """
    Skew-symmetric matrix

    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b. 

    Parameters
    ----------
        a: np.ndarray
            vector

    Returns
    -------
        S: np.ndarray
            skew-symmetric matrix

    """

    S = np.array([
        [0, -a[2], a[1]],
        [a[2],   0,     -a[0]],
        [-a[1],   a[0],   0]])

    return S


def Hmtrx(r):
    """
    Transformation from CG to CO

    H = Hmtrx(r) computes the 6x6 system transformation matrix
    H = [eye(3)     S'
         zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)

    If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
    CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
    force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG 

    Parameters
    ----------
        phi: float
            roll angle
        theta: float
            pitch angle
        psi: float
            yaw angle

    Returns
    -------
        R: np.ndarray
            Rotation matrix

    """

    H = np.identity(6, float)
    H[0:3, 3:6] = Smtrx(r).T

    return H


def Rzyx(phi, theta, psi):
    """
    Rotation transformation matrix in SO(3)

    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention

    Parameters
    ----------
        phi: float
            roll angle
        theta: float
            pitch angle
        psi: float
            yaw angle

    Returns
    -------
        R: np.ndarray
            Rotation matrix

    """

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array([
        [cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth],
        [spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
        [-sth,      cth*sphi,                 cth*cphi]])

    return R


def Tzyx(phi, theta):
    """
    Euler angle transformation

    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention

    Parameters
    ----------
        phi: float
            roll angle
        theta: float
            pitch angle

    Returns
    -------
        Tzyx: np.ndarray
            Euler angle transformation

    """

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)

    try:
        T = np.array([
            [1,  sphi*sth/cth,  cphi*sth/cth],
            [0,  cphi,          -sphi],
            [0,  sphi/cth,      cphi/cth]])

    except ZeroDivisionError:
        print("Tzyx is singular for theta = +-90 degrees.")

    return T


def R(psi: float) -> np.ndarray:
    """
    Simple 2x2 rotation matrix

    Parameters
    ----------
        psi : float
            Heading angle

    Returns
    -------
        R : np.ndarray
            2D rotation matrix

    """
    return np.array([[np.cos(psi), -np.sin(psi)],
                     [np.sin(psi), np.cos(psi)]])


def Rz(psi: float) -> np.ndarray:
    """
    Rotation matrix around the z axis

    Parameters
    ----------
        psi : float
            Heading angle

    Returns
    -------
        R : np.ndarray
            2D rotation matrix

    """
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])


def moore_penrose(A: np.ndarray):
    """
    Right hand Moore-Penrose pseudo-inverse

    A^T(AA^T)^-1

    Parameters
    ----------
        A : np.ndarray
            Non-invertible matrix

    Returns
    -------
        moore_penrose : np.ndarray
            Inverted matrix

    """

    return A.T.dot(np.linalg.inv(A.dot(A.T)))


def singular_projection(A, sing_val_thres=1e-2, svd_tol=1e-7):
    """
    Makes a singular projection based on the smallest value below 
    sing_val_thres and returns the projection. If a projection could 
    not be made, an identity matrix of size A.shape[0] will be returned

    Additionally, we check if the SVD is within svd_tol

    Parameters
    ----------
        A : np.ndarray
            Any matrix
        sing_val_thres : float
            Upper limit to where we can pick singular values from
        svd_tol : float
            Upper limit on absolute SVD recomposition error

    Returns
    -------
        projection : np.ndarray
            Singular projection or identity matrix if a singular 
            projection could not be made

    """

    U, S, Vh = np.linalg.svd(A)

    A_recomposed = U @ np.diag(S) @ Vh

    if np.allclose(A, A_recomposed, atol=svd_tol):
        mask = S < sing_val_thres
        if np.any(mask):
            Vh_reduced = Vh[mask]
            return Vh_reduced.T @ Vh_reduced
        else:
            print(f"No singular values, projection not made")
    else:
        print(f"Recomposition of SVD matrices failed, projection not made")

    return np.eye(A.shape[0])

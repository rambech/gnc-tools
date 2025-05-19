"""
GNC functions. 

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen


This is an extension from PythonVehicleSimulator
Author on some of the function is @rambech
"""

import numpy as np
from numpy.linalg import LinAlgError
from .linalg import moore_penrose, Smtrx, Rzyx, Tzyx, R, Rz


def distance_along_great_circle(lat0: float, lon0: float, lat1: float, lon1: float):
    """
    Computes the distance between two points on the earths surface

    Based on on the haversine formula 
    and provided on stackoverflow by user b-h-:
    https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters   

    Parameters
    ----------
        lat0 : float
            Latitude at start point
        lon0 : float
            Longitude at start point
        lat1 : float
            Latitude at start point
        lon1 : float
            Longitude at start point

    Returns
    -------
        distance : float
            Distance from start to end in meters

    """

    earth_radius = 6378.137 * 1000
    delta_lat = D2R(lat1) - D2R(lat0)
    delta_lon = D2R(lon1) - D2R(lon0)

    a = np.sin(delta_lat/2)**2 + np.cos(D2R(lat0)) * \
        np.cos(D2R(lat1)) * np.sin(delta_lon/2)**2

    angle = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a))
    distance = earth_radius * angle

    return distance


def ssa(angle: np.ndarray | float):
    """
    Smallest-signed angle in [-pi, pi)

    angle = ssa(angle)

    Parameters
    ----------
        angle: np.ndarray | float
            angle in radians

    Returns
    -------
        angle: np.ndarray | float
            angle in radians mapped to [-pi, pi)

    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    return angle


def sat(x_min, x, x_max):
    """
    Saturation, bounds input value between two given values

    x = sat(x_min,x,x_max) saturates a signal x such that x_min <= x <= x_max

    Parameters
    ----------
        x_min: float
            lower bound 
        x: float
            value to be bounded
        x_max: float
            upper bound

    Returns
    -------
        x: float
            saturated value

    """
    if x > x_max:
        x = x_max
    elif x < x_min:
        x = x_min

    return x


def attitudeEuler(eta, nu, dt):
    """
    Forward euler intgration of kinematics

    eta = attitudeEuler(eta,nu,dt) computes the generalized 
    position/Euler angles eta[k+1]

    Parameters
    ----------
        eta: np.ndarray
            current vehicle pose in NED
        nu: np.ndarray
            vehicle velocity in BODY
        dt: float
            time step

    Returns
    -------
        eta: np.ndarray
            next vehicle pose in NED

    """

    eta_dot = B2N(eta).dot(nu)
    eta = eta + eta_dot * dt

    return eta


def m2c(M, nu):
    """
    Coriolis-centripetal matrix from mass matrix

    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)

    Parameters
    ----------
        M: np.ndarray
            mass matrix
        nu: np.ndarray
            vehicle velocity in BODY

    Returns
    -------
        C: np.ndarray
            coriolis-centripetal matrix

    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):
        # 6-DOF model
        M11 = M[0:3, 0:3]
        M12 = M[0:3, 3:6]
        M21 = M12.T
        M22 = M[3:6, 3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11, nu1) + np.matmul(M12, nu2)
        dt_dnu2 = np.matmul(M21, nu1) + np.matmul(M22, nu2)

        # C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C = np.zeros((6, 6))
        C[0:3, 3:6] = -Smtrx(dt_dnu1)
        C[3:6, 0:3] = -Smtrx(dt_dnu1)
        C[3:6, 3:6] = -Smtrx(dt_dnu2)

    else:
        # 3-DOF model (surge, sway and yaw)
        # C = [            0                    0       M(2,2)*nu(2)+M(2,3)*nu(3)
        #                  0                    0             -M(0,0)*nu(0)
        #      -M(2,2)*nu(2)-M(2,3)*nu(3)  M(0,0)*nu(0)            0             ]
        C = np.zeros((3, 3))
        C[0, 2] = M[1, 1] * nu[1] + M[1, 2] * nu[2]
        C[1, 2] = -M[0, 0] * nu[0]
        C[2, 0] = -C[0, 2]
        C[2, 1] = -C[1, 2]

    return C


def Hoerner(B, T):
    """
    Hoerner's method in 2D

    CY_2D = Hoerner(B,T)
    Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam 
    B and draft T.The data is digitized and interpolation is used to compute 
    other data point than those in the table

    Parameters
    ----------
        B: float
            beam
        T: float
            draft

    Returns
    -------
        CY_2D: np.ndarray
            cross-flow coefficients

    """

    # DATA = [B/2T  C_D]
    DATA1 = np.array([
        0.0109, 0.1766, 0.3530, 0.4519, 0.4728, 0.4929, 0.4933, 0.5585, 0.6464, 0.8336,
        0.9880, 1.3081, 1.6392, 1.8600, 2.3129, 2.6000, 3.0088, 3.4508, 3.7379, 4.0031
    ])
    DATA2 = np.array([
        1.9661, 1.9657, 1.8976, 1.7872, 1.5837, 1.2786, 1.2108, 1.0836, 0.9986, 0.8796,
        0.8284, 0.7599, 0.6914, 0.6571, 0.6307, 0.5962, 0.5868, 0.5859, 0.5599, 0.5593
    ])

    CY_2D = np.interp(B / (2 * T), DATA1, DATA2)

    return CY_2D


def crossFlowDrag(L, B, T, nu_r):
    """
    Cross-flow Drag

    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
    integrals for a marine craft using strip theory. 

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow

    Parameters
    ----------
        L: float
            craft length
        B: float
            craft beam
        T: float
            draft
        nu_r: np.array
            velocity relative to water

    Returns
    -------
        tau_crossflow: np.ndarray
            cross-flow drag force

    """

    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20
    Cd_2D = Hoerner(B, T)    # 2D drag coefficient based on Hoerner's curve

    Yh = 0
    Nh = 0
    xL = -L/2

    for i in range(0, n+1):
        v_r = nu_r[1]             # relative sway velocity
        r = nu_r[5]               # yaw rate
        Ucf = abs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
        xL += dx

    tau_crossflow = np.array([0, Yh, 0, 0, 0, Nh], float)

    return tau_crossflow


def forceLiftDrag(b, S, CD_0, alpha, U_r):
    """
    Lift drag force

    tau_liftdrag = forceLiftDrag(b,S,CD_0,alpha,Ur) computes the hydrodynamic
    lift and drag forces of a submerged "wing profile" for varying angle of
    attack (Beard and McLain 2012). Application:

      M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_liftdrag

    Parameters
    ----------
        b: float
            wing span (m)
        S: float
            wing area (m^2)
        CD_0: float 
            parasitic drag (alpha = 0), typically 0.1-0.2 for a 
            streamlined body
        alpha: float
            angle of attack, scalar or vector (rad)
        U_r: float  
            relative speed (m/s)

    Returns
    -------
        tau_liftdrag: np.ndarray 
            6x1 generalized force vector
    """

    # constants
    rho = 1026

    def coeffLiftDrag(b, S, CD_0, alpha, sigma):
        """
        Lift drag coefficients

        [CL,CD] = coeffLiftDrag(b,S,CD_0,alpha,sigma) computes the hydrodynamic 
        lift CL(alpha) and drag CD(alpha) coefficients as a function of alpha
        (angle of attack) of a submerged "wing profile" (Beard and McLain 2012)

        CD(alpha) = CD_p + (CL_0 + CL_alpha * alpha)^2 / (pi * e * AR)
        CL(alpha) = CL_0 + CL_alpha * alpha

        where CD_p is the parasitic drag (profile drag of wing, friction and
        pressure drag of control surfaces, hull, etc.), CL_0 is the zero angle 
        of attack lift coefficient, AR = b^2/S is the aspect ratio and e is the  
        Oswald efficiency number. For lift it is assumed that

        CL_0 = 0
        CL_alpha = pi * AR / ( 1 + sqrt(1 + (AR/2)^2) );

        implying that for alpha = 0, CD(0) = CD_0 = CD_p and CL(0) = 0. For
        high angles of attack the linear lift model can be blended with a
        nonlinear model to describe stall

        CL(alpha) = (1-sigma) * CL_alpha * alpha + ...
            sigma * 2 * sign(alpha) * sin(alpha)^2 * cos(alpha) 

        where 0 <= sigma <= 1 is a blending parameter. 


        Parameters
        ----------
            b: float
                wing span (m)
            S: float
                wing area (m^2)
            CD_0: float 
                parasitic drag (alpha = 0), typically 0.1-0.2 for a 
                streamlined body
            alpha: float
                angle of attack, scalar or vector (rad)
            sigma: float
                blending parameter between 0 and 1, use sigma = 0 f
                or linear lift 
            display: use 1 to plot CD and CL (optionally)

        Returns
        -------
            CL: lift coefficient as a function of alpha   
            CD: drag coefficient as a function of alpha   

        Example:
            Cylinder-shaped AUV with length L = 1.8, diameter D = 0.2 and 
            CD_0 = 0.3

            alpha = 0.1 * pi/180
            [CL,CD] = coeffLiftDrag(0.2, 1.8*0.2, 0.3, alpha, 0.2)
        """

        e = 0.7             # Oswald efficiency number
        AR = b**2 / S       # wing aspect ratio

        # linear lift
        CL_alpha = np.pi * AR / (1 + np.sqrt(1 + (AR/2)**2))
        CL = CL_alpha * alpha

        # parasitic and induced drag
        CD = CD_0 + CL**2 / (np.pi * e * AR)

        # nonlinear lift (blending function)
        CL = (1-sigma) * CL + sigma * 2 * np.sign(alpha) \
            * np.sin(alpha)**2 * np.cos(alpha)

        return CL, CD

    [CL, CD] = coeffLiftDrag(b, S, CD_0, alpha, 0)

    F_drag = 1/2 * rho * U_r**2 * S * CD    # drag force
    F_lift = 1/2 * rho * U_r**2 * S * CL    # lift force

    # transform from FLOW axes to BODY axes using angle of attack
    tau_liftdrag = np.array([
        np.cos(alpha) * (-F_drag) - np.sin(alpha) * (-F_lift),
        0,
        np.sin(alpha) * (-F_drag) + np.cos(alpha) * (-F_lift),
        0,
        0,
        0])

    return tau_liftdrag


def gvect(W, B, theta, phi, r_bg, r_bb):
    """
    g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring 
    forces about an arbitrarily point CO for a submerged body. 

    Parameters
    ----------
        W: float
            weight (N)
        B: float
            buoyancy (N)
        phi: float
            roll angle (rad)
        theta: float
            pitch angle (rad)
        r_bg = [x_g y_g z_g]: np.ndarray
            location of the CG with respect to the CO (m)
        r_bb = [x_b y_b z_b]: np.ndarray
            location of the CB with respect to th CO (m)

    Returns
    -------
        g: np.ndarray
            6x1 vector of restoring forces about CO

    """

    sth = np.sin(theta)
    cth = np.cos(theta)
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    g = np.array([
        (W-B) * sth,
        -(W-B) * cth * sphi,
        -(W-B) * cth * cphi,
        -(r_bg[1]*W-r_bb[1]*B) * cth * cphi +
        (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
        (r_bg[2]*W-r_bb[2]*B) * sth + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
        -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth
    ])

    return g


def D2R(deg: float) -> float:
    """
    degrees to radians

    rad = (deg * pi) / 180

    Parameters
    ----------
        deg: float
            Degrees

    Returns
    -------
        rad: float
            Angle in rad
    """

    return deg*np.pi/180


def R2D(rad: float) -> float:
    """
    radians to degrees

    deg = (rad * 180) / pi

    Parameters
    ----------
        rad: float 
            Radians

    Returns
    -------
        deg: float
            Smallest signed angle in degrees

    """

    return ssa(rad)*180/np.pi


def kts2ms(kts: float) -> float:
    """
    knots to m/s

    ms = 0.5144 * kts

    Parameters
    ----------
        kts: float
            Knots

    Returns
    -------
        ms: float
            Meters per second

    """

    return 0.5144 * kts


def ms2kts(ms: float) -> float:
    """
    m/s to knots

    kts = 1.9438 * ms

    Parameters
    ----------
        ms: float
            meters per second

    Returns
    -------
        kts: float
            knots

    """

    return 1.9438 * ms


def B2N(eta: np.ndarray) -> np.ndarray:
    """
    BODY to NED transformation

    J_Theta(eta) = [R_b^n(Theta)       0_3x3    
                       0_3x3      T_Theta(Theta)]

    or 

    J_Theta(eta) = R_b^n(psi)

    Parameters
    ----------
        eta : np.ndarray 
            vehicle pose in NED frame

    Returns
    -------
        J : np.ndarray 
            BODY to NED transformation

    """

    if len(eta) == 6:
        # 6 DOF transform
        ROT = Rzyx(eta[3], eta[4], eta[5])
        T = Tzyx(eta[3], eta[4])

        J = np.block([[ROT, np.zeros((3, 3), float)],
                      [np.zeros((3, 3), float), T]])
    else:
        # 3 DOF transform
        J = Rz(eta[-1])

    return J


def N2B(eta: np.ndarray) -> np.ndarray:
    """
    NED to BODY transformation

    J^-1(eta) = (J_Theta(eta))^-1

    Parameters
    ----------
        eta: np.ndarray 
            vehicle pose in NED frame

    Returns
    -------
        J_inv: np.ndarray 
            NED to BODY transformation

    """

    J = B2N(eta)
    if len(eta) == 6:
        # 6 DOF version
        try:
            J_inv = np.linalg.inv(J)
        except LinAlgError:
            J_inv = moore_penrose(J)
    else:
        # 3 DOF version
        J_inv = J.T

    return J_inv


def N2S(eta_n, scale, origin):
    """
    Go from screen coordinates to NED coordinates

    Parameters
    ----------
        eta_n: np.ndarray
            vehicle pose in NED
        scale: float
            scale of render
        origin: np.ndarray
            origin in screen coordinates

    Returns
    -------
        eta_s: np.ndarray
            vehicle pose in "typical" screen coordinates

    """
    psi_offset = np.pi/2

    if len(eta_n) == 6:
        # 6 DOF version
        eta_s = N2B(np.array([0, 0, 0, 0, 0, psi_offset], float)).dot(eta_n)
        eta_s[:3] = eta_s[:3]*scale + origin

    else:
        # 3 DOF version
        eta_s = N2B(np.array([0, 0, psi_offset], float)).dot(eta_n)
        eta_s[:2] = eta_s[:2]*scale + origin[:2]

    return eta_s


def S2N(eta_s, scale, origin):
    """
    Go from NED coordinates to screen coordinates

    Parameters
    ----------
        eta_s: tuple[float, float]
            vehicle pose in "typical" screen coordinates
        scale: float
            scale of render
        origin: np.ndarray
            origin in screen coordinates

    Returns
    -------
        eta_n: np.ndarray
            vehicle pose in NED

    """
    psi_offset = np.pi/2

    if len(eta_s) == 6:
        # 6 DOF version
        eta_s[:3] = (eta_s[:3] - origin)/scale
        eta_n = B2N(np.array([0, 0, 0, 0, 0, psi_offset], float)).dot(eta_s)

    else:
        # 3 DOF version
        eta_s[:2] = (eta_s[:2] - origin[:2])/scale
        eta_n = N2B(np.array([0, 0, psi_offset], float)).dot(eta_s)

    return eta_n


def N2S2D(eta_n_2D: np.ndarray, scale: float, origin: np.ndarray) -> np.ndarray:
    """
    Go from NED coordinates to screen coordinates

    Parameters
    ----------
        eta_n_2D: np.ndarray
            planer vehicle pose in NED
        scale: float
            scale of render
        origin: np.ndarray
            origin in screen coordinates

    Returns
    -------
        eta_s_2D: np.ndarray
            planer vehicle pose in "typical" screen coordinates

    """

    psi_offset = np.pi/2
    rotated = R(psi_offset).T.dot(eta_n_2D)
    scaled = rotated*scale
    eta_s_2D = scaled
    eta_s_2D[0:2] += origin[0:2]

    return eta_s_2D


def S2N2D(eta_s_2D: tuple[float, float], scale: float, origin: np.ndarray) -> np.ndarray:
    """
    Go from screen coordinates to NED coordinates

    Parameters
    ----------
        eta_s_2D: tuple[float, float]
            planer vehicle pose in "typical" screen coordinates
        scale: float
            scale of render
        origin: np.ndarray
            origin in screen coordinates

    Returns
    -------
        eta_n_2D: np.ndarray
            planer vehicle pose in NED

    """

    psi_offset = np.pi/2
    descaled = (eta_s_2D - origin[0:2])/scale
    eta_n_2D = R(psi_offset).dot(descaled)

    return eta_n_2D


def D2L(edge: tuple[tuple[float, float], tuple[float, float]], pos: np.ndarray) -> tuple[float, float]:
    """
    Calculates distance from a point to a line using vector projection

    Parameters
    ----------
        edge : tuple[tuple[float, float], tuple[float, float]]
            points that make up the line
        pos: np.ndarray
            point to calculate distance from

    Returns
    -------
        angle : float
            angle between (x_n,y_n) and the closest point on the line,
            expressed in {n}
        dist : float
            distance from vessel to the closet point on the line

    """

    # Make tuples into ndarrays for easier calculation
    v1 = np.asarray(edge[0], float)
    v2 = np.asarray(edge[1], float)

    a = pos - v1    # Vector from one vertex to the vessel
    b = v2 - v1     # Vector making up the edge

    # Projection from vessel down to the edge
    proj = np.asarray(a.dot(b) / b.dot(b)).dot(b)

    # Is projection on the edge
    if 0 <= proj[0] <= b[0] and 0 <= proj[1] <= b[1]:
        dist = np.linalg.norm(a - proj, 2)
        angle = np.arctan2(proj[1] - a[1], proj[0] - a[0])

    # Which vertex is closer to the vessel
    elif np.linalg.norm(0 - a, 2) < np.linalg.norm(b - a, 2):
        dist = np.linalg.norm(0 - a, 2)
        angle = np.arctan2(0 - a[1], 0 - a[0])

    else:
        dist = np.linalg.norm(b - a, 2)
        angle = np.arctan2(b[1] - a[1], b[0] - a[0])

    return ssa(angle), dist


def V2C(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vertices to constraints
    Convert a set of vertices making up a polygon into inequality constraints

    Initially made by Michael Kleder in 2005 and rewritten in python by me (@rambech):
    https://se.npworks.com/matlabcentral/fileexchange/7895-vert2con-vertices-to-constraints

    Ax <= b

    The code will give the same answer as in the Matlab code, but the may switch the row position.

    Parameters
    ----------
        vertices : np.ndarray
            vertices in NED representing geometrical constraints

    Returns
    -------
        A : np.ndarray
            inequality coefficient matrix
        b : np.ndarray
            new constraint

    """

    try:
        from scipy.spatial import ConvexHull
    except ModuleNotFoundError:
        print("scipy not installed")

    # Remove duplicate vertices
    vertices = np.unique(vertices, axis=0)

    # Create convex hull
    k = ConvexHull(vertices)

    # Subtract vertices by row mean
    c = np.mean(vertices, axis=0)
    v = vertices - np.tile(c, (len(vertices), 1))

    # Initialize A matrix
    A = np.zeros((len(k.simplices), len(v[1])))

    count = 0
    for idx in range(len(k.simplices)):
        # Process one edge at a time
        F = v[k.simplices[idx, :], :]

        if np.linalg.matrix_rank(F, 1e-5) == len(F[0]):
            A[count, :] = np.linalg.solve(F, np.ones(
                (len(F[0]),)))

            count += 1

    A = A[:count, :]
    b = np.ones((len(A), 1))
    b_temp = A.dot(c.T)
    b += b_temp.reshape(len(b_temp), 1)

    # Ensure uniqueness along system rows
    I = np.unique(np.hstack([A, b]), axis=1)
    A = I[:, :len(A[0])]
    b = I[:, -1]

    return A, b


def PP(pos_diff: np.ndarray, k: float) -> np.ndarray:
    """
    Pure-pursuit velocity given in BODY

    Parameters
    ----------
        pos_diff : np.ndarray
            difference in position on a 2D-plane or 3D-space, 
            delta_p = [delta_x, delta_y] or delta_p = [delta_x, delta_y, delta_z]
        k : float
            pure-pursuit scaling factor, must be positive

    Returns
    -------
        v_d : np.ndarray
            desired velocity towards target
    """

    return -k*(pos_diff/np.linalg.norm(pos_diff))


def RK4(x, u, dt, ode):
    """
    General RK4 function

    Parameters
    ----------
        x: np.array | float
            states to be integrated
        u: np.array | float
            control signal
        dt: float
            time step
        ode: function
            ODE describing the system

    Returns
    -------
        x: np.array | float
            updated states

    """
    print(f"x in RK4: {x}")
    k1 = ode(x, u)
    print(f"k1: {k1}")
    k2 = ode(x + dt/2 * k1, u)
    k3 = ode(x + dt/2 * k2, u)
    k4 = ode(x + dt * k3,   u)

    return x + dt/6 * (k1+2*k2+2*k3+k4)

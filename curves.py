"""
Geometrical Geodesy Curves
Thomas H Meyer
Department of Natural Resources and the Environment
University of Connecticut
6-March-2024
Functions for drawing curve of alignments, normal sections, great ellipses, loxodromes
"""

import numpy as np
from scipy.optimize import newton
from typing import Tuple

from numpy import sqrt, sin, cos, tan, arctan, arctan2, pi, sign, log, exp, arccos, arctanh, tanh, arcsin
from numpy import transpose as Tr

eps = np.finfo(float).eps
pi2 = np.pi / 2.0


class refEllipsoid:
    def __init__(self, a: float, f: float):
        self.a = a
        self.asq = a*a
        self.f = f
        self.esq = 2*f - f*f
        self.ecc = sqrt(self.esq)
        self.b = a * (1 -f)
        self.bsq = self.b*self.b
        self.oneMinusEsq = 1 - self.esq
        # these are for XYZ -> LBH
        self.c1 = self.asq / self.bsq
        self.c2 = 1.0 / self.c1
        self.c3 = self.c1 * self.c1
        self.c4 = self.a * (1 - self.c2)
        self.c5 = self.a

    def N(self, phi: float) -> float:
        """
        Radius of curvature in the prime vertical
        :param phi: float in [-pi/2, pi/2] = geodetic latitude
        :param a: float = ref. ellipsoid semi-major axis
        :return:
        """
        assert -pi2 <= phi <= pi2
        sinPhi = sin(phi)
        return self.a / sqrt(1 - self.esq * sinPhi*sinPhi)


    def LBH_to_XYZ(self, LBH) -> Tuple[float, float, float]:
        """
        LBH to XYZ
        :param LBH: long_rad, lat_rad, h
        :param a: float = ref. ellipsoid semi-major axis
        :param esq: float = ref. ellipsoid (first) eccentricity squared
        :return: x,y,z
        """
        lamda, phi, h = LBH
        cos_phi = cos(phi)
        N_phi = self.N(phi)
        x = (N_phi + h) * cos_phi * cos(lamda)
        y = (N_phi + h) * cos_phi * sin(lamda)
        z = (self.oneMinusEsq * N_phi + h) * sin(phi)
        return np.array([x, y, z])


    def XYZ_to_LBH(self, xyz: np.array) -> Tuple[float, float, float]:
        """
        There are many implementations of this conversion. This one is the Bowring, new implementation from
        Claessens, S.J. (2019) Efficient transformation from Cartesian to geodetic coordinates. In
        Computers and Geosceinces. It is fast and accurate, but is intended for points "fairly near"
        the surface of the ellipsoid. It is an "approximate" method because, although iterative in principle,
        one iteration is good enough for the anticipated domain
        :param xyz: a collection holding the x, y, and z coordinates to convert such that x, y, z = xyz is valid syntax
        :param a: float = ref. ellipsoid semi-major axis
        :param b: float = ref. ellipsoid semi-minor axis = sqrt((1 - esq) * a*a)
        :param esq: float = ref. ellipsoid (first) eccentricity squared
        :return: tuple(lon, lat, h0)
        """
        x, y, z = xyz
        if -eps <= x <= eps and -eps <= y <= eps:  # at the poles
            return 0, np.sign(z) * pi2, np.abs(z) - self.b
        W2 = x * x + y * y
        W = sqrt(W2)
        Z2 = z * z
        K = W2 + self.c1 * Z2
        L = self.c4 / (K * sqrt(K))
        tau = (z + self.c3 * Z2 * z * L) / (W - W2 * W * L)
        phi = np.arctan(tau)
        tau2 = tau * tau
        h = (W + z * tau - self.c5 * sqrt(1 + self.c2 * tau2)) / sqrt(1 + tau2)
        return 2.0 * np.arctan2(y, x + W), phi, h


GRS80 = refEllipsoid(6378137, 1/298.257222101)


def CoAPoint(From: np.array, To: np.array, t: float, re=GRS80) -> Tuple[float, float, float]:
    """
    Returns a single point on the curve of alignment from point From to point To.
    From: numpy array [x, y, z] of geocentric coordinates for the starting point.
    To: numpy array [x, y, z] of geocentric coordinates for the ending point.
    t: float in the interval [0, 1].
    return: coordinates in LBH.
    """
    v = To - From
    l, b, h = re.XYZ_to_LBH(From + t * v)
    return l, b, 0.0


def CoA(From: np.array, To: np.array, re=GRS80, npts=50):
    """
    returns an array of npts (lon, lat, h) or XYZ coordinates on the curve
    of alignment starting at From, ending at To.
    From and To are numpy arrays of geocentric coordinates (m).
    coordinateSysName is a flag indicating the coordinate system for the results.
    LBH means lon, lat, h
    """
    assert npts > 1
    assert len(To) == len(From) == 3
    v = To - From
    xyzPnts = [From + t * v for t in np.linspace(0, 1, npts)]
    lbh = np.array([re.XYZ_to_LBH(xyz) for xyz in xyzPnts])
    lbh[:,2] = 0  # set the height coordinates to zero
    return lbh


def coaPoint_deakin(lat:float, C, U, H, V, W, re=GRS80) -> float:
    nu = re.N(lat) * re.oneMinusEsq * sin(lat)
    P = C * nu - U
    Q = H * nu + V
    S = W * re.oneMinusEsq * tan(lat)
    lon = arccos(S / sqrt(P*P + Q*Q)) + arctan2(-Q,P)
    if lon > np.pi:
        lon -= 2*np.pi
    return lon


def coaConstants_deakin(From: np.array, To: np.array, re=GRS80):
    C = re.esq * (To[1] - From[1])  # eSq (y2 - y1)
    H = re.esq * (To[0] - From[0])  # eSq (x2 - x1)
    U = re.oneMinusEsq * (From[1]*To[2] - To[1]*From[2])  # (1-esq) (y1 z2 - y2 z1)
    V = re.oneMinusEsq * (To[0]*From[2] - From[0]*To[2])  # (1-esq) (x2 z1 - x1 z2)
    W = From[0]*To[1] - To[0]*From[1]  # x1 y2 - x2 y1
    return C, H, U, V, W


def coa_deakin(From: np.array, To: np.array, re=GRS80, npts=50):
    assert npts > 1
    assert len(To) == len(From) == 3
    fromLBH = re.XYZ_to_LBH(From)
    toLBH = re.XYZ_to_LBH(To)
    C = re.esq * (To[1] - From[1])  # eSq (y2 - y1)
    H = re.esq * (To[0] - From[0])  # eSq (x2 - x1)
    U = re.oneMinusEsq * (From[1]*To[2] - To[1]*From[2])  # (1-esq) (y1 z2 - y2 z1)
    V = re.oneMinusEsq * (To[0]*From[2] - From[0]*To[2])  # (1-esq) (x2 z1 - x1 z2)
    W = From[0]*To[1] - To[0]*From[1]  # x1 y2 - x2 y1
    lats = np.linspace(fromLBH[1], toLBH[1], num=npts, endpoint=True)
    resLBH = [([coaPoint_deakin(lat, C, U, H, V, W, re), lat, 0]) for lat in lats]
    return resLBH


def uStar_ns(V: np.array, n: np.array, re=GRS80):
    """
    Returns u*
    :param V: point on the Z-axis such that VA is normal to RE
    :param n: vector = p - V
    :param RE: ref ellip
    :return: float
    """
    nx, ny, nz = n
    nx2 = nx*nx
    ny2 = ny*ny
    nz2 = nz*nz
    Vz = V[2]
    Vz2 = Vz * Vz
    a = re.a
    a2inv = 1/(a * a)
    one_minus_e2_inv = 1 / re.oneMinusEsq
    k0 = a2inv * one_minus_e2_inv * Vz2 - 1
    k1 = 2 * a2inv * one_minus_e2_inv * n[2] * Vz
    k1_2 = k1*k1
    k2 = a2inv * (nx2 + ny2 + one_minus_e2_inv * nz2)
    return (-k1 + sqrt(k1_2 - 4.0 * k2 * k0)) / (2.0 * k2)


def ns(From: np.array, To: np.array, t: float, re=GRS80):
    """
    Normal section: returns the geocentric coordinates of the point on the normal
        section from From to To at parameter t.
    From: numpy array [x, y, z] of geocentric coordinates for the starting point.
    To: numpy array [x, y, z] of geocentric coordinates for the ending point.
    t: float in the interval [0, 1].
    RE: ReferenceEllipsoid object.
    """
    lat = re.XYZ_to_LBH(From)[1]
    V = np.array([0, 0, -re.N(lat) * re.esq * sin(lat)])
    p_t = From + t * (To - From)
    pMinusV = p_t - V
    ustar = uStar_ns(V, pMinusV, re)
    return V + ustar * pMinusV


def normalSection(From: np.array, To: np.array, re=GRS80, npts=50):
    """
    returns an array of npts (lon, lat, h) or XYZ coordinates on the normal
    section starting at From, ending at To. (The h value is
    always 0 because h = 0 by definition.)
    From and To are numpy arrays of geocentric coordinates (m).
    coordinateSysName is a flag indicating the coordinate system for the results.
    LBH means lon, lat, h; XYZ means geocentric Cartesian.
    """
    assert npts > 1
    assert len(To) == len(From) == 3
    xyzPnts = [ns(From, To, t, re) for t in np.linspace(0, 1, npts)]
    return xyzPnts


def ge(From: np.array, To: np.array, t: float, re=GRS80):
    """
    Great ellipse: returns the geocentric coordinates of the point on the great
        ellipse from From to To at parameter t.
    From: numpy array [x, y, z] of geocentric coordinates for the starting point.
    To: numpy array [x, y, z] of geocentric coordinates for the ending point.
    t: float in the interval [0, 1].
    RE: ReferenceEllipsoid object.
    """
    v = To - From
    n = From + t * v
    nx, ny, nz = n
    const = re.oneMinusEsq
    a = re.a
    a2 = a*a
    uStar = a2 * sqrt( const / (const*(nx*nx + ny*ny) + nz*nz) )
    lat = re.XYZ_to_LBH(From)[1]
    V = np.array([0, 0, -re.N(lat) * re.esq * sin(lat)])
    p_t = From + t * (To - From)
    pMinusV = p_t - V
    res = V + uStar * pMinusV
    return res


def greatEllipse(From: np.array, To: np.array, re=GRS80, npts=50):
    """
    returns an array of npts (lon, lat, h) or XYZ coordinates on the normal
    section starting at From, ending at To. (The h value is
    always 0 because h = 0 by definition.)
    From and To are numpy arrays of geocentric coordinates (m).
    coordinateSysName is a flag indicating the coordinate system for the results.
    LBH means lon, lat, h; XYZ means geocentric Cartesian.
    """
    assert npts > 1
    assert len(To) == len(From) == 3
    xyzPnts = [ge(From, To, t, re) for t in np.linspace(0, 1, npts)]
    return xyzPnts


def loxPoint(From, To, t, re=GRS80):
    """
    Loxodrome: returns the geocentric coordinates of the point on the loxodrome
        from From to To at parameter t.
    From: numpy array [x, y, z] of geocentric coordinates for the starting point.
    To: numpy array [x, y, z] of geocentric coordinates for the ending point.
    t: float in the interval [0, 1].
    RE: ReferenceEllipsoid object.
    """
    ecc = re.ecc
    # find mercator coordinates for From
    lonFrom, latFrom = re.XYZ_to_LBH(From)[:2]
    xa = re.a * lonFrom
    sinLatFrom = sin(latFrom)
    qa = arctanh(sinLatFrom) - ecc * arctanh(ecc * sinLatFrom)
    ya = re.a * qa
    fromM = np.array([xa, ya])
    # find mercator coordinates for To
    lonTo, latTo = re.XYZ_to_LBH(To)[:2]
    xb = re.a * lonTo
    sinLatTo = sin(latTo)
    qa = arctanh(sinLatTo) - ecc * arctanh(ecc * sinLatTo)
    yb = re.a * qa
    toM = np.array([xb, yb])
    # linear interpolation to find the point in the Mercator coordinates
    px, py = fromM + t * (toM - fromM)
    qp = py / re.a
    # reverse Mercator to get back to LBH
    f = lambda s: s - (tanh(qp + ecc * arctanh(ecc * s)))
    initialGuess = arcsin(tanh(qp))
    sinphi = newton(f, initialGuess)
    phi = arcsin(sinphi)
    return px/re.a, phi, 0


def loxodrome(From: np.array, To: np.array, re=GRS80, npts=50):
    """
    returns an array of npts (lon, lat, h) or XYZ coordinates on the normal
    section starting at From, ending at To. (The h value is
    always 0 because h = 0 by definition.)
    From and To are numpy arrays of geocentric coordinates (m).
    coordinateSysName is a flag indicating the coordinate system for the results.
    LBH means lon, lat, h; XYZ means geocentric Cartesian.
    """
    assert npts > 1
    assert len(To) == len(From) == 3
    lbhPnts = [loxPoint(From, To, t, re) for t in np.linspace(0, 1, npts)]
    return lbhPnts




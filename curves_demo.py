"""
Geometrical Geodesy Curves: examples
Thomas H Meyer
Department of Natural Resources and the Environment
University of Connecticut
6-March-2024
Functions for drawing curve of alignments, normal sections, great ellipses, loxodromes
"""

import numpy as np
from curves import *
from matplotlib import pyplot as plt


# (10,20) to (-30, -40)
FROM = {'LBH':np.array([np.deg2rad(10), np.deg2rad(20), 0])}
FROM['XYZ'] = np.array(GRS80.LBH_to_XYZ(FROM['LBH']))

TO = {'LBH':np.array([np.deg2rad(-30), np.deg2rad(-40), 0])}
TO['XYZ'] = GRS80.LBH_to_XYZ(TO['LBH'])

"""
CoAPoint(From: np.array, To: np.array, t: float) -> Tuple[float, float, float]
"""

print( GRS80.LBH_to_XYZ(CoAPoint(FROM['XYZ'], TO['XYZ'], 0)) - FROM['XYZ'])
print( GRS80.LBH_to_XYZ(CoAPoint(FROM['XYZ'], TO['XYZ'], 1)) - TO['XYZ'])

"""
CoA(From: np.array, To: np.array, re=GRS80, npts=50)
"""

coaLBH_Pts = CoA(FROM['XYZ'], TO['XYZ'], npts=1000)
lonPts, latPts, hPts = np.transpose(coaLBH_Pts)
C, H, U, V, W = coaConstants_deakin(FROM['XYZ'], TO['XYZ'], GRS80)
coaLons = np.array([coaPoint_deakin(lat, C, U, H, V, W) for lat in latPts])
difs = coaLons - lonPts
print(min(difs), max(difs))

"""
planar plot of coa
"""

plt.plot(np.rad2deg(FROM['LBH'][0]), np.rad2deg(FROM['LBH'][1]), 'k+', label="EndPoint")
plt.plot(np.rad2deg(TO['LBH'][0]), np.rad2deg(TO['LBH'][1]), 'k+')

plt.plot(np.rad2deg(lonPts), np.rad2deg(latPts), 'k', label="CoA")

"""
planar plot of normal section, FROM to TO
"""
nsXYZF_Pts = normalSection(FROM['XYZ'], TO['XYZ'], npts=1000)
nsFL, nsFB, nsFH = np.transpose([GRS80.XYZ_to_LBH(xyz) for xyz in nsXYZF_Pts])
plt.plot(np.rad2deg(nsFL), np.rad2deg(nsFB), 'b-', label="NS(forward)")

"""
planar plot of normal section, TO to FROM
"""

nsXYZR_Pts = normalSection(TO['XYZ'], FROM['XYZ'], npts=1000)
nsRL, nsRB, nsRH = np.transpose([GRS80.XYZ_to_LBH(xyz) for xyz in nsXYZR_Pts])
plt.plot(np.rad2deg(nsRL), np.rad2deg(nsRB), 'b-', label="NS(Reverse)")

"""
planar plot of great ellipse, TO to FROM
"""

geXYZR_Pts = greatEllipse(TO['XYZ'], FROM['XYZ'], npts=1000)
geRL, geRB, geRH = np.transpose([GRS80.XYZ_to_LBH(xyz) for xyz in geXYZR_Pts])
plt.plot(np.rad2deg(geRL), np.rad2deg(geRB), 'g', label="Great Ellipse")

"""
planar plot of loxodrome, TO to FROM
"""

loxLBH_Pts = loxodrome(TO['XYZ'], FROM['XYZ'], npts=1000)
loxRL, loxRB, loxRH = np.transpose(loxLBH_Pts)
plt.plot(np.rad2deg(loxRL), np.rad2deg(loxRB), 'm', label="Loxodrome")

plt.legend()
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.title("Curve of Alignment, Normal Sections,\nGreat Ellipse, Loxodrome")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
xPts, yPts, zPts = np.transpose(geXYZR_Pts)
ax.plot(xPts, yPts, zPts, c='k')
xPts, yPts, zPts = np.transpose(nsXYZF_Pts)
ax.plot(xPts, yPts, zPts, c='r')
xPts, yPts, zPts = np.transpose([GRS80.LBH_to_XYZ(lbh) for lbh in coaLBH_Pts])
ax.plot(xPts, yPts, zPts, c='b')

plt.show()
import numpy as np
import pysixtrack
import CollimationToolKit as ctk

#-------------------------------------------------------------------------------
#--- basic foil class with default scatter function -------------------------
#------------ should act like LimitRect -------------------------------------
#-------------------------------------------------------------------------------
def compare_arrays(a,b):
    if a.shape == b.shape:
        if (a == b).all():
            return True
    return False
    

rect_aperture = pysixtrack.elements.LimitRect(
    min_x=-1e-2, max_x=2e-2, min_y=-0.5e-2, max_y=2.5e-2
)
foil_aperture = ctk.elements.LimitFoil(
    min_x=-1e-2, max_x=2e-2, min_y=-0.5e-2, max_y=2.5e-2
)

p_foil = pysixtrack.Particles()
N_part = 10000
p_foil.x = np.random.uniform(low=-3e-2, high=3e-2, size=N_part)
p_foil.y = np.random.uniform(low=-3e-2, high=3e-2, size=N_part)
p_foil.state = np.ones_like(p_foil.x, dtype=np.int)

p_rect = p_foil.copy()

foil_aperture.track(p_foil)
assert not compare_arrays(p_foil.state, p_rect.state), "Particles not affacted by tracking"

rect_aperture.track(p_rect)
# LimitFoil with default scatter function should act like LimitRect
assert compare_arrays(p_foil.state, p_rect.state), "Particles after tracking are not identical"


#-------------------------------------------------------------------------------
#--- basic foil class with testing scatter function -------------------------
#-------------------------------------------------------------------------------
foil_min_x = -0.11
stripperfoil_test = ctk.elements.LimitFoil(
        min_x=foil_min_x,
        func_scatter=ctk.elements.test_strip_ions)

p_testscatter = pysixtrack.Particles(q0=28, mass0 = 238.02891*931.49410242e6)
p_testscatter.x = -0.12
p_testscatter.y = 0.02
p_testscatter.state = 1
p_testscatter.Z = 92

assert p_testscatter.qratio == 1.0

stripperfoil_test.track(p_testscatter)

assert p_testscatter.qratio == (p_testscatter.Z-1)/p_testscatter.q0



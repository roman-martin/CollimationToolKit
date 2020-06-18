import numpy as np
import shutil
import pytest
import pysixtrack
import CollimationToolKit as ctk


#-------------------------------------------------------------------------------
#--- basic foil class with default scatter function -------------------------
#------------ should act like LimitRect -------------------------------------
#-------------------------------------------------------------------------------
def test_foil_default():
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
    assert not np.array_equal(p_foil.state, p_rect.state), "Particles not affected by tracking"

    rect_aperture.track(p_rect)
    # LimitFoil with default scatter function should act like LimitRect
    assert np.array_equal(p_foil.state, p_rect.state), "Particles after tracking are not identical"


#-------------------------------------------------------------------------------
#--- basic foil class with testing scatter function -------------------------
#-------------------------------------------------------------------------------
foil_min_x = -0.11


def test_foil_testfunction():

    stripperfoil_test = ctk.elements.LimitFoil(
            min_x=foil_min_x,
            scatter=ctk.elements.test_strip_ions)

    p_testscatter = pysixtrack.Particles(q0=28, mass0 = 238.02891*931.49410242e6)
    p_testscatter.x = -0.12
    p_testscatter.y = 0.02
    p_testscatter.state = 1
    p_testscatter.Z = 92

    assert p_testscatter.qratio == 1.0

    stripperfoil_test.track(p_testscatter)

    assert p_testscatter.qratio == (p_testscatter.Z-1)/p_testscatter.q0
    assert p_testscatter.chi == p_testscatter.qratio



#-------------------------------------------------------------------------------
#--- Foil with GLOBAL charge exchange code as scatter function---------------
#-------------------------------------------------------------------------------

stripperfoil_GLOBAL = ctk.elements.LimitFoil(
        min_x=foil_min_x,
        scatter=ctk.ScatterFunctions.GLOBAL)

Ekin = 200.0e6*238
uranium_mass = 238.0507884 * 931.49410242e6



def test_GLOBAL():
    if not shutil.which('global'):
        pytest.skip("GLOBAL executable not found in PATH")
        
    p_GLOBAL = pysixtrack.Particles(q0=28, mass0 = uranium_mass,
                                    x = -0.12, y = 0.02, 
                                    p0c = np.sqrt(Ekin**2 + 2*Ekin*uranium_mass))
    p_GLOBAL.state = 1
    p_GLOBAL.Z = 92

    assert p_GLOBAL.qratio == 1.0
    assert p_GLOBAL.delta == 0.0

    stripperfoil_GLOBAL.track(p_GLOBAL)

    assert p_GLOBAL.qratio <= (92-0)/28
    assert p_GLOBAL.qratio > (92-6)/28
    assert p_GLOBAL.chi == p_GLOBAL.qratio
    assert p_GLOBAL.delta < -0.07



#-------------------------------------------------------------------------------
#--- Foil with GLOBAL as scatter function for mutliple particles-------------
#-------------------------------------------------------------------------------
def test_GLOBAL_vec():
    if not shutil.which('global'):
        pytest.skip("GLOBAL executable not found in PATH")
        
    N_part = 200
    p_vec = pysixtrack.Particles(q0=28, mass0 = uranium_mass,
                                    p0c = np.sqrt(Ekin**2 + 2*Ekin*uranium_mass))
                                    
    p_vec.x = np.random.uniform(low=-3e-1, high=3e-1, size=N_part)
    p_vec.y = np.random.uniform(low=-3e-1, high=3e-1, size=N_part)
    p_vec.state = np.ones_like(p_vec.x, dtype=np.int)
    p_vec.qratio = np.ones_like(p_vec.x, dtype=np.float)
    p_vec.delta = np.zeros_like(p_vec.x, dtype=np.float)
    p_vec.Z = np.ones_like(p_vec.x, dtype=np.int) * 92



    stripperfoil_GLOBAL.track(p_vec)


    for ii in range(len(p_vec.x)):
        if p_vec.x[ii] <= foil_min_x:
            assert p_vec.qratio[ii] <= (92-0)/28
            assert p_vec.qratio[ii] > (92-6)/28
            assert p_vec.delta[ii] < -0.07
        else:
            assert p_vec.qratio[ii] == 1.0
            assert p_vec.delta[ii] == 0.0
    assert np.array_equal(p_vec.chi, p_vec.qratio)


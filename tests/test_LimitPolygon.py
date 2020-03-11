import numpy as np
import pysixtrack
import CollimationToolKit as ctk
from mpmath import mp



aper_min_x = -0.04
aper_max_x = 0.03
aper_min_y = -0.02
aper_max_y = 0.01
mypolygon = np.array([  [aper_max_x, aper_max_y],
                        [aper_max_x, aper_min_y],
                        [aper_min_x, aper_min_y],
                        [aper_min_x, aper_max_y]]
                    ).transpose()

poly_aper = ctk.elements.LimitPolygon(aperture = mypolygon)
rect_aper = pysixtrack.elements.LimitRect(
    min_x=aper_min_x,
    max_x=aper_max_x,
    min_y=aper_min_y,
    max_y=aper_max_y
)



N_part = 20000
#-------------------------------------------------------
#----Test scalar----------------------------------------
#-------------------------------------------------------
def test_scalar():
    p_scalar = pysixtrack.Particles()
    passed_particles_x_poly = []
    passed_particles_y_poly = []
    passed_particles_x_rect = []
    passed_particles_y_rect = []
    lost_particles_x_poly = []
    lost_particles_y_poly = []
    lost_particles_x_rect = []
    lost_particles_y_rect = []
    for n in range(N_part):
        p_scalar.x = (np.random.rand()-0.5) * 2.*8.5e-2
        p_scalar.y = (np.random.rand()-0.5) * 2.*8.5e-2
        p_scalar.state = 1

        poly_aper.track(p_scalar)
        if p_scalar.state == 1:
            passed_particles_x_poly += [p_scalar.x]
            passed_particles_y_poly += [p_scalar.y]
        else:
            lost_particles_x_poly += [p_scalar.x]
            lost_particles_y_poly += [p_scalar.y]
        # check against LimitRect
        p_scalar.state = 1
        rect_aper.track(p_scalar)
        if p_scalar.state == 1:
            passed_particles_x_rect += [p_scalar.x]
            passed_particles_y_rect += [p_scalar.y]
        else:
            lost_particles_x_rect += [p_scalar.x]
            lost_particles_y_rect += [p_scalar.y]


    assert passed_particles_x_poly == passed_particles_x_rect
    assert passed_particles_y_poly == passed_particles_y_rect
    assert lost_particles_x_poly == lost_particles_x_rect
    assert lost_particles_y_poly == lost_particles_y_rect



#-------------------------------------------------------
#----Test vector----------------------------------------
#-------------------------------------------------------
def compare_arrays(a,b):
    if a.shape == b.shape:
        if (a == b).all():
            return True
    return False

def test_vector():
    p_vec_poly = pysixtrack.Particles()
    p_vec_poly.x = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_vec_poly.y = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_vec_poly.state = np.ones_like(p_vec_poly.x, dtype=np.int)

    p_vec_rect = p_vec_poly.copy()


    poly_aper.track(p_vec_poly)
    rect_aper.track(p_vec_rect)


    assert compare_arrays(p_vec_poly.state,p_vec_rect.state)
    assert compare_arrays(p_vec_poly.x,p_vec_rect.x)
    assert compare_arrays(p_vec_poly.y,p_vec_rect.y)


#-------------------------------------------------------
#----Test mpmath compatibility--------------------------
#-------------------------------------------------------
def test_mpmath_compatibility():
    p_mp = pysixtrack.Particles()
    mp.dps = 25
    p_mp.x = mp.mpf(0.03) - mp.mpf(1e-27)
    p_mp.y = mp.mpf(0.01)
    p_mp.state = 1
    polygon_mp = [[mp.mpf(-0.03),mp.mpf(0.03),mp.mpf(0.03),mp.mpf(-0.03)],
                  [mp.mpf(0.04),mp.mpf(0.04),mp.mpf(-0.04),mp.mpf(-0.04)]]
    aper_elem_mp = ctk.elements.LimitPolygon(aperture = polygon_mp)

    aper_elem_mp.track(p_mp)

    assert p_mp.state == 1

    p_mp.x = mp.mpf(0.03) + mp.mpf(1e-27)
    p_mp.y = mp.mpf(0.01)
    p_mp.state = 1
    aper_elem_mp.track(p_mp)

    assert p_mp.state == 0




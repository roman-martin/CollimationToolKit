#   Copyright 2020 CERN
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import csv
import numpy as np
from pysixtrack import elements as pysixtrack_elements
from CollimationToolKit.elements import LimitPolygon

# The following function is a modified version of iter_from_madx_sequence()
# from pysixtrack.loader_mad.py. It was modified to load apertures provided
# as files to MAD-X as LimitPolygon class elements in pysixtrack.
# You may obtain the original source code at
# https://github.com/SixTrack/pysixtrack
def iter_from_madx_sequence_ctk(
    sequence,
    classes=pysixtrack_elements,
    ignored_madtypes=[],
    exact_drift=False,
    drift_threshold=1e-6,
    install_apertures=False,
):

    if exact_drift:
        myDrift = classes.DriftExact
    else:
        myDrift = classes.Drift
    seq = sequence

    elements = seq.elements
    ele_pos = seq.element_positions()

    old_pp = 0.0
    i_drift = 0
    for ee, pp in zip(elements, ele_pos):

        if pp > old_pp + drift_threshold:
            yield "drift_%d" % i_drift, myDrift(length=(pp - old_pp))
            old_pp = pp
            i_drift += 1

        eename = ee.name
        mad_etype = ee.base_type.name

        if ee.length > 0:
            raise ValueError(f"Sequence {seq} contains {eename} with length>0")

        if mad_etype in [
            "marker",
            "monitor",
            "hmonitor",
            "vmonitor",
            "collimator",
            "rcollimator",
            "elseparator",
            "instrument",
            "solenoid",
            "drift",
        ]:
            newele = myDrift(length=ee.l)

        elif mad_etype in ignored_madtypes:
            pass

        elif mad_etype == "multipole":
            knl = ee.knl if hasattr(ee, "knl") else [0]
            ksl = ee.ksl if hasattr(ee, "ksl") else [0]
            newele = classes.Multipole(
                knl=list(knl),
                ksl=list(ksl),
                hxl=knl[0],
                hyl=ksl[0],
                length=ee.lrad,
            )

        elif mad_etype == "tkicker" or mad_etype == "kicker":
            hkick = [-ee.hkick] if hasattr(ee, "hkick") else []
            vkick = [ee.vkick] if hasattr(ee, "vkick") else []
            newele = classes.Multipole(
                knl=hkick, ksl=vkick, length=ee.lrad, hxl=0, hyl=0
            )

        elif mad_etype == "vkicker":
            newele = classes.Multipole(
                knl=[], ksl=[ee.kick], length=ee.lrad, hxl=0, hyl=0
            )

        elif mad_etype == "hkicker":
            newele = classes.Multipole(
                knl=[-ee.kick], ksl=[], length=ee.lrad, hxl=0, hyl=0
            )
        elif mad_etype == "dipedge":
            newele = classes.DipoleEdge(
                h=ee.h, e1=ee.e1, hgap=ee.hgap, fint=ee.fint
            )

        elif mad_etype == "rfcavity":
            newele = classes.Cavity(
                voltage=ee.volt * 1e6,
                frequency=ee.freq * 1e6,
                lag=ee.lag * 360,
            )

        elif mad_etype == "rfmultipole":
            newele = classes.RFMultipole(
                voltage=ee.volt * 1e6,
                frequency=ee.freq * 1e6,
                lag=ee.lag * 360,
                knl=ee.knl[:],
                ksl=ee.ksl[:],
                pn=[v * 360 for v in ee.pnl],
                ps=[v * 360 for v in ee.psl],
            )

        elif mad_etype == "crabcavity":
            newele = classes.RFMultipole(
                frequency=ee.freq * 1e6,
                knl=[ee.volt / sequence.beam.pc],
                pn=[ee.lag * 360 - 90],
            )

        elif mad_etype == "beambeam":
            if ee.slot_id == 6 or ee.slot_id == 60:
                # BB interaction is 6D
                newele = classes.BeamBeam6D(
                    phi=0.0,
                    alpha=0.0,
                    x_bb_co=0.0,
                    y_bb_co=0.0,
                    charge_slices=[0.0],
                    zeta_slices=[0.0],
                    sigma_11=1.0,
                    sigma_12=0.0,
                    sigma_13=0.0,
                    sigma_14=0.0,
                    sigma_22=1.0,
                    sigma_23=0.0,
                    sigma_24=0.0,
                    sigma_33=0.0,
                    sigma_34=0.0,
                    sigma_44=0.0,
                    x_co=0.0,
                    px_co=0.0,
                    y_co=0.0,
                    py_co=0.0,
                    zeta_co=0.0,
                    delta_co=0.0,
                    d_x=0.0,
                    d_px=0.0,
                    d_y=0.0,
                    d_py=0.0,
                    d_zeta=0.0,
                    d_delta=0.0,
                )
            else:
                # BB interaction is 4D
                newele = classes.BeamBeam4D(
                    charge=0.0,
                    sigma_x=1.0,
                    sigma_y=1.0,
                    beta_r=1.0,
                    x_bb=0.0,
                    y_bb=0.0,
                    d_px=0.0,
                    d_py=0.0,
                )
        elif mad_etype == "placeholder":
            if ee.slot_id == 1:
                newele = classes.SpaceChargeCoasting(
                    line_density=0.0,
                    sigma_x=1.0,
                    sigma_y=1.0,
                    length=0.0,
                    x_co=0.0,
                    y_co=0.0,
                )
            elif ee.slot_id == 2:
                newele = classes.SpaceChargeBunched(
                    number_of_particles=0.0,
                    bunchlength_rms=0.0,
                    sigma_x=1.0,
                    sigma_y=1.0,
                    length=0.0,
                    x_co=0.0,
                    y_co=0.0,
                )
            else:
                newele = myDrift(length=ee.l)
        else:
            raise ValueError(f'MAD element "{mad_etype}" not recognized')

        yield eename, newele

        if install_apertures & (min(ee.aperture) > 0):
            if ee.apertype == "rectangle":
                newaperture = pysixtrack_elements.LimitRect(
                    min_x=-ee.aperture[0],
                    max_x=ee.aperture[0],
                    min_y=-ee.aperture[1],
                    max_y=ee.aperture[1],
                )
            elif ee.apertype == "ellipse":
                newaperture = pysixtrack_elements.LimitEllipse(
                    a=ee.aperture[0], b=ee.aperture[1]
                )
            elif ee.apertype == "circle":
                newaperture = pysixtrack_elements.LimitEllipse(
                    a=ee.aperture[0], b=ee.aperture[0]
                )
            elif ee.apertype == "rectellipse":
                newaperture = pysixtrack_elements.LimitRectEllipse(
                    max_x=ee.aperture[0],
                    max_y=ee.aperture[1],
                    a=ee.aperture[2],
                    b=ee.aperture[3],
                )
            elif ee.apertype == "rectellipse":
                newaperture = pysixtrack_elements.LimitRectEllipse(
                    max_x=ee.aperture[0],
                    max_y=ee.aperture[1],
                    a=ee.aperture[2],
                    b=ee.aperture[3],
                )
            else:
                raise ValueError("Aperture type not recognized")

            yield eename + "_aperture", newaperture
        # modifications to load LimitPolygon
        elif install_apertures & os.path.isfile(ee.apertype):
            with open(ee.apertype,'r') as aper_file:
                aper_reader = csv.reader(aper_file, delimiter=' ',
                                         skipinitialspace=True,
                                         quoting=csv.QUOTE_NONNUMERIC)
                #reader -> list and non-numpy transposing
                aper_coords = list(map(list, zip(*aper_reader)))
            newaperture = LimitPolygon(
                aperture = aper_coords
            )
            yield eename + "_aperture", newaperture
        # /modifications to load LimitPolygon

    if hasattr(seq, "length") and seq.length > old_pp:
        yield "drift_%d" % i_drift, myDrift(length=(seq.length - old_pp))


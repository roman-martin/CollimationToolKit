def default_scatter(self, particle, idx=[]):
        # default behaviour: black hole
        if not hasattr(particle.state, "__iter__"):
            particle.state = 0
            return "Particle lost"
        else:
            particle.state[idx] = 0
            particle.remove_lost_particles()
            if len(particle.state) == 0:
                return "All particles lost"


def test_strip_ions(self, particle, idx=[]):
    if not hasattr(particle, "Z"):
        raise AttributeError("""Partices have no atomic number Z
                                provide Z via e.g.
                                >>> particle.Z = 92
                            """)
    if not hasattr(particle.state, "__iter__"):
        particle.qratio = (particle.Z-1) / particle.q0
    else:
        particle[idx].qratio = np.divide(particle.Z[idx]-1, particle.q0[idx])
        tmp_qratio = particle.qratio    # this is needed to...
        particle.qratio = tmp_qratio    # ... trigger the qratio setter

from pysixtrack.elements import Element
from operator import sub, mul
from CollimationToolKit.ScatterFunctions import default_scatter, test_strip_ions
import numpy as np
import types



#-------------------------------------------------------------------------------
#------- Polygonal aperture class -------------------------------------------
#-------------------------------------------------------------------------------

class LimitPolygon(Element):
    # the input coords must be ordered, i.e the lines connecting 
    # neighbours must be the sides of the polygon
    #
    # convention: aperture[0] = x-coords, aperture[1] = y-coords
    _description = [
        (
            "aperture", 
            "m",
            "Array with coords of polygon vertices",
            lambda: [[-1.0,1.0,1.0,-1.0], [1.0,1.0,-1.0,-1.0]]
        )
    ]
      
    @staticmethod
    def np_is_right_of(line_start, line_end, point):
        # decides if point is left of line going through line_start and 
        # line_end by checking the 3rd dimension direction of the cross 
        # product of the involved vectors
        def __crossproduct_z__(vector1, vector2):
            return np.multiply(vector1[0],vector2[1]) - np.multiply(vector1[1],vector2[0])
        
        return __crossproduct_z__(line_end-line_start, point-line_start) < 0.0
    
    @staticmethod
    def map_is_right_of(line_start, line_end, point):
        # pedestrian implementation of np_is_right_of()
        def __crossproduct_z__(vector1, vector2):
            z_coord = list(map(sub, list(map(mul,vector1[0],vector2[1])),
                                    list(map(mul,vector1[1],vector2[0]))
                          ))
            return z_coord
        
        def _get_to_length(vec, num):
            for dim in [0,1]:
                value = vec[dim][0]
                vec[dim] = [value for i in range(num)]
            return vec
        
        #make input mappable against each other
        max_len = max(len(line_start[0]), len(line_end[0]), len(point[0]))
        if len(line_start[0]) == 1:
            line_start = _get_to_length(line_start, max_len)
        if len(line_end[0]) == 1:
            line_end = _get_to_length(line_end, max_len)
        if len(point[0]) == 1:
            point = _get_to_length(point, max_len)
        
        vec_line = [list(map(sub,le,ls)) for le,ls in zip(line_end, line_start)]
        vec_line_to_point = [list(map(sub,pt,ls)) for pt,ls in zip(point, line_start)]
        z_coord = __crossproduct_z__(vec_line, vec_line_to_point)
        return [z < 0.0 for z in z_coord]



    def track(self, particle):
        if not hasattr(particle.state, "__iter__"):
            if isinstance(self.aperture, np.ndarray):
                self.aperture = self.aperture.tolist()
            func_array = lambda x: x
            func_expand_dims = lambda x, axis: x
            func_roll = lambda array,shift, axis:   [array[0][-1*shift:] + array[0][:-1*shift],\
                                                    array[1][-1*shift:] + array[1][:-1*shift]]
            func_is_right_of = self.map_is_right_of
            func_output = lambda odd_or_even,tmp1,tmp2: int(odd_or_even)
            func_shape = lambda x: len(x)
        else:
            func_array = np.array
            func_expand_dims = np.expand_dims
            func_roll = np.roll
            
            func_is_right_of = self.np_is_right_of
            func_output = np.where
            func_shape = lambda x: x.shape
            
        
        
        aper_1 = func_expand_dims(self.aperture, axis=2)
        aper_2 = func_roll(aper_1, 1, axis=1)
        # prepare reference point outside aperture
        refpoint = func_array([[1.1*abs(max(aper_1[0]))], [1.1*abs(max(aper_1[1]))]] )
        
        coords = func_array([[particle.x], [particle.y]])
        

        particle_is_right = func_is_right_of(aper_1, aper_2, coords)
        refpoint_is_right = func_is_right_of(aper_1, aper_2, refpoint)
        
        aper_1_is_right = func_is_right_of(coords, refpoint, aper_1)
        aper_2_is_right = func_is_right_of(coords, refpoint, aper_2)
        
                
        lines_intersect = np.where( np.logical_and(
                                       np.logical_xor(particle_is_right,refpoint_is_right),
                                       np.logical_xor(aper_1_is_right,aper_2_is_right)
                                    ),
                                    np.ones(func_shape(aper_1[0]), dtype=int),
                                    np.zeros(func_shape(aper_1[0]), dtype=int)
                                  ) # todo: make sure that end points are included
        num_intersects = np.sum(lines_intersect, axis=0)
        # if num_intersect is odd -> particle is inside aperture
        particle.state = func_output(num_intersects % 2 == 1,
                                     np.ones(num_intersects.shape, dtype=int),
                                     np.zeros(num_intersects.shape, dtype=int)
                                    )
        if hasattr(particle.state, "__iter__"):
            particle.remove_lost_particles()
            if len(particle.state) == 0:
                return "All particles lost"



#-------------------------------------------------------------------------------
#------- Foil class ---------------------------------------------------------
#-------------------------------------------------------------------------------

class LimitFoil(Element):
    _description = [
        ("min_x", "m", "Minimum horizontal aperture", -1.0),
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("min_y", "m", "Minimum vertical aperture", -1.0),
        ("max_y", "m", "Minimum vertical aperture", 1.0),
        ("thickness", "m", "Foil thickness", 0.001),
        ("density", "g/cm^3", "Foil meterial density", 1.86),
        ("Z", "1", "proton number of foil material", 6),
        ("A", "1", "Standard atomic weight of foil material", 12.0096), #...or use the relative atomic mass instead...
        ("scatter", "<function>", "scatter function", "this should be replaced on creation"),
    ]

    def __post_init__(self):
        # the whole problem leading to this __post_init__ is that
        # 1. defaults of type function are weeded out in
        #    pysixtrack.base_classes._default_pro
        # 2. the scatter functions must be bound methods
        if self.scatter == "this should be replaced on creation":
            # if "default" string
            self.scatter = types.MethodType(default_scatter, self)
        else:
            # if scatter keyword argument is given
            if not type(self.scatter) == type(lambda: None):
                raise ValueError("scatter must be function")
            self.scatter = types.MethodType(self.scatter, self)

    
    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            if (x < self.min_x or x > self.max_x
                    or y < self.min_y or y > self.max_y):
                self.scatter(particle)
        else:
            hitting_particles_idx = np.where(
                    (x <= self.min_x)
                    | (x >= self.max_x)
                    | (y <= self.min_y)
                    | (y >= self.max_y)
                )[0]
            self.scatter(particle, idx = hitting_particles_idx)


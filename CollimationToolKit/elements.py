from pysixtrack.elements import Element
from operator import sub, mul
import numpy as np


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
        refpoint = func_array([[1.1*abs(max(aper_1[0]))], [1.1*abs(max(aper_1[1]))]] ) # this line won't work without numpy
        
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


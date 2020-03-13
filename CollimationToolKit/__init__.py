import pysixtrack
from . import elements
from . import ScatterFunctions
from .loader_mad import iter_from_madx_sequence_ctk


# monkey patching iter_from_madx_sequence() to include LimitPolygon loader
pysixtrack.loader_mad.iter_from_madx_sequence_old = pysixtrack.loader_mad.iter_from_madx_sequence
pysixtrack.line.iter_from_madx_sequence_old = pysixtrack.line.iter_from_madx_sequence
pysixtrack.loader_mad.iter_from_madx_sequence = iter_from_madx_sequence_ctk
pysixtrack.line.iter_from_madx_sequence = iter_from_madx_sequence_ctk

"""
The visualize module has submodules that enable the user to visualize and
interact with a variety of data types.

The guitools submodule provides OpenCV2-based methods for casting scalar data
to HSV, quickly rendering images in a window, and selecting points or regions
from image data in an interactive session.

The geoplot submodule contains matplotlib-based methods for generating
animations and figures using a variety of data types, united under a common
format configuration (the plot_spec dictionary).

The TextFormat class acts as an interface for applying ANSI escape-coded
terminal colors a string for more expressive terminal printing.
"""
from .TextFormat import TextFormat

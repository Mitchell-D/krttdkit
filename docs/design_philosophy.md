# krttdkit design philosophy

My goal in developing `krttdkit` is to provide classes and methods
which are reusable at every abstraction level, and to keep

### A functional backbone

Pure-function modules are more efficient and reliable building
blocks for packages with heavily interdependent higher-level
modules since _state-saving is prone to side-effects_. Without
a persistent state that needs to be managed at runtime, it's
easier to build lofty abstractions with compositions of the functions
in the module and its pure-module dependencies.

(add example after migrating `classify` module)

### The role of classes

Classes should have a purposeful need to justify their state-saving
ability. This need is typically found where use-cases for a class are
either *(1)* very specific, as in __products__ like `MOD021KM`, which
enables the user to download, analyze, and plot L1b MODIS granules,
or *(2)* general enough to be useful as a pure abstraction, as in the
`GeosGeom` class for doing sun/satellite geometry calculations.

Classes are most usable when they represent a straightforward
concept with methods that provide useful and intuitive behaviors.
States should change predictably, and attributes visible to the
user should have few side-effects. Transparent static methods should
be used wherever possible. Absolutely minimize the number of visible
state-changing methods, but don't lose time over-generalizing rare
use cases.

### Quick Points
 - Dependencies should be vigilantly minimized
 - New abstraction levels should be added sparingly and purposefully
 - Code is only as good as it is usable. Usability is a combination
   of API clarity and demonstration quality.
 - Minimize side effects and always justify state saving
 - Abide by a well-defined integration procedure for new code.

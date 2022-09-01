
Visualization and Analysis of Measured Optical Wavefronts
=========================================================

Laser beams are often characterized by the phase and intensity
distributions of the propagating electric field, as measured in 
various focal planes and their conjugates. A useful tool in 
such measurements is a wavefront sensor, a camera-based device
with pixelated optical elements.

The principles of operation of an example sensor technology,
such as a `Shack Hartmann Wavefront Sensor <https://en.wikipedia.org/wiki/Shack%E2%80%93Hartmann_wavefront_sensor>`_,
are beyond the scope of this library, but in essence amount to 
visualizing the direction of propagation of incident photons within
a laser beam. Assuming the photons are temporally coherent, their
propagation direction can be used to infer the underlying phase
distribution across a pixelated detector.

This code library has been designed to process and visualize data
from the `Thorlabs WFS series detectors <https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5287>`_.
Intensity and wavefront measurements are exported as CSV files with
specific formatting. These files are parsed, loaded in to NumPy 
arrays and then used for various activities, such as visualization.


Install
-------

We recommend using a virtual environment such as Conda, venv, or
virtualenv. The authors of this particular code base make use of 
`virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/#>`_
to facilitate the setup and management of their own environments.

From sources
````````````

To install system-wide, noting the path to the src since no wheels
exist on PyPI, use::

   pip install ./wavefronts

If you intend to edit the code and want the import calls to reflect
those changes, install in developer mode::

   pip install -e wavefronts

If you don't want a global installation (i.e. if multiple users will
engage with and/or edit this library) and you don't want to use venv
or some equivalent::

   pip install -e wavefronts --user

where pip is pip3 for Python3 (tested on Python 3.9.10). Be careful 
NOT to use ``sudo``, as the latter two installations make a file
``easy-install.pth`` in either the global or the local directory
``lib/python3.X/site-packages/easy-install.pth``, and sudo will
mess up the permissions of this file such that uninstalling is very
complicated.


Uninstall
---------

If installed without ``sudo`` as instructed, uninstalling should be 
as easy as::

   pip uninstall wavefronts

If installed using ``sudo`` and with the ``-e`` and ``--user`` flags, 
the above uninstall will encounter an error.

Navigate to the file ``lib/python3.X/site-packages/easy-install.pth``, 
located either at  ``/usr/local/`` or ``~/.local`` and ensure there
is no entry for ``wavefronts``.


License
-------

The package is distributed under an open license (see LICENSE file for
information).


Authors
-------

Charles Blakemore (chas.blakemore@gmail.com)
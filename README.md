# openscad-python-wrapper
Wraps OpenSCAD's python support with some helpful additional features

Notes:

I'm running OpenSCAD on Fedora Linux using the AppImage Development Snapshots.
YMMV if you are using another OS/Platform/Version of OpenSCAD.

OpenSCAD's Python support is pretty rough right now. Configuring it to make
it useful is a pain.  The AppImage version of OpenSCAD adds some additional challenges.
Here's some tips.

First, create a venv workspace. In OpenSCAD:
`File->Python->Create Virtual Environment`

I chose to put this virtual environment in a directory called `OscadPy`

Quit OpenSCAD

Next, install numpy and any other Python libs you might find helpful.
`PIP_TARGET=~/OscadPy/lib/python3.10/site-packages/ OpenSCAD-2025.06.22.ai25960-x86_64.AppImage --python-module pip install numpy`

Running pip this way is required because of how the AppImage is packaged. The `python` executable isn't installed to
the venv, but instead a link is placed there to `/sys/self/exe` which is useless to anyone but the running copy of OpenSCAD.

Specifying `PIP_TARGET` is necessary, otherwise it installs to your default local python library dir instead of the venv.

Now run OpenSCAD
`PYTHONPATH=/path/to/your/working/directory/ OpenSCAD-2025.06.22.ai25960-x86_64.AppImage`

`PYTHONPATH` Needs to be specified because OpenSCAD won't import modules from the directory you load your
working files from. And you may want to split your work into multiple files.

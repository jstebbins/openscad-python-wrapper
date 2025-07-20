import os
import sys
from shapes import *
from transforms import *
from dataclasses import dataclass
from sweep import *

print(sys.path)

fn = 0
fa = 1
fs = 1

def test_sphere():
    c = cube(4, center=True).color("blue")
    t1 = sphere(d=30)
    t1 |= c.attach(t1, "top")
    t1 |= c.attach(t1, "bottom")
    t1 |= c.attach(t1, "left")
    t1 |= c.attach(t1, "right")
    t1 |= c.attach(t1, "front")
    t1 |= c.attach(t1, "back")

    return t1

def test_cylinder():
    c = cube(4, center=True).color("blue")
    t1 = cylinder(h=50, r=[10, 15], ends=EdgeTreatment(round=4))
    t1 |= c.attach(t1, where="right")

    t2 = cylinder(h=50, r=[15, 10], ends=EdgeTreatment(round=-4)).right(35)
    t2 |= c.attach(t2, where="top")

    t3 = cylinder(h=50, r=[15, 10], ends=EdgeTreatment(chamf=-4)).left(35)
    t3 |= c.attach(t3, where="left")

    return t1 | t2 | t3

def test_prisnoid():
    c = cube(4, center=True).color("blue")
    t1 = prisnoid(40, 20, 6, 4, 25, shift=[-5, -5])
    t2 = wireframe(t1, unify=True).fwd(45)
    t1 |= c.attach(t1, where="right")
    t1 |= c.attach(t1, where="top")

    return t1 | t2

def test_rounding_sweep():
    l = 50
    d = 40
    h = 30 + 0.02
    r = [18, -8]

    c = circle(d=d)
    l1 = l - d
    print(l1)
    shape = hull(c.left(l1 / 2) | c.right(l1 / 2))
    mesh = rounding_sweep(shape, h, r)
    t1 = polyhedron(mesh[0], mesh[1]).up(5)

    l = 50 + 10
    d = 40 + 10
    h = 30 + 5

    c = circle(d=d)
    l1 = l - d
    shape = hull(c.left(l1 / 2) | c.right(l1 / 2))
    mesh = rounding_sweep(shape, h, r)
    t2 = polyhedron(mesh[0], mesh[1])

    return t2 - t1

def test_rotate_sweep():
    def sweepShapeExample(context):
        if not hasattr(context, "radius"):
            # This is a little dirty ;)
            # I'm hijacking the context that rotate_sweep makes for itelf
            # and stashing my own parameters in it.
            context.radius = 5
            context.fn     = 40

        shape = circle(r=context.radius, fn=context.fn).right(20)
        context.radius += fs / 60
        return shape

    s = rotate_sweep(sweepShapeExample, angle=360)
    t1 = polyhedron(points=s[0], faces=s[1])

    shape = circle(d=10, fn=40).right(20)
    s = rotate_sweep(shape, angle=360)
    t2 = polyhedron(points=s[0], faces=s[1]).up(20)

    return t1 | t2

def test_path_sweep():
    path = circle(r=26).mesh().points
    shape = square(6, center=True)
    s = path_sweep(shape, path, closed=True)
    t1 = polyhedron(points=s[0], faces=s[1])

    path = Points([[ theta / 10, 10 * np.sin(np.radians(theta))] for theta in range(-180, 180, 5)])
    shape = square(6, center=True)
    s = path_sweep(shape, path, closed=False)
    t2 = polyhedron(points=s[0], faces=s[1])

    return t1 | t2

def test_sweep():
    """
    sweep usage examples
    """
    @dataclass()
    class SweepContextExample():
        """
        Example transform context for sweep
        """
        # Context for the transform
        index           : int    = 0
        stop            : int    = 25
        sweep_radius    : float  = 75
        angle           : float  = np.radians(40)

        # Context for the shape
        radius          : float  = 5
        fn              : int    = 40

    def sweepTransformExample(context):
        if context.index > context.stop:
            return None

        m = (Affine.around_center(cp=[0, context.sweep_radius, 0],
                                  m=Affine.xrot3d(-context.angle * context.index / 25)) @
             Affine.scale3d([1 + context.index / 25, 2 - context.index / 25, 1]))
        context.index += 1
        return m

    def sweepShapeExample(context):
        shape = circle(r=context.radius, fn=context.fn)
        context.radius += fs / 20
        return shape

    # sweep with transform callback and shape callback
    context = SweepContextExample()
    s = sweep(sweepShapeExample, sweepTransformExample, context=context)

    t1 = polyhedron(points=s[0], faces=s[1]).fwd(20)

    # sweep with transform list and static shape
    radius = 75
    angle = np.radians(40)
    shape = circle(r=5, fn=40)
    T = [
        Affine.around_center(cp=[0, radius, 0], m=Affine.xrot3d(-angle * ii / 25)) @
            Affine.scale3d([1 + ii / 25, 2 - ii / 25, 1])
        for ii in range(25 + 1)
    ]
    s = sweep(shape, T)

    t2 = polyhedron(points=s[0], faces=s[1]).back(20)

    return t1 | t2

def test_justify():
    t1 = cube(20, center=True).color("green")
    c1 = cube([15, 5, 7], center=True)
    c2 = cube(5, center=True).color("blue")
    c1 |= c2.attach(c1, "top")
    t1 |= c1.attach(t1, "right").justify("left")

    return t1

def test_composition():
    class Tube(Composition):
        def __init__(self, tag):
            super().__init__()
            self.tube_tag = tag

        def build(self):
            """
            Build the object.

            Called by Composition.compose()
            """

            l       = 120
            r_in    = 15
            r_out   = 20

            # Body
            c1 = cylinder(l=l, r=r_out).attach(self, where="top", justify="bottom")

            c2 = cylinder(l=l+0.2, r=r_in).down(.1).attach(c1, "bottom", justify="bottom", inside=True)

            # Create a tube that is a difference of the above
            # Note that when we punch the tube through the enclosure, we will
            # need to compose both the tube and the void with the enclosure
            # in order to add the tube and make the hole
            t = c1 - c2

            # Now tag our components for use with compose()
            t.tag(self.tube_tag)
            c2.tag("remove")

            self.objects    = [t, c2]

        def compose(self, parent, operations=None):
            if operations is None:
                # Default operations for this composition
                operations = Composition.AddRemoveKeep
            return super().compose(parent, operations)

    t1 = cube(200, center=True) - cube([170, 185, 170]).translate([0, -8, 0])

    p1 = Tube("add").attach(t1, "left", inside=True)
    p2 = Tube("keep").attach(t1, "back", inside=True)
    p3 = Tube("add").left(50).attach(t1, "back", inside=True)
    #p = p1 + p2 + p3       # One way to agregate compositions
    p1.append([p2, p3])     # Another way to agregate compositions
    t1 = p1.compose(t1)

    return t1

def test_attach():
    c1 = cube(15, center=True).rotate([0, 0, 0])
    c2 = cube([5, 10, 15], center=True).color("blue").up(0).attach(c1, where="bottom", justify="right", inside=True)
    t1 = c1 | c2

    return t1

def test_plot():

    @dataclass()
    class PlotContextExample():
        """
        Example transform context for sweep
        """
        amplitude   : float = None
        wavelength  : float = None

    def plotFuncExample(x, y, context):
        scale = tau / context.wavelength
        return context.amplitude / 2 * np.sin(scale * x) + context.amplitude / 2 * np.sin(scale * y)

    width   = 40
    depth   = 20
    wavelen = 10
    height  = 5

    context = PlotContextExample(amplitude=height, wavelength=wavelen)
    width = int(width / fs + 1) * fs
    range_x = np.arange(-width / 2, width / 2, fs)
    range_y = np.arange(-depth / 2, depth / 2, fs)
    p = plot3d(plotFuncExample, range_x, range_y, base=0, context=context)

    t1 = polyhedron(points=p[0], faces=p[1])

    return t1

def run_test(test):
    t = test.func().translate(test.pos)
    prof_time(f"    {test.name}")
    return t

def run_enabled_tests(tests):

    print("Testing...")
    u = []
    for test in tests:
        if test.enabled:
            u.append( run_test(test) )
    prof_time("All", final=True)
    return u

def find_and_run_test(tests, name):
    """
    Find a test by name and run it.

    The name may be abbreviated and the first matching test will be chosen
    """

    u = []
    for test in tests:
        if test.name.lower().startswith(name.lower()):
            u.append( run_test(test) )
    return u

def run_tests():

    @dataclass()
    class Test():
        """
        Example transform context for sweep
        """
        name      : str     = "Test"
        enabled   : bool    = False
        func      : ...     = None
        pos       : list    = (0, 0, 0)

    tests = [
        Test(name="Sweep",          enabled=True,  func=test_sweep,             pos=[  0,   0,   0]),
        Test(name="Path Sweep",     enabled=True,  func=test_path_sweep,        pos=[-50,   0,   0]),
        Test(name="Rotate Sweep",   enabled=True,  func=test_rotate_sweep,      pos=[ 50,   0,   0]),
        Test(name="Rounding Sweep", enabled=True,  func=test_rounding_sweep,    pos=[-50,   0, -50]),
        Test(name="Prisnoid",       enabled=True,  func=test_prisnoid,          pos=[  0, -50,   0]),
        Test(name="Cylinder",       enabled=True,  func=test_cylinder,          pos=[  0,  60,   0]),
        Test(name="Sphere",         enabled=True,  func=test_sphere,            pos=[ 50,   0,  50]),
        Test(name="Justify",        enabled=True,  func=test_justify,           pos=[  0,   0, -40]),
        Test(name="Composition",    enabled=False, func=test_composition),
        Test(name="Attach",         enabled=False, func=test_attach),
        Test(name="Plot",           enabled=True,  func=test_plot,              pos=[  0,   0, -60]),
    ]

    u = run_enabled_tests(tests)
    #u = find_and_run_test(tests, "prisn")

    show(u)

if __name__ == "__main__":
    run_tests()

import os
print(os.getcwd())
# Fix search path since OpenSCAD doesnt' add CWD
os.sys.path.insert(0, os.getcwd())

from shapes import *
from transforms import *
from dataclasses import dataclass
from sweep import *

fn = None       
fa = 3          
fs = 3  

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
    t1 |= c.attach(t1, where=RT)
    t1 |= c.attach(t1, where=TP)

    faces = t1.faces()
    t2 = polyhedron(points=faces.points, faces=faces.faces).wireframe().fwd(45)

    return t1 | t2

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

def run_tests():
    print("Testing...")
    u = test_sweep()
    u |= test_path_sweep().left(50)
    u |= test_rotate_sweep().right(50)
    u |= test_prisnoid().fwd(50)
    u |= test_cylinder().back(60)
    u |= test_sphere().right(50).up(50)
    u.show()

if __name__ == "__main__":
    run_tests()

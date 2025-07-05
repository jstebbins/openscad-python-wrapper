import openscad as scad
from transforms import *
from dataclasses import dataclass
import copy
import functools

fn = None
fa = 2
fs = 2

RAD_90  = np.pi / 2
RAD_45  = RAD_90 / 2
RAD_180 = np.pi
RAD_270 = RAD_180 + RAD_90
tau = 2 * np.pi

def is_tup(var):
    """
    Check if a variable is a tuple or a list

    Note: 'isinstance' is not used in this case because I don't want to
          identify classes derived from these as a 'tup'
    """

    return type(var) == tuple or type(var) == list

def tup(val, dim, fill=None):
    """
    Convert a scaler or tupple/list to a tupple of a defined dimension

    val     -   Value to convert
    dim     -   Dimension of the output tuple
    fill    - Value to fill new elements with. Repeat last element if fill is None
    """

    if val is None: return None
    if is_vec(val):
        l = len(val)
        fill = val[l-1] if fill is None else fill
        return tuple(val[ii] if ii < l else fill for ii in range(dim))
    return tuple(val for ii in range(dim))

def lst(val, dim, fill=None):
    """
    Convert a scaler or tupple/list to a list of a defined dimension

    Note:   I would have just used 'tup' but OpenSCAD does not permit 'tuple' where
            lists are expected as arguments

    val     -   Value to convert
    dim     -   Dimension of the output tuple
    fill    - Value to fill new elements with. Repeat last element if fill is None
    """

    if val is None: return None
    if is_vec(val):
        l = len(val)
        fill = val[l-1] if fill is None else fill
        return tuple(val[ii] if ii < l else fill for ii in range(dim))
    return [val for ii in range(dim)]

def center_arc(r, center, angle_range):
    """
    Create a set of points defining an arc centered on a point with an angle range

    r       - Radius of the arc
    center  - Center point of the arc
    angle_rnage - Tuple defining the start and sweep angles
    """

    start = angle_range[0]
    angle = angle_range[1] - angle_range[0]

    n = int(np.fmax(3, np.ceil(segs(r) * np.fabs(angle) / tau)))
    points = Points(affine=True)
    for ii in range(n):
        theta = start + ii * angle / (n - 1)
        pt = Vector([np.cos(theta), np.sin(theta)]) * r + center
        pt.affine()
        points.append(pt)

    return points

def corner_arc(r, corner):
    """
    Create a set of points defining an arc spanning it's tangent points on 2 corner vectors

    r       - Radius of the arc
    corner  - 2 Vectors defining a corner
    """

    (cp, n, tp1, tp2) = circle_2tan(r, corner)

    dir         = det2d(corner[1] - corner[0], corner[2] - corner[1]) > 0
    corner      = [tp1, tp2] if dir else [tp2, tp1]
    theta_start = np.atan2(corner[0].y - cp.y, corner[0].x - cp.x)
    theta_end   = np.atan2(corner[1].y - cp.y, corner[1].x - cp.x)
    angle       = posmod(theta_end - theta_start, tau)
    angle_range = [theta_start, theta_start + angle] if dir else [theta_start + angle, theta_start]
    return center_arc(r, cp, angle_range)

def arc(r, center=None, angle_range=None, corner=None):
    """
    Create an arc by specifying either a corner tangent or center point with angle range

    center          - Center point. Can not be used with 'corner'
    angle_ranage    - 2-Touple specifying angle start and sweep. Can not be used with 'corner'
    corner          - 2-Touple of Vectors specifying a corner tangent. Can not be used with 'center' or 'angle_range'
    """

    if corner is not None:
        return corner_arc(r, Points(corner, affine=True))
    else:
        center      = [0, 0] if center is None else center
        angle_range = [0, tau] if angle_range is None else angle_range
        return center_arc(r, center, angle_range)

@dataclass()
class EdgeTreatment():
    """
    EdgeTreatment

    Used for specifying rounding and chamfering of edges

    chamf   - length of chamfer
    chang   - angle of chamfer
    round   - radius of rounding
    """

    chamf   : float = None
    chang   : float = None
    round   : float = None

def cyl_path(r, l, ends=None):
    """
    Create a path for generating a cylinder using 'rotate_extrude'
    The path supports 'EndTreatments' that allow inside and outside chamfering and rounding

    r       - Radius of top and bottom of cylinder. 2-tuple if top and bottom have different radii
    l       - Length of the cylinder.
    ends    - See 'EdgeTreatment'. 2-tuple if top and bottom have different end treatments
    """

    r           = tup(r, 2)
    ends        = tup(ends, 2)

    vang = np.atan2((r[0] - r[1]) / 2, l)
    if ends is not None:
        e = ends[0]
        if e.chamf is not None:
            chang0 = radians(e.chang) if e.chang is not None else RAD_45 + np.sign(e.chamf) * vang
            chamf0 = (e.chamf,
                      np.fabs(law_of_sines(a=e.chamf, A=RAD_180 - chang0 - (RAD_90 - np.sign(e.chamf) * vang), B=chang0)))
        if e.round is not None:
            roundx0 = e.round / np.tan(RAD_45 - vang) if e.round >= 0 else e.round / np.tan(RAD_45 + vang)
        e = ends[1]
        if e.chamf is not None:
            chang1 = radians(e.chang) if e.chang is not None else RAD_45 - np.sign(e.chamf) * vang
            chamf1 = (e.chamf,
                      np.fabs(law_of_sines(a=e.chamf, A=RAD_180 - chang1 - (RAD_90 + np.sign(e.chamf) * vang), B=chang1)))
        if e.round is not None:
            roundx1 = e.round / np.tan(RAD_45 + vang) if e.round >= 0 else e.round / np.tan(RAD_45 - vang)

    pre  = [[0, -l / 2]]
    post = [[0,  l / 2]]
    if ends is None:
        bot = Points(val=[[r[0], -l / 2]]).affine()
    elif ends[0].round is not None:
        bot = arc(r=np.fabs(ends[0].round),
                  corner=[
                            [r[0] - 2 * roundx0,    -l / 2],
                            [r[0],                  -l / 2],
                            [r[1],                   l / 2]
                         ]
        )
    elif ends[0].chamf is not None:
        d = polar_to_xy(chamf0[1], RAD_90 + vang)
        bot = Points(
            val = [
                [r[0] - chamf0[0],  -l / 2],
                [r[0] + d[0],       -l / 2 + d[1]]
              ]
        ).affine()
    else:
        assert False, f"Invalid end treatment specified {ends[0]}"

    if ends is None:
        top = Points(val=[[r[1], l / 2]]).affine()
    elif ends[1].round is not None:
        top = arc(r=np.fabs(ends[1].round),
                  corner=[
                            [r[0],                  -l / 2],
                            [r[1],                   l / 2],
                            [r[1] - 2 * roundx1,     l / 2],
                         ]
        )
    elif ends[1].chamf is not None:
        d = polar_to_xy(chamf1[1], RAD_270 + vang)
        top = Points(
            val = [
                [r[1] + d[0],       l / 2 + d[1]],
                [r[1] - chamf1[0],  l / 2],
              ]
        ).affine()
    else:
        assert False, f"Invalid end treatment specified {ends[1]}"

    return Points.concat_init([pre, bot, top, post])


def extrude_from_to(obj, pt1, pt2):
    pt1 = point3d(pt1)
    pt2 = point3d(pt2)
    rtp = xyz_to_spherical(pt2 - pt1)
    h   = float(rtp[0])
    ay  = np.degrees(float(rtp[2]))
    az  = np.degrees(float(rtp[1]))
    obj = obj.linear_extrude(height=h, center=False).rotate([0, ay, az]).translate(pt1.list())
    return obj

def line(pt1, pt2, d=1):
    circle = scad.circle(d=d)
    return extrude_from_to(circle, pt1, pt2)

def mesh_edges(mesh):
    edges = []
    for face in mesh[1]:
        face_edges = []
        final = prev = face[0]
        for pt in face[1:]:
            edge = [mesh[0][prev], mesh[0][pt]]
            face_edges.append(edge)
            prev = pt
        edge = [mesh[0][prev], mesh[0][final]]
        face_edges.append(edge)
        edges.append(face_edges)
    return edges

@dataclass()
class FaceMetrics():
    """
    Metrics collected from the faces in an 'Object' mesh. These metrics are used to position
    objects relative to another object's face.
    """

    index  : int    = 0     # This element's index into the FaceMetrics list
    normal : Vector = None  # A unit vector perpendicular to the face.
    matrix : Matrix = None  # Transform matrix maping coordinates onto a face.
                            # 'matrix' will transform points to be relative to the center
                            # point of the face with the face 'normal' as the positive Z-axis.
    area   : float  = None  # The area of the face, used for filtering out small faces
    size   : list   = None  # the size of the face, used for filtering out small faces
                            # The size is defined by a square bounding box that contains all face points

class Object():
    ATTACH_LARGE = 0
    ATTACH_NORM  = 1

    """
    Base class for all OpenSCAD objects. This allows adding features that are "missing"
    from the openscad module. I put "missing" in quotes because TBH, for most of these
    features it is probably better to implement them in Python code rather than native.
    So far, I have found that the foundation supplied by the openscad module is enough.

    It would be nice however if the openscad module supplied a base class that I could
    subclass from.  I have hacked around this with some '__getattr__' magic.
    """

    def __init__(self, object=None):
        self.name           = "Base"
        self.oscad_obj      = None
        self.face_cache     = None
        self.hooks          = dict()
        if object is not None:
            self.name           = object.name
            self.oscad_obj      = object.oscad_obj
            self.face_cache     = object.face_cache
            self.hooks          = object.hooks

    def __setattr__(self, attr, value):
        """
        The 'origin' attribute of OpenSCAD objects is writable, but writing to it
        does not change the origin of the object.  So prevent any misunderstandings
        by making it read-only
        """
        if attr == "origin":
            raise AttributeError(f"Read-only attribute '{attr}'")
        super().__setattr__(attr, value)

    def __getattr__(self, attr):
        """
        Pass any attribute references that haven't been defined by me down to OpenSCAD
        if they are defined by the OpenSCAD object.

        Note:   I would have done this with subclassing.  But python openscad does not
                appear to have any base class to subclass off of.
        """
        if hasattr(self.oscad_obj, attr):
            if callable(getattr(self.oscad_obj, attr)):
                def redirect(*args, **kwargs):
                    return getattr(self.oscad_obj, attr)(*args, **kwargs)
                return(redirect)
            else:
                return getattr(self.oscad_obj, attr)
        else:
            assert False, f"Object '{self.name}' has no attribute '{attr}'"

    def dir_oscad_obj(self, msg=None):
        """
        Helper to show the currently available attributes available in OpenSCAD Python objects.
        This unfortunately does not show all that is available because the OpenSCAD module does
        not provide an `__dict__` interface. So it's `origin` attribute is hidden.
        """

        print("Object: ", self.oscad_obj)
        print(f"{msg} - Dir:")
        print(dir(self.oscad_obj))

    def __sub__(self, other):
        """
        Pass operators to OpenSCAD objects.
        Unfortunately, there does not appear to be a catch-all method of handling
        operator overload like there is for method attributes
        """
        res = Object(self)
        res.oscad_obj -= other.oscad_obj
        res.name = f"{self.name} - ({other.name})"
        return res
    def __add__(self, other):
        res = Object(self)
        res.oscad_obj += other.oscad_obj
        res.name = f"{self.name} + ({other.name})"
        return res
    def __or__(self, other):
        res = Object(self)
        res.oscad_obj |= other.oscad_obj
        res.name = f"{self.name} | ({other.name})"
        return res

    def point_cmp(pt1, pt2):
        """
        Compare 2 points

        Helper function for dedup_mesh()
        """
        if pt1 == pt2: return 0
        if pt1[0] < pt2[0]: return -1
        if pt1[0] > pt2[0]: return  1
        if pt1[1] < pt2[1]: return -1
        if pt1[1] > pt2[1]: return  1
        if pt1[2] < pt2[2]: return -1
        if pt1[2] > pt2[2]: return  1

    def edge_cmp(edge1, edge2):
        """
        Compare 2 edges

        Helper function for dedup_mesh()
        """
        c_0_0 = Object.point_cmp(edge1[0], edge2[0])
        if c_0_0 == 0:
            # edges have a common vertex
            return Object.point_cmp(edge1[1], edge2[1])
        return c_0_0

    def orient_edge(self, edge):
        """
        Orient a edge in a uniform direction

        Helper function for dedup_mesh()
        """
        if Object.point_cmp(edge[0], edge[1]) > 0:
            edge[0], edge[1] = edge[1], edge[0]
        return edge

    def dedup_mesh(self, mesh):
        """
        Create a list of edges (2 points connected by an edge)

        The edges are all oriended in a predictable manner to
        make sorting easier.
        """

        face_edges = mesh_edges(mesh)
        edges = [self.orient_edge(edge) for face in face_edges for edge in face]

        # Sort the edges (left to right, front to back, bottom to top)
        edges.sort(key=functools.cmp_to_key(Object.edge_cmp))

        # Remove duplicate edges from the list
        prev = edges[0]
        for edge in edges[1:]:
            if edge == prev:
                edges.remove(prev)
            prev = edge
        return edges

    def wireframe(self):
        """
        Create a wireframe visualization of the Object

        OpenSCAD has a tendency to crash when the wireframe detail gets too high.
        I have improved it's robustness and speed some by eliminating duplicate
        edges from the mesh. But I still experience crashes if fn, fs, fa are too
        fine with an Object that has lots of detail.
        """

        mesh = self.oscad_obj.mesh()
        edges = self.dedup_mesh(mesh)

        wf = Object()
        wf.name = f"{self.name} - Wireframe"

        for edge in edges:
            wf.oscad_obj |= line(edge[0], edge[1])

        return wf

    def translate(self, v):
        res = Object(self)

        # v may be a Vector, make it compatible with OpenSCAD
        res.oscad_obj = res.oscad_obj.translate(list(v))

        return res

    def rotate(self, v):
        res = Object(self)

        # v may be a Vector, make it compatible with OpenSCAD
        res.oscad_obj = res.oscad_obj.rotate(list(v))

        return res

    def color(self, c):
        res = Object(self)

        res.oscad_obj = res.oscad_obj.color(c)

        return res

    def up(self, val):
        """
        Aliases for "translate"
        """
        return self.translate(up(val))
    def down(self, val):
        return self.translate(dn(val))
    def left(self, val):
        return self.translate(lt(val))
    def right(self, val):
        return self.translate(rt(val))
    def fwd(self, val):
        return self.translate(ft(val))
    def back(self, val):
        return self.translate(bk(val))

    def attachment_hook(self, name, matrix):
        """
        Add named hooks that can be used for attachments.
        Hooks are specific attachment points that can be added at any time.
        These are most useful when subclassing an Object. During __init__ you
        add hooks for where attachments are to be made to the object. Then every
        instance of the subclass will have these named hooks.
        """
        self.hooks[name] = matrix

    def faces(self):
        """
        Computes face attributes on demand and caches the results
        """

        if self.face_cache is None:
            self.face_cache = Faces(self)
        return self.face_cache

    def attach(self, parent, where, how=ATTACH_LARGE):
        """
        Called on child object to attach to a parent at the position specifiec by face

        parent  - The parent object being linked to
        where   - A reference to a 'face' of the parent. Options are:
                    A face retrieved from class Faces
                    A vector that will trigger a search for a face
                    A named attachment hook
        how     - Specifices the search algorithm to use when 'where' is a vector.
                  Options are currently:
                    Object.ATTACH_NORM  - selects a face whose normal vector is closest to given vector
                    Object.ATTACH_LARGE - selects a face whose area is 1/2 std deviation larger than
                                          the mean and whose normal points in roughly the right direction

        E.g. using the vector 'RT' will lookup the face whose normal vector is the
        "closest match" to 'RT' (i.e. the right face).

        Note: Creation of face metrics that are necessary for attachments based on faces can be
              slow when detail is high. Using named attachment hooks will result in the fastest 
              rendering objects since they don't rely on face metrics.
        """

        if isinstance(where, str):
            m = parent.hooks[where]
        else:
            parent_faces        = parent.faces()
            parent_face_index   = parent_faces.find_face(where, how);
            parent_faceMetrics  = parent_faces.faceMetrics[parent_face_index];
            m = parent_faces.get_matrix(parent_face_index)

        origin  = Matrix(parent.oscad_obj.origin, affine=True) @ m

        obj = Object(self)
        obj.oscad_obj = self.oscad_obj.align(origin.list())

        return obj

class cube(Object):
    """
    A Cube

    So far, just your standard everyday cube. Cubes are useful test objects, so this is one of my
    first object overloads.

    Has example attachment hooks.

    size    - [width, depth, height]. If passed a scaler, width, depth, and height are all the same value
    center  - Centers the cube on the current origin. Else the cube is positioned with it's bottom-front-left
              corner at the current origin
    """

    def __init__(self, size, center=False):
        super().__init__()

        self.name = "Cube"
        size = lst(size, 3)
        self.oscad_obj = scad.cube(size, center)

        """
        Some examples of a named attachment hook.
        """
        hook_defs = [
            ["front", [ 90,   0, 0], size[1] / 2 if center else 0],
            ["back",  [-90,   0, 0], size[1] / 2 if center else size[1]],
            ["right", [  0,  90, 0], size[0] / 2 if center else size[0]],
            ["left",  [  0, -90, 0], size[0] / 2 if center else 0],
            ["top",   [  0,   0, 0], size[2] / 2 if center else size[2]],
            ["bottom",[180,   0, 0], size[2] / 2 if center else 0],
        ]
        for hook in hook_defs:
            m = Affine.rot3d(np.radians(hook[1])) @ Affine.trans3d([0, 0, hook[2]])
            self.attachment_hook(hook[0], m)

class sphere(Object):
    """
    A Sphere

    So far, just your standard everyday sphere.

    Has example attachment hooks.

    r   - Radius of the sphere.
    d   - Diameter of the sphere
    """

    def __init__(self, r=None, d=None):
        super().__init__()

        self.name = "Sphere"
        if d is not None:
            r = d / 2
        self.oscad_obj = scad.sphere(r=r)

        """
        Some examples of a named attachment hook.
        """
        hook_defs = [
            ["front", [ 90,   0, 0], r],
            ["back",  [-90,   0, 0], r],
            ["right", [  0,  90, 0], r],
            ["left",  [  0, -90, 0], r],
            ["top",   [  0,   0, 0], r],
            ["bottom",[180,   0, 0], r],
        ]
        for hook in hook_defs:
            m = Affine.rot3d(np.radians(hook[1])) @ Affine.trans3d([0, 0, hook[2]])
            self.attachment_hook(hook[0], m)

class cylinder(Object):
    """
    A cylinder with additional features

    The cylinder supports 'EndTreatments' that allow inside and outside chamfering and rounding.

    r       - Radius of top and bottom of cylinder. 2-tuple if top and bottom have different radii
    l       - Length of the cylinder.
    ends    - See 'EdgeTreatment'. 2-tuple if top and bottom have different end treatments
    """

    def __init__(self, r, h, ends=None, d=None):
        super().__init__()

        if d is not None:
            d           = tup(r, 2)
            r           = [d[0] / 2, d[1] / 2]

        r               = tup(r, 2)
        self.name       = "Cylinder"
        cpath           = cyl_path(r=r, l=h, ends=ends)
        self.oscad_obj  = scad.polygon(cpath.deaffine().list()).rotate_extrude()

        r_mid = np.fmin(r[0], r[1]) + np.fabs(r[0] - r[1]) / 2
        angle = np.degrees(np.atan((r[0] - r[1]) / h))
        print("angle", angle)

        """
        Some examples of a named attachment hook.
        """
        hook_defs = [
            ["front", [  90 - angle,            0, 0], r_mid],
            ["back",  [ -90 + angle,            0, 0], r_mid],
            ["right", [            0,  90 - angle, 0], r_mid],
            ["left",  [            0, -90 + angle, 0], r_mid],
            ["top",   [            0,           0, 0], h / 2],
            ["bottom",[          180,           0, 0], h / 2],
        ]
        for hook in hook_defs:
            m = Affine.rot3d(np.radians(hook[1])) @ Affine.trans3d([0, 0, hook[2]])
            self.attachment_hook(hook[0], m)

class polyhedron(Object):
    """
    A polyhedron defined by points and faces.

    Wrapper for OpesSCAD polyhedron

    points  - List of [x, y, z] points
    faces   - List of faces. Faces are sublists of indicies into points
    """

    def __init__(self, points, faces):
        super().__init__()

        self.name       = "Polyhedron"
        self.points     = Points(points);
        self.faces      = faces
        self.oscad_obj  = scad.polyhedron(points=self.points.deaffine().list(), faces=faces)

class prisnoid(Object):
    """
    A prismoid-like object

    A prisnoid is a prism like object who's top and bottom faces are parallel. The top and bottom
    faces may be of differing sizes and the top face may be offset from the center point. Each corner
    is defined by a sphere which may be of differeing radii. So all edges are rounded and edge
    roundness tapers from one corner sphere radius to the next.

    size1   - [width, depth] size of the bottom of the prisnoid.
              May be a scaler in which case width and depth are the same
    size2   - [width, depth] size of the top of the prisnoid.
              May be a scaler in which case width and depth are the same
    round1  - Tuple specifying the radius of the spheres of the bottom of the prisnoid.
              Order of the tuple starts with back right (+X+Y) and goes counter-clockwise.
              May be a scaler in which all spheres are of the same radius.
    round2  - Tuple specifying the radius of the spheres of the top of the prisnoid.
              Order of the tuple starts with back right (+X+Y) and goes counter-clockwise.
              May be a scaler in which all spheres are of the same radius.
    h       - Height of the prisnoid
    shift   - [x, y] offset of the top of the prisnoid relative to the bottom
    """

    def __init__(self, size1, size2, round1, round2, h, shift=None):
        super().__init__()

        sz1     = tup(size1, 2);
        sz2     = tup(size2, 2);
        rnd1    = tup(0, 4) if round1 is None else tup(round1, 4)
        rnd2    = tup(0, 4) if round2 is None else tup(round2, 4)
        sh      = tup(0, 2) if shift is None else tup(shift, 2)

        bot_corners = (
            [ (sz1[0] / 2 - rnd1[0]),  (sz1[1] / 2 - rnd1[0]), -(h / 2 - rnd1[0])],
            [-(sz1[0] / 2 - rnd1[1]),  (sz1[1] / 2 - rnd1[1]), -(h / 2 - rnd1[1])],
            [-(sz1[0] / 2 - rnd1[2]), -(sz1[1] / 2 - rnd1[2]), -(h / 2 - rnd1[2])],
            [ (sz1[0] / 2 - rnd1[3]), -(sz1[1] / 2 - rnd1[3]), -(h / 2 - rnd1[3])],
        )
        top_corners = (
            [ (sz2[0] / 2 - rnd2[0]) + sh[0],  (sz2[1] / 2 - rnd2[0]) + sh[1],  (h / 2 - rnd2[0])],
            [-(sz2[0] / 2 - rnd2[1]) + sh[0],  (sz2[1] / 2 - rnd2[1]) + sh[1],  (h / 2 - rnd2[1])],
            [-(sz2[0] / 2 - rnd2[2]) + sh[0], -(sz2[1] / 2 - rnd2[2]) + sh[1],  (h / 2 - rnd2[2])],
            [ (sz2[0] / 2 - rnd2[3]) + sh[0], -(sz2[1] / 2 - rnd2[3]) + sh[1],  (h / 2 - rnd2[3])],
        )
        spheres = (tuple(scad.sphere(rnd1[ii]).translate(bot_corners[ii]) for ii in range(4)) +
                   tuple(scad.sphere(rnd2[ii]).translate(top_corners[ii]) for ii in range(4)))

        self.oscad_obj = scad.hull(*spheres)

class Faces():
    """
    Class for dealing with faces in an 'Object' mesh.
    """
    def __init__(self, object):
        """
        Creates a list of 'FaceMetrics' from the faces in an 'Object' mesh

        The mesh faces are unified so that adjacent faces that share the same normal become
        a single face. This can be slow when detail is high.

        object  - The 'Object' to process
        """

        mesh = object.mesh()

        # Stash the object origin and inverse transform matrix for use with attachments
        self.origin = Matrix(affine=True, val=object.origin)
        self.i_origin = self.origin.inv()

        # Remap object points to be relative to [0, 0, 0] with no rotation
        # I want all values *except* FaceMetrics.origin to be independent of the objects
        # current position and orientation.
        self.points         = self.i_origin @ Points(mesh[0])
        self.faceMetrics    = [FaceMetrics() for _ in range(len(mesh[1]))]

        self.get_normals(mesh[1], self.faceMetrics)
        self.get_areas(mesh[1], self.faceMetrics)

        # Unify adjacent triangles that share a common normal
        self.faces, deleted = self.unify_faces(mesh[1], self.faceMetrics)
        # Eliminate metrics for deleted faces
        self.faceMetrics = [self.faceMetrics[ii] for ii in range(len(self.faceMetrics)) if ii not in deleted]

        # Calculate the mean area of a face
        sum = 0
        for fm in self.faceMetrics:
            sum += fm.area
        self.mean_area = sum / len(self.faceMetrics)

        # Calculate the standard deviation of face areas
        sosd = 0
        for fm in self.faceMetrics:
            sosd += (fm.area - self.mean_area) ** 2
        self.deviation_area = np.sqrt(sosd / (len(self.faceMetrics) - 1))

    def get_matrix(self, index):

        face    = self.get_points(index)
        normal  = self.faceMetrics[index].normal

        # Then calculate the x and y rotation angles to rotate the face flat on the XY plane
        a = -np.asin(-normal[1])
        f = constrain(normal[0] / np.cos(a), -1, 1)
        b =  np.asin(f) if np.fabs(a) != np.pi / 2 else 0
        if normal[2] < 0:
            b = np.pi - b
        b = -b
        toXY = Affine.xrot3d(a) @ Affine.yrot3d(b)
        normalized_face = toXY @ face

        # The face is 'mostly' normalized, but it still has a XYZ offsets.
        # Merge the offsets into the transformation matrix
        z_off    = float(normalized_face[0][2])
        bound_lo = copy.deepcopy(normalized_face[0])
        bound_hi = copy.deepcopy(normalized_face[0])
        for point in normalized_face:
            if point[0] < bound_lo[0]: bound_lo[0] = float(point[0])
            if point[1] < bound_lo[1]: bound_lo[1] = float(point[1])
            if point[0] > bound_hi[0]: bound_hi[0] = float(point[0])
            if point[1] > bound_hi[1]: bound_hi[1] = float(point[1])

        size        = [bound_hi[0] - bound_lo[0], bound_hi[1] - bound_lo[1]]

        toXY[0][3]  = -(bound_lo[0] + size[0] / 2)
        toXY[1][3]  = -(bound_lo[1] + size[1] / 2)
        toXY[2][3]  = -z_off

        # matrix is a transformation matrix that is used to map attached objects
        # onto faces. First the 'origin' of the object the face belongs to must be applied
        # to the attaching object, then matrix is applied.
        matrix      = toXY.inv()

        return matrix

    def get_normals(self, faces, faceMetrics):
        for ii in range(len(faces)):
            points = self.get_points(faces[ii])
            faceMetrics[ii].normal = vector_axis(unit(points[2] - points[1]), unit(points[0] - points[1]))

    def get_areas(self, faces, faceMetrics):
        for ii in range(len(faces)):
            points = self.get_points(faces[ii])
            s1 = np.sqrt((points[0][0] - points[1][0]) ** 2 +
                         (points[0][1] - points[1][1]) ** 2 +
                         (points[0][2] - points[1][2]) ** 2)
            s2 = np.sqrt((points[1][0] - points[2][0]) ** 2 +
                         (points[1][1] - points[2][1]) ** 2 +
                         (points[1][2] - points[2][2]) ** 2)
            s3 = np.sqrt((points[0][0] - points[2][0]) ** 2 +
                         (points[0][1] - points[2][1]) ** 2 +
                         (points[0][2] - points[2][2]) ** 2)
            semi = (s1 + s2 + s3) / 2
            faceMetrics[ii].area = np.sqrt(semi * (semi - s1) * (semi - s2) * (semi - s3))

    def unify_faces(self, faces, faceMetrics):
        """
        Find adjacent faces that have the same normal vector and merge them
        """

        edges = []
        for ii in range(len(faces)):
            face = faces[ii]
            face_edges = []
            prev = final = face[0]
            for pt in face[1:]:
                edges.append([[int(np.fmin(prev, pt)), int(np.fmax(prev, pt))], ii])
                prev = pt
            edges.append([[int(np.fmin(prev, final)), int(np.fmax(prev, final))], ii])
        # Sort the edges so we can do a *much* faster binary search
        edges.sort(key=functools.cmp_to_key(Faces.edge_cmp))

        ii = 0
        curface = 0
        deleted = set()
        while curface < len(faces):
            if curface in deleted:
                curface += 1
                continue
            face        = faces[curface]
            neighbors   = self.neighbors(faces, curface, edges, deleted, faceMetrics)
            if len(neighbors) > 0:
                # As long as we find new neighbors to merged faces, we reprocess the same face
                new_face = self.merge_face(faces, curface, neighbors, faceMetrics)
                faces[curface] = new_face
                deleted.update([n.face for n in neighbors])
            else:
                curface += 1
            ii += 1
            assert ii < 1000000, f"Loop seems to be infinite!"

        # Remove the deleted faces
        faces   = [faces[ii]   for ii in range(len(faces)) if ii not in deleted]

        return faces, deleted

    def edge_cmp(edge1, edge2):
        """
        Compare 2 edges

        Helper function for search_edges()
        """
        c_0_0 = edge1[0][0] - edge2[0][0]
        if c_0_0 == 0:
            # edges have a common vertex
            return edge1[0][1] - edge2[0][1]
        return c_0_0

    def binary_search_first(edge, edges):
        """
        Find the first occurance of an edge in the edges list

        Helper function for search_edges()
        """

        e   = [edge, 0]
        low = 0
        hi  = len(edges) - 1
        mid = 0

        while low <= hi:
            mid = (hi + low) // 2

            if  Faces.edge_cmp(edges[mid], e) < 0:
                low = mid + 1
            elif Faces.edge_cmp(edges[mid], e) > 0:
                hi  = mid - 1
            else:
                # Equal, now find first that is equal
                while mid > 0 and Faces.edge_cmp(edges[mid - 1], e) == 0:
                    mid -= 1
                return mid

        return -1

    def search_edges(self, edge, edges, curface, deleted, faceMetrics):
        """
        Create a list of faces that are adjacent to a given edge and have
        the same normal vector as the current face

        Helper function to unify_faces
        """


        matches = []
        start   = Faces.binary_search_first(edge, edges)
        for ii in range(start, len(edges)):
            candidate = edges[ii]
            if edge != candidate[0]: break
            if (candidate[1] != curface and candidate[1] not in deleted and 
                faceMetrics[candidate[1]].normal @ faceMetrics[curface].normal > (1 - eps)):
                matches.append(candidate[1])

        return matches

    class Neighbor():
        def __init__(self, face, ind):
            self.face   = face
            self.ind    = ind

        def __hash__(self):
            return self.face

        def __eq__(self, other):
            return self.face == other.face

    def neighbors(self, faces, curface, edges, deleted, faceMetrics):
        """
        Create a list of faces that are adjacent to the current face and have
        the same normal vector as the current face

        Helper function to unify_faces
        """

        face        = faces[curface]
        neighbors   = set()
        prev        = final = face[0]
        for ii in range(1, len(face)):
            pt = face[ii]
            ind = ii - 1
            edge = [np.fmin(prev, pt), np.fmax(prev, pt)]
            matches = self.search_edges(edge, edges, curface, deleted, faceMetrics)
            if len(matches) > 0:
                # One match on an edge is expected
                match = matches[0]  # match is a neighbor face, ind an edge of the current face
                n = Faces.Neighbor(face = match, ind = ind)
                if n not in neighbors:
                    neighbors.add(n)
            prev = pt
        ind = len(face) - 1
        edge = [np.fmin(prev, final), np.fmax(prev, final)]
        matches = self.search_edges(edge, edges, curface, deleted, faceMetrics)
        if len(matches) > 0:
            # One match on an edge is expected
            match = matches[0]  # match is a face, ind an edge of the current face
            n = Faces.Neighbor(face = match, ind = ind)
            if n not in neighbors:
                neighbors.add(n)

        return neighbors

    def merge_face(self, faces, curface, neighbors, faceMetrics):
        """
        For each edge in the current face, merge any neighbor to that edge

        Helper function to unify_faces
        """

        face = faces[curface]
        new_face = []
        # for each edge in face, merge any neighbor to that edge
        for ii in range(len(face)):
            # get any neighbor to edge face[ii], may not exist
            neighbor = list(filter(lambda n: n.ind == ii, neighbors))
            assert len(neighbor) <= 1, f"Unexpected number of neighbors to an edge {len(neighbor)}"
            if len(neighbor) == 0:
                new_face.append(face[ii])
            else:
                neighbor = neighbor[0]
                neighbor_face = faces[neighbor.face]
                neighbor_area = faceMetrics[neighbor.face].area
                faceMetrics[curface].area += neighbor_area
                ind = neighbor_face.index(face[ii])  # an exception will be raised if not found!
                assert ind >= 0, f"Expected to find point {face[ii]} in neighbor face {neighbor[0]}"
                stop = face[ii+1] if ii + 1 < len(face) else face[0]
                stop_ind = neighbor_face.index(stop) # an exception will be raised if not found!
                while ind != stop_ind:
                    new_face.append(neighbor_face[ind])
                    ind = ind + 1 if ind + 1 < len(neighbor_face) else 0


        return new_face

    def get_points(self, desc):
        """
        Retreive a list of 'Points' that defines a face

        desc    - Reference to a face. Can be an index into the mesh's list of faces, or a list of
                  indexes into the mesh's list of points.
        """

        if isinstance(desc, int):
            return Points([self.points[pt] for pt in self.faces[desc]], affine=self.points.is_affine)
        elif isinstance(desc, list):
            return Points([self.points[pt] for pt in desc], affine=self.points.is_affine)
        else:
            assert False, f"Invalid reference type for face points {type(desc)}"

    def nearest(self, vec):
        """
        Retrieve the 'Face' who's 'normal' is the closest match to a given vector.

        This can be used to find the appropriate face when you wish to attach to some side of an
        object and don't know what particular face that side corresponds to.

        Note: Irregulary shaped objects with rounded corners can lead to surprising selection results.

              Use: Object.attach(parent, where=RT, how=Object.ATTACH_LARGE)
              or use named attachment hooks if this selection is not appropriate for your use case.

        vec - The vector to compare face normals with
        """

        best_angle  = tau
        selection   = 0
        for ii in range(len(self.faceMetrics)):
            angle  = vector_angle(self.faceMetrics[ii].normal, vec)
            if angle < best_angle:
                selection   = ii
                best_angle  = angle

        return selection

    def large(self, vec):
        """
        Heuristic for selection of the "best" face who's normal points roughly in the direction
        of a given vector.

        This can be used to find the appropriate face when you wish to attach to some side of an
        irregularly shaped object and don't know what particular face that side corresponds to.

        Nots: This heuristic does not work well when attempting to select the top or bottom
              of spherical objects.  The faces of sperical objects become smaller as you
              approach the top and bottom. 

              Use: Object.attach(parent, where=TP, how=Object.ATTACH_NORM)
              or use named attachment hooks if this selection is not appropriate for your use case.

        vec - The vector to compare face normals with
        """

        best_angle    = tau
        selection     = 0
        # Choose a face that is larger than the average face by a couple
        # standard deviations.  This assures that we will generally select
        # a large flat face over a small rounded corner even though the 
        # corner may match the given vector better.
        threshold     = self.mean_area + self.deviation_area / 2
        for ii in range(len(self.faceMetrics)):
            angle  = vector_angle(self.faceMetrics[ii].normal, vec)
            area = self.faceMetrics[ii].area
            if area >= threshold and angle <= best_angle:
                selection   = ii
                best_angle  = angle

        return selection

    def __len__(self):
        """
        The number of faces
        """

        return len(self.faces)

    def find_face(self, key, how):
        if isinstance(key, Vector):
            if how == Object.ATTACH_NORM:
                return self.nearest(key)
            else:
                return self.large(key)
        if isinstance(key, int):
            return key
        if isinstance(key, Face):
            return key.index

# Note to self.  The prisnoid mesh looks to have a lot of concentric triangles.
# I don't know why hull would do this, but look into simplifying... somehow...

#p = prisnoid(250, 140, 20, 33, 170, shift=[-55, -55])
#p = cube(80, center=True)
#p = sphere(d=80)
p = cylinder(h=100, r=[20, 10], ends=EdgeTreatment(round=5))
c = cube(10, center=True).color("blue")
c2 = c.attach(p, where="back")
#c2 = c.attach(p, where=RT, how=Object.ATTACH_LARGE)
#c2 = c.attach(p, where=RT, how=Object.ATTACH_NORM)
u1 = p | c2
u1.show()
#c2.show()
#p.faces()

#c = cube(10, center=True)
#c1 = cube(20, center=True).right(30)
#c2 = c.attach(c1, "bottom")
#u1 = c1 | c2
#u1.show()
#print(c.origin)
#print(c.oscad_obj.origin)


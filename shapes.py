import openscad as scad
from transforms import *
from dataclasses import dataclass
import copy
import functools
from sweep import *

fn = None
fa = 3
fs = 3

RAD_90  = np.pi / 2
RAD_45  = RAD_90 / 2
RAD_180 = np.pi
RAD_270 = RAD_180 + RAD_90

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
        return [val[ii] if ii < l else fill for ii in range(dim)]
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
        return corner_arc(r, Points(corner))
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

    e0 = EdgeTreatment()
    e1 = EdgeTreatment()
    if ends is not None:
        if ends[0] is not None:
            e0 = ends[0]
        if ends[1] is not None:
            e1 = ends[1]

    vang = np.atan2((r[0] - r[1]) / 2, l)
    if e0.chamf is not None:
        chang0 = radians(e.chang) if e0.chang is not None else RAD_45 + np.sign(e0.chamf) * vang
        chamf0 = (e0.chamf,
                  np.fabs(law_of_sines(a=e0.chamf, A=RAD_180 - chang0 - (RAD_90 - np.sign(e0.chamf) * vang), B=chang0)))
    if e0.round is not None:
        roundx0 = e0.round / np.tan(RAD_45 - vang) if e0.round >= 0 else e0.round / np.tan(RAD_45 + vang)
    if e1.chamf is not None:
        chang1 = radians(e1.chang) if e1.chang is not None else RAD_45 - np.sign(e1.chamf) * vang
        chamf1 = (e1.chamf,
                  np.fabs(law_of_sines(a=e1.chamf, A=RAD_180 - chang1 - (RAD_90 + np.sign(e1.chamf) * vang), B=chang1)))
    if e1.round is not None:
        roundx1 = e1.round / np.tan(RAD_45 + vang) if e1.round >= 0 else e1.round / np.tan(RAD_45 - vang)

    pre  = [[0, -l / 2]]
    post = [[0,  l / 2]]
    if e0.round is not None:
        bot = arc(r=np.fabs(e0.round),
                  corner=[
                            [r[0] - 2 * roundx0,    -l / 2],
                            [r[0],                  -l / 2],
                            [r[1],                   l / 2]
                         ]
        )
    elif e0.chamf is not None:
        d = polar_to_xy(chamf0[1], RAD_90 + vang)
        bot = Points(
            val = [
                [r[0] - chamf0[0],  -l / 2],
                [r[0] + d[0],       -l / 2 + d[1]]
              ]
        ).affine()
    else:
        bot = Points(val=[[r[0], -l / 2]]).affine()

    if e1.round is not None:
        top = arc(r=np.fabs(e1.round),
                  corner=[
                            [r[0],                  -l / 2],
                            [r[1],                   l / 2],
                            [r[1] - 2 * roundx1,     l / 2],
                         ]
        )
    elif e1.chamf is not None:
        d = polar_to_xy(chamf1[1], RAD_270 + vang)
        top = Points(
            val = [
                [r[1] + d[0],       l / 2 + d[1]],
                [r[1] - chamf1[0],  l / 2],
              ]
        ).affine()
    else:
        top = Points(val=[[r[1], l / 2]]).affine()

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

def line(pt1, pt2, d=None):
    if d is None: d = fs / 2
    circle = scad.circle(d=d)
    return extrude_from_to(circle, pt1, pt2)

def mesh_edges(mesh):
    points  = mesh.points
    faces   = mesh.faces
    edges   = []
    for face in faces:
        face_edges = []
        final = prev = face[0]
        for pt in face[1:]:
            edge = [points[prev], points[pt]]
            face_edges.append(edge)
            prev = pt
        edge = [points[prev], points[final]]
        face_edges.append(edge)
        edges.append(face_edges)
    return edges

def parseArgs(defaults, *args, **kwargs):
    """
    Preprocess argument list

    defaults    - A list of arguments we care about
    args        - Positional arguments
    kwargs      - Key value pair arguments

    defaults contains a dict of args that we will be processing
    ourself.  These are stripped from kwargs upon return.  The remaining
    kwargs are passed through to OpenSCAD. This ensures arguments
    such as fn, fs, fa that are universal to OpenSCAD get passed
    along even though I mostly don't need to know about them.
    """

    d = dict()
    for ii in range(len(args)):
        d[list(defaults)[ii]] = args[ii]

    for k, v in kwargs.items():
        assert k not in d, f"positional argument repeated as kwarg {k}"
        if k in defaults:
            d[k] = v

    for k in defaults:
        if k not in d:
            d[k] = defaults[k]

    # strip away kwargs that we will process
    for k in d:
        kwargs.pop(k, None)

    # prepare the list of values we will be handling
    result = []
    for k in defaults:
        result.append(d[k])

    # result contains args we will process.  kwargs contains args we are just passing along
    return result, kwargs

@dataclass()
class FaceMetrics():
    """
    Metrics collected from the faces in an 'Object' mesh. These metrics are used to position
    objects relative to another object's face.
    """

    normal : Vector = None  # A unit vector perpendicular to the face.
    area   : float  = None  # The area of the face, used for filtering out small faces

@dataclass()
class Mesh():
    points  : Points    = None
    faces   : list      = None

class Object():
    """
    Base class for all OpenSCAD objects. This allows adding features that are "missing"
    from the openscad module. I put "missing" in quotes because TBH, for most of these
    features it is probably better to implement them in Python code rather than native.
    So far, I have found that the foundation supplied by the openscad module is enough.

    It would be nice however if the openscad module supplied a base class that I could
    subclass from.  I have hacked around this with some '__magicfunc__' magic.
    """
    ATTACH_LARGE = 0
    ATTACH_NORM  = 1

    once = False    # Gets set after one-time initializations are done

    def __init__(self):
        global fn, fa, fs

        if not Object.once:
            fn, fa, fs  = get_fnas()
            Object.once = True

        self.name           = "Base"
        self.oscad_obj      = None
        self.face_cache     = None
        self.hooks          = dict()
        self.tags           = set()

    def clone(self, object):
        for k, v in object.__dict__.items():
            self.__dict__[k] = v

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

    def get_first_obj(*args):
        if isinstance(args[0], list):
            return args[0][0]
        else:
            return args[0]

    def get_oscad_obj_list(*args):
        oscad_objs = []
        for arg in args:
            if isinstance(arg, list):
                oscad_objs.extend([obj.oscad_obj for obj in arg])
            else:
                oscad_objs.append(arg.oscad_obj)
        return oscad_objs

    def difference(self, *args):
        """
        Difference a list of objects from self
        """
        oscad_objs = Object.get_oscad_obj_list(*args)
        C = type(self)
        res = C.copy(self)
        res.oscad_obj = res.oscad_obj.difference(oscad_objs)
        return res

    def union(self, *args):
        """
        Union a list of objects from self
        """
        oscad_objs = Object.get_oscad_obj_list(*args)
        C = type(self)
        res = C.copy(self)
        res.oscad_obj = res.oscad_obj.union(oscad_objs)
        return res

    def intersection(self, *args):
        """
        Union a list of objects from self
        """
        oscad_objs = Object.get_oscad_obj_list(*args)
        C = type(self)
        res = C.copy(self)
        res.oscad_obj = res.oscad_obj.intersection(oscad_objs)
        return res

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
        C = type(self)
        res = C.copy(self)
        res.oscad_obj -= other.oscad_obj
        res.name = f"{self.name} - ({other.name})"
        return res
    def __add__(self, other):
        C = type(self)
        res = C.copy(self)
        res.oscad_obj += other.oscad_obj
        res.name = f"{self.name} + ({other.name})"
        return res
    def __or__(self, other):
        C = type(self)
        res = C.copy(self)
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
            c_1_1 = Object.point_cmp(edge1[1], edge2[1])
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
        deleted = set()
        prev = edges[0]
        dd = 0
        for edge in edges[1:]:
            if edge == prev:
                deleted.add(dd)
            prev = edge
            dd += 1
        return [edges[ii] for ii in range(len(edges)) if ii not in deleted]

    def wireframe(self):
        """
        Create a wireframe visualization of the Object

        OpenSCAD has a tendency to crash when the wireframe detail gets too high.
        I have improved it's robustness and speed some by eliminating duplicate
        edges from the mesh. But I still experience crashes if fn, fs, fa are too
        fine with an Object that has lots of detail.
        """

        wf = wireframe(self)
        return wf

    def mesh(self):
        mesh = self.oscad_obj.mesh()
        # 2D objects only have points, no faces
        if len(mesh) > 1:
            return Mesh(points=Points(mesh[0]), faces=mesh[1])
        elif len(mesh) == 1:
            return Mesh(points=Points(mesh[0]), faces=None)
        else:
            return Mesh(points=Points([]), faces=None)

    def translate(self, v):
        C = type(self)
        res = C.copy(self)

        # v may be a Vector, make it compatible with OpenSCAD
        if res.oscad_obj is not None:
            res.oscad_obj = res.oscad_obj.translate(list(v))
        elif hasattr(res, "origin"):
            m = Affine.trans3d(v)
            res.origin = m @ res.origin

        return res

    def rotate(self, v):
        C = type(self)
        res = C.copy(self)

        # v may be a Vector, make it compatible with OpenSCAD
        if res.oscad_obj is not None:
            res.oscad_obj = res.oscad_obj.rotate(list(v))
        elif hasattr(res, "origin"):
            m = Affine.rot3d([ np.radians(v[0]), np.radians(v[1]), np.radians(v[2]) ])
            res.origin = m @ res.origin

        return res

    def xrot(self, v):
        return self.rotate([v, 0, 0])

    def yrot(self, v):
        return self.rotate([0, v, 0])

    def zrot(self, v):
        return self.rotate([0, 0, v])

    def color(self, c):
        C = type(self)
        res = C.copy(self)
        res.oscad_obj = res.oscad_obj.color(c)
        return res

    def scale(self, s):
        C = type(self)
        res = C.copy(self)
        res.oscad_obj = res.oscad_obj.scale(s)
        return res

    def offset(self, r=None, delta=None, chamfer=False):
        C = type(self)
        res = C.copy(self)
        if r is not None:
            res.oscad_obj = res.oscad_obj.offset(r=r)
        else:
            res.oscad_obj = res.oscad_obj.offset(delta=delta, chamfer=chamfer)
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

    def justify(self, where, how=ATTACH_LARGE):
        """
        Called on child object to align an attachment point to it's current origin

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
            m = self.hooks[where]
        else:
            faces        = self.faces()
            face_index   = faces.find_face(where, how);
            m = faces.get_matrix(face_index)

        m = m.inv() * -1
        m[3][3] = 1

        C = type(self)
        obj = C.copy(self)
        origin  = Matrix(self.origin, affine=True)

        m = origin @ m @ origin.inv()
        if obj.oscad_obj is not None:
            obj.oscad_obj = obj.oscad_obj.multmatrix(m.list())
        elif hasattr(obj, "origin"):
            obj.origin = m @ obj.origin

        return obj

    def attach(self, parent, where, justify=None, how=ATTACH_LARGE, inside=False):
        """
        Called on child object to attach to a parent at the position specified by face

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
            m = parent_faces.get_matrix(parent_face_index)

        if inside:
            m = m @ Affine.xrot3d(np.pi)
        origin  = Matrix(parent.origin, affine=True) @ m

        C = type(self)
        obj = C.copy(self)
        if obj.oscad_obj is not None:
            obj.oscad_obj = self.oscad_obj.multmatrix(origin.list())
        elif hasattr(obj, "origin"):
            obj.origin = origin @ obj.origin

        if justify is not None:
            return obj.justify(where=justify, how=how)

        return obj

    def tag(self, tag):
        self.tags.add(tag)

    def has_tag(self, tag):
        return tag in self.tags

    def __copy__(self):
        C = type(self)
        obj = C.copy(self)
        return obj

    @classmethod
    def copy(cls, object):
        c = cls.__new__(cls)
        c.clone(object)
        return c

class wireframe(Object):
    def __init__(self, object=None, points=None, faces=None, unify=False):
        super().__init__();

        if object is not None:
            self.mesh = object.mesh()
        elif points is not None and faces is not None:
            self.mesh.points = Points(points)
            self.mesh.faces  = faces
        else:
            assert False, f"Either 'object' or 'points' and 'faces' must be defined"

        if unify:
            faces       = Faces(points=self.mesh.points, faces=self.mesh.faces)
            self.mesh.points = faces.points
            self.mesh.faces  = faces.faces

        edges = self.dedup_mesh(self.mesh)

        self.name = "Wireframe"

        edge = edges[0]
        self.oscad_obj = line(edge[0], edge[1])
        for edge in edges[1:]:
            self.oscad_obj |= line(edge[0], edge[1])

class Null(Object):
    """
    A naked singularity.

    Null has an origin, so thus has a position and orientation. But it has no
    size and no faces. Null may be attached to other objects, but other objects
    may attach to Null only through a named attachment hook. Null has some
    default named attachment hooks that permit attaching and orienting objects
    at it's origin.
    """
    def __init__(self, object=None):
        super().__init__();

        self.origin = Matrix([[1,0,0],[0,1,0],[0,0,1]]).affine()
        self.name   = "Null"

        hook_defs = [
            ["front", [ 90,   0, 0], 0],
            ["back",  [-90,   0, 0], 0],
            ["right", [  0,  90, 0], 0],
            ["left",  [  0, -90, 0], 0],
            ["top",   [  0,   0, 0], 0],
            ["bottom",[180,   0, 0], 0],
        ]
        for hook in hook_defs:
            m = Affine.rot3d(np.radians(hook[1])) @ Affine.trans3d([0, 0, hook[2]])
            self.attachment_hook(hook[0], m)

    def __getattr__(self, attr):
        """
        Prevent passing of attributs to OpenSCAD for the null object
        """
        #assert False, f"Object '{self.name}' has no attribute '{attr}'"

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value

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

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__()

        defaults = {"size" : 10, "center" : True}
        [size, center], kwargs = parseArgs(defaults, *args, **kwargs)

        self.name = "Cube"
        size = lst(size, 3)
        self.oscad_obj = scad.cube(size, center, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__()

        defaults = {"r" : None, "d" : None}
        [r, d], kwargs = parseArgs(defaults, *args, **kwargs)

        self.name = "Sphere"
        if d is not None:
            r = d / 2
        self.oscad_obj = scad.sphere(r=r, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__()

        defaults = {"r" : None, "h" : None, "l" : None, "ends" : None, "d" : None}
        [r, h, l, ends, d], kwargs = parseArgs(defaults, *args, **kwargs)

        if l is not None:
            h = l
        if d is not None:
            d           = tup(d, 2)
            r           = [d[0] / 2, d[1] / 2]

        assert r is not None and h is not None, f"cylinder: Invalid parameters"

        r               = tup(r, 2)
        self.name       = "Cylinder"
        if ends is None:
            # Simple cylinder, use OpenSCAD
            self.oscad_obj  = scad.cylinder(r1=r[0], r2=r[1], h=h, center=True)
        else:
            # Enhanced cylinder
            cpath           = cyl_path(r=r, l=h, ends=ends)
            self.oscad_obj  = scad.polygon(cpath.deaffine().list()).rotate_extrude(convexity=2)

        r_mid = np.fmin(r[0], r[1]) + np.fabs(r[0] - r[1]) / 2
        angle = np.degrees(np.atan((r[0] - r[1]) / h))

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
        rnd1    = lst(0, 4) if round1 is None else lst(round1, 4)
        rnd2    = lst(0, 4) if round2 is None else lst(round2, 4)
        sh      = tup(0, 2) if shift is None else tup(shift, 2)

        # Merge rounding lists, bottom first
        radius = rnd1 + rnd2
        # Since zero size spheres are not permitted, make minimum round fs
        for ii in range(8):
            radius[ii] = radius[ii] if radius[ii] > 0 else fs
        self.radius = radius

        # Where the corners would be if they were not rounded.
        # These are listed in the same bottom-to-top, counter-clockwise
        # order as 'radius'
        corners = [
            [ sz1[0] / 2,  sz1[1] / 2, -h / 2],  # bottom back right
            [-sz1[0] / 2,  sz1[1] / 2, -h / 2],  # bottom back left
            [-sz1[0] / 2, -sz1[1] / 2, -h / 2],  # bottom front left
            [ sz1[0] / 2, -sz1[1] / 2, -h / 2],  # bottom front right

            [ sz2[0] / 2 + sh[0],  sz2[1] / 2 + sh[1],  h / 2],  # top back right
            [-sz2[0] / 2 + sh[0],  sz2[1] / 2 + sh[1],  h / 2],  # top back left
            [-sz2[0] / 2 + sh[0], -sz2[1] / 2 + sh[1],  h / 2],  # top front left
            [ sz2[0] / 2 + sh[0], -sz2[1] / 2 + sh[1],  h / 2],  # top front right
        ]
        self.corners = corners

        # I want the faces to have the same angle across the entire face, which means
        # doing a little arithmatic. It also means that bottom/top corners will not be
        # square if rounding for each corner differs. You can have one or the other,
        # but not both.
        #
        # Find the center points for the rounded corners.
        centers = self.find_center_points(corners, radius)
        self.centers = centers

        spheres = tuple(scad.sphere(radius[ii]).translate(centers[ii]) for ii in range(8))
        self.oscad_obj = scad.hull(*spheres)

        # More arithmatic to find some attachment hooks
        front_offset    = (sz1[1] - sz2[1]) / 2 + sh[1]
        front_angle     = np.atan(front_offset / h)
        hyp             = sz1[1] / 2 - front_offset / 2
        front_mid       = hyp * np.cos(front_angle)
        front_angle     = np.degrees(front_angle)

        back_offset     = (sz1[1] - sz2[1]) / 2 - sh[1]
        back_angle      = np.atan(back_offset / h)
        hyp             = sz1[1] / 2 - back_offset / 2
        back_mid        = hyp * np.cos(back_angle)
        back_angle      = np.degrees(back_angle)

        right_offset    = (sz1[0] - sz2[0]) / 2 - sh[0]
        right_angle     = np.atan(right_offset / h)
        hyp             = sz1[0] / 2 - right_offset / 2
        right_mid       = hyp * np.cos(right_angle)
        right_angle     = np.degrees(right_angle)

        left_offset     = (sz1[0] - sz2[0]) / 2 + sh[0]
        left_angle      = np.atan(left_offset / h)
        hyp             = sz1[0] / 2 - left_offset / 2
        left_mid        = hyp * np.cos(left_angle)
        left_angle      = np.degrees(left_angle)

        """
        Named attachment hooks.
        """
        hook_defs = [
            ["front", [  90 - front_angle,                 0, 0], front_mid],
            ["back",  [ -90 +  back_angle,                 0, 0],  back_mid],
            ["right", [                 0,  90 - right_angle, 0], right_mid],
            ["left",  [                 0, -90 +  left_angle, 0],  left_mid],
            ["top",   [                 0,                 0, 0],     h / 2],
            ["bottom",[               180,                 0, 0],     h / 2],
        ]
        for hook in hook_defs:
            m = Affine.rot3d(np.radians(hook[1])) @ Affine.trans3d([0, 0, hook[2]])
            self.attachment_hook(hook[0], m)

    def find_center_points(self, corners, radius):
        # For each corner, we need to find the center point of an arc
        # that is tangent to 2 vectors on each of 2 faces:
        #
        # corner[0] - bottom back right
        # face 1 (YZ): bottom front right (3), bottom back right (0), top back right (4)
        # face 2 (XZ): bottom back left (1), bottom back right (0), top back right (4)
        #
        # corner[1] - bottom back left
        # face 1 (XZ): bottom back right (0), bottom back left (1), top back left (5)
        # face 2 (YZ): bottom front left (2), bottom back left (1), top back left (5)
        #
        # ...
        # corner[4] - top back right
        # face 1 (YZ): top front right (7), top back right (4), bottom back right (0)
        # face 2 (XZ): top back left (5), top back right (4), bottom back right (0)
        #
        # A pattern emerges ...

        result = copy.deepcopy(corners)
        corners = Points(corners)

        for ii in range(8):
            # X vs Y axis swaps for each ii
            offset = 1 if ii % 2 == 0 else -1

            # p2     - point at current corner
            # p1, p2 - edge on top or bottom
            # p2, p3 - edge from bottom to top
            # [p1, p2], [p2, p3] are the 2 edges I am fitting a sphere to
            p1 = (ii - offset) % 4 + 4 * int(ii / 4) # ii < 4 - bottom, ii >= 4 - top
            p2 = ii
            p3 = (p2 + 4) % 8

            # YZ face
            c = Points([ [corners[p1].y, corners[p1].z], [corners[p2].y, corners[p2].z], [corners[p3].y, corners[p3].z] ])
            (cp1, n, tp1, tp2) = circle_2tan(radius[ii], c)

            p1 = (ii + offset) % 4 + 4 * int(ii / 4) # ii < 4 - bottom, ii >= 4 - top
            # XZ face
            c = Points([ [corners[p1].x, corners[p1].z], [corners[p2].x, corners[p2].z], [corners[p3].x, corners[p3].z] ])
            (cp2, n, tp1, tp2) = circle_2tan(radius[ii], c)

            result[p2][0] = float(cp2[0])   # X from XZ center point
            result[p2][1] = float(cp1[0])   # Y from YZ center point
            result[p2][2] = float(cp1[1])   # Z should be the same for both XZ and YZ

        return result


class circle(Object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        defaults = {"r" : 10, "d" : None}
        [r, d], kwargs = parseArgs(defaults, *args, **kwargs)

        if d is not None:
            r = d / 2;

        self.name       = "Circle"
        self.oscad_obj  = scad.circle(r=r, **kwargs)

class square(Object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        defaults = {"size" : 10, "center" : False}
        [size, center], kwargs = parseArgs(defaults, *args, **kwargs)

        size = lst(size, 2)

        self.name       = "Square"
        self.oscad_obj  = scad.square(size, center=center, **kwargs)

class polygon(Object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.name       = "Polygon"
        self.oscad_obj  = scad.polygon(*args, **kwargs)

class text(Object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        defaults = {"text" : "", "size" : 10}
        [text, size], kwargs = parseArgs(defaults, *args, **kwargs)

        self.name       = "Text"
        self.oscad_obj = scad.text(text, size, **kwargs)

class Faces():
    """
    Class for dealing with faces in an 'Object' mesh.
    """
    def __init__(self, object=None, points=None, faces=None):
        """
        Creates a list of 'FaceMetrics' from the faces in an 'Object' mesh

        The mesh faces are unified so that adjacent faces that share the same normal become
        a single face. This can be slow when detail is high.

        object  - The 'Object' to process
        """

        if object is not None:
            i_origin = Matrix(val=object.origin, affine=True).inv()
            mesh = object.mesh()

            # Remap object points to be relative to [0, 0, 0] with no rotation
            # I want all values *except* FaceMetrics.origin to be independent of the objects
            # current position and orientation.
            self.points = i_origin @ Points(mesh.points)
            self.faces  = mesh.faces
        elif points is not None and faces is not None:
            self.points = Points(points)
            self.faces  = faces
        else:
            assert False, f"Either 'object' or 'points' and 'faces' must be defined"

        self.faces          = self.prune_degenerate(self.faces)
        self.faceMetrics    = [FaceMetrics() for _ in range(len(self.faces))]

        self.get_normals(self.faces, self.faceMetrics)
        self.get_areas(self.faces, self.faceMetrics)

        # Unify adjacent triangles that share a common normal
        self.faces, deleted = self.unify_faces(self.faces, self.faceMetrics)
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

    def prune_degenerate(self, faces):
        """
        Remove degenerate faces from the faces list.

        I discovered the cylinder object has a lot of degenerate faces.

        A degenerate face has a zero length side and it's normal is invalid.
        Invalid normals interfere with unification of faces. So we need
        to prune these degenerate faces before unify_faces()
        """

        degenerate = set()
        for ii in range(len(faces)):
            points = self.get_points(faces[ii])
            edge1 = points[1] - points[0]
            edge2 = points[2] - points[0]
            if edge1.cross(edge2).norm() < eps:
                degenerate.add(ii)
        return [faces[ii]   for ii in range(len(faces)) if ii not in degenerate]

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
            if point.x < bound_lo.x: bound_lo.x = float(point.x)
            if point.y < bound_lo.y: bound_lo.y = float(point.y)
            if point.x > bound_hi.x: bound_hi.x = float(point.x)
            if point.y > bound_hi.y: bound_hi.y = float(point.y)

        size        = [bound_hi.x - bound_lo.x, bound_hi.y - bound_lo.y]

        toXY[0][3]  = -(bound_lo.x + size[0] / 2)
        toXY[1][3]  = -(bound_lo.y + size[1] / 2)
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
        # Choose a face that is larger than the average face by about a
        # standard deviation.  This assures that we will generally select
        # a large flat face over a small rounded corner even though the 
        # corner may match the given vector better.

        # Tweak the threshold using relative standard devaition, if RSD is < 1
        # there is low variance. I.e. all sides have similar area. We need
        # to bring the threshold down in this case.
        rsd = self.deviation_area / self.mean_area
        threshold = self.mean_area + (rsd - 1) * self.mean_area
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

@dataclass()
class CompositionOperation():
    """
    An operation.

    The left hand operand is either the 'parent' passed in to Composition.compose()
    or it is the result from a previous operation.

    The right hand operand is a list of objects that have the tag specified
    in the CompositionOperation().

    Currently supported operations are:
    PASS        -   Essentially a No-Op, presumes one of the operands is None
    UNION       -   Union left operand with right hand list
    DIFF        -   Diff the right hand list from the left operand
    INTERSECT   -   Produce the intersection of all operands
    """

    op  : str   = None
    tag : str   = None

class Composition(Null):
    """
    A composition is a kind of virtual object with delayed instantiation.
    Composition objects are composed with a parent object and other
    composition objects using 'Composition.compose()'. At the time of
    composition, their objects are created and any attachments/justifications
    are resolved.

    Composition is a subclass of Null, so inherits it's 'origin' attribute.
    A composition may be attached *to* other objects and it's position and
    orientation may be changed prior to 'Composition.compose()'. But generally
    Composition objects are not meant to be the target of an attachment. Prior
    to 'Composition.compose()' a Composition has no faces and it acts like a
    point sized object. I.e.  it has position, orientation, but no size.

    Compose is useful when you have multiple overlapping objects where you
    would like the voids of an object to create voids in the other objects.

    An example subclass implementation of Composition is in test.py
    """

    # Operations
    PASS        = 0
    UNION       = 1
    DIFF        = 2
    INTERSECT   = 3

    """
    A composition operation that provides a 3 step difference.

    Step 1 - Some objects are unioned together
    Step 2 - Some objects are removed (differenced) from the union of Step 1
    Step 3 - Some objects are unioned to the results of Steps 1 & 2
    """
    AddRemoveKeep = [
        CompositionOperation(tag="add",    op=UNION),
        CompositionOperation(tag="remove", op=DIFF),
        CompositionOperation(tag="keep",   op=UNION)
    ]

    def __init__(self):
        super().__init__()

        self.name = "Composition"

        # The objects that will be composed
        self.objects = []
        # Compositions may be added together and treated as one
        self.compositions = []

    def clone(self, object):
        """
        Shallow copy a composition.
        Used by Object.copy()
        """
        super().clone(object)
        self.compositions   = copy.copy(object.compositions)

    def build(self):
        """
        build must be implemented by subclasses of Composition

        This will be called by Composition.compose() to instantiate the
        objects that are about to be composed.

        This is where a Composition's component objects are defined, tagged, and
        added to the 'objects' list.

        It's also generally a good idea to overload Composition.compose() in the
        subclass and define your list of CompositionOperations there. Then call
        super().compose() with those operations.

        In order to position and orient the components of a Composition relative
        to the world outside the Composition, they should be attached to and positioned
        relative to 'self' which is the Null object that may be attached to
        objects outside the Composition.
        """
        C = type(self)
        assert False, "Required build() method missing for class {C}"

    def build_all(self):
        """
        Build self and all linked compositions
        """

        self.build()
        for composition in self.compositions:
            composition.build_all()

    def append(self, other):
        """
        Append another Composition to this Composition's list of
        Compositions.

        When you have a collection of Composition objects, you can apply
        the operations in Composition.compose() in the intended order to
        all component members of the Compositions by grouping the Compositions
        with this list.

        E.g. using the AddRemoveKeep Operation:
        Step 1  -   All objects tagged with 'add' from all Compositions in the list
                    are unioned together
        Step 2  -   All objects tagged with 'remove' from all Compositions in the list
                    are differenced from the result of Step 1
        Step 3  -   All objects tagged with 'keep' from all Compositions in the list
                    are unioned with the results of Steps 1 & 2

        other   -   The Composition to append

        Returns self
        """

        if isinstance(other, list):
            self.compositions.extend(other)
        else:
            self.compositions.append(other)
        return self

    def __add__(self, other):
        """
        Append another Composition to this Composition's list of
        Compositions.

        See 'append()'

        other   -   The Composition to append

        Returns a copy of self with the other Composition appended
        """

        C = type(self)
        res = C.copy(self)
        res.compositions.append(other)
        return res

    def find_tagged_objects(self, tag):
        """
        Find objects in the Composition list of 'objects' that have
        the given tag.

        Descends into Compositions linked to this one through 'compositions'
        to collect all linked objects with the given tag.

        tag     -   The tag to search for

        Returns a list of objects
        """
        if tag is None: return None

        tagged_objs = [obj for obj in self.objects if obj.has_tag(tag)]
        for composition in self.compositions:
            tagged_objs.extend(composition.find_tagged_objects(tag))
        return tagged_objs

    def do_op(self, left, right, op):
        """
        Perform an operation.

        left    -   An object
        right   -   A list of objects
        op      -   The operation to perform.
                        result = left op right
                    or in some cases
                        result = op([left].extend(right))
        """
        if op == self.PASS:
            return right if left is None else left
        if op == self.UNION:
            if left is not None:
                return left.union(right)
            return union(right)
        if op == self.DIFF:
            return left.difference(right)
        if op == self.INTERSECT:
            if left is not None:
                return left.intersection(right)
            return intersection(right)

    def compose(self, parent=None, operations=None):
        """
        Recursively build and compose compositions

        Compositions that have been linked to this composition through
        'append()' or '+' are also built and composed using the operations
        provided.

        parent      -   The composition operations will be applied against
                        the parent object and the other objects in the
                        composition.  Parent may be none if the first composition
                        operation does not require a left hand side (difference)

        operations  -   A list of CompositionOperations(). For each operation, objects
                        that have the tag associated with the operation are collected
                        and act as the right hand side of the operation.
        """

        self.build_all()

        left = parent
        for op in operations:
            right = self.find_tagged_objects(op.tag)
            left  = self.do_op(left, right, op.op)

        return left

def difference(*args):
    """
    Difference a list of objects

    For some reason, I appear to get a union out of the difference below.
    Bug... or me? The OpenSCAD code looks a little sus. It appears to
    be adding only the first item in the list as a child to the node.
    """

    obj = Object.get_first_obj(*args)
    C = type(obj)
    res = C.copy(obj)
    oscad_objs = Object.get_oscad_obj_list(*args)
    res.oscad_obj = scad.difference(oscad_objs)
    return res

def intersection(*args):
    """
    Produce the intersection a list of objects

    For some reason, I appear to get a union out of the intersection below.
    Bug... or me? The OpenSCAD code looks a little sus. It appears to
    be adding only the first item in the list as a child to the node.
    """
    obj = Object.get_first_obj(*args)
    C = type(obj)
    res = C.copy(obj)
    oscad_objs = Object.get_oscad_obj_list(*args)
    res.oscad_obj = scad.intersection(oscad_objs)
    return res

def union(*args):
    """
    Union a list of objects
    """
    obj = Object.get_first_obj(*args)
    C = type(obj)
    res = C.copy(obj)
    oscad_objs = Object.get_oscad_obj_list(*args)
    res.oscad_obj = scad.union(oscad_objs)
    return res

def hull(*args):
    """
    Union a list of objects
    """
    obj = Object.get_first_obj(*args)
    C = type(obj)
    res = C.copy(obj)
    oscad_objs = Object.get_oscad_obj_list(*args)
    res.oscad_obj = scad.hull(oscad_objs)
    return res

def show(*args):
    """
    Show a list of objects
    """
    oscad_objs = Object.get_oscad_obj_list(*args)
    scad.show(oscad_objs)


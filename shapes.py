import openscad as scad
from transforms import *
from collections import namedtuple
import time
import inspect
from dataclasses import dataclass
import copy

fn = None
fa = 10
fs = 10

prof_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
prof_last  = prof_start
profile = False

def prof_time(msg="", final=False):
    global prof_last
    global prof_start
    global verbose

    if not profile: return
    last = prof_start if final else prof_last
    now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    print(f"{msg} - ellapsed: {(now - last) / (1000*1000)}ms")
    prof_last = now

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
        bot = Points(val=[r[0], -l / 2]).affine()
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
        top = Points(val=[r[1], l / 2]).affine()
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
    origin : Matrix = None  # Transform matrix mapping coordinates onto a face.
                            # 'origin' will transform points to be relative to the object
                            # the face is a part of with the face 'normal' as the positive Z-axis.
                            # 'origin' is appropriate to use with 'align()' to position objects
                            # relative to an object's face.
                            # !!Warining: 'origin' must be recalculated if the face's object's
                            #             position or orientation is changed after computing
                            #             FaceMetrics
    area   : float  = None  # The area of the face, used for filtering out small faces
    size   : list   = None  # the size of the face, used for filtering out small faces
                            # The size is defined by a square bounding box that contains all face points

class Object():
    """
    Base class for all OpenSCAD objects. This allows adding features that are "missing"
    from the openscad module. I put "missing" in quotes because TBH, for most of these 
    features it is probably better to implement them in Python code rather than native. 
    So far, I have found that the foundation supplied by the openscad module it enough.

    It would be nice however if the openscad module supplied a base class that I could
    subclass from.  I have hacked around this with some '__getattr__' magic.
    """

    def __init__(self, object=None):
        self.name           = "Base"
        self.oscad_obj      = None
        self.face_cache     = None
        if object is not None:
            self.name           = object.name
            self.oscad_obj      = object.oscad_obj
            self.face_cache     = object.face_cache

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

    """
    TODO: Create object modifiers to allow different visualizations
          E.g. Wireframe, ghost, etc
    """
    def wireframe(self):
        mesh = self.oscad_obj.mesh()
        wf = Object()
        wf.name = f"{self.name} - Wireframe"
        for face in mesh[1]:
            final = prev = face[0]
            for pt in face[1:]:
                self.line(mesh[0][prev], mesh[0][pt])
                prev = pt
            wf.line(mesh[0][prev], mesh[0][final])
        return wf

    def translate(self, v):
        res = Object(self)

        # v may be a Vector, make it compatible with OpenSCAD
        res.oscad_obj = res.oscad_obj.translate(list(v))

        m = Affine.trans3d(v)
        # Update face_cache[].FaceMetrics.origin to the new position
        if res.face_cache is not None:
            for metrics in res.face_cache.faceMetrics:
                metrics.origin = m @ metrics.origin

        return res

    def rotate(self, v):
        res = Object(self)

        # v may be a Vector, make it compatible with OpenSCAD
        res.oscad_obj = res.oscad_obj.rotate(list(v))

        m = Affine.rot3d(v)
        # Update face_cache[].FaceMetrics.origin to the new position
        if res.face_cache is not None:
            for metrics in res.face_cache.faceMetrics:
                metrics.origin = m @ metrics.origin

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

    def faces(self):
        """
        Computes face attributes on demand and caches the results
        """

        if self.face_cache is None:
            self.face_cache = Faces(self)
        return self.face_cache

    def attach(self, parent, face):
        """
        Called on child object to attach to a parent at the position specifiec by face

        parent      - The parent object being linked to
        face        - A reference to a 'face' of the parent. This can be a face retrieved from
                      class Faces, or it can be a vector specifying the direction to use to find a face.
                      E.g. using the vector 'rt()' will lookup the face whose normal vector is the
                      closest match to 'rt()' (i.e. the right face).
        """

        parent_faces        = parent.faces()
        parent_face_index   = parent_faces.find_face(face);
        parent_faceMetrics  = parent_faces.faceMetrics[parent_face_index];

        obj = Object(self)
        obj.oscad_obj = self.oscad_obj.align(parent_faceMetrics.origin.list())

        return obj

class cube(Object):
    """
    Subclass of 'Ojbect'. 

    So far, just your standard everyday cube. Cubes are useful test objects, so this is one of my
    first object overloads.

    size    - [width, depth, height]. If passed a scaler, width, depth, and height are all the same value
    center  - Centers the cube on the current origin. Else the cube is positioned with it's bottom-front-left
              corner at the current origin
    """

    def __init__(self, size, center=False):
        super().__init__()

        self.name = "Cube"
        size = lst(size, 3)
        self.oscad_obj = scad.cube(size, center)


class cylinder(Object):
    """
    Subclass of 'Object'. 

    The cylinder supports 'EndTreatments' that allow inside and outside chamfering and rounding.

    r       - Radius of top and bottom of cylinder. 2-tuple if top and bottom have different radii
    l       - Length of the cylinder.
    ends    - See 'EdgeTreatment'. 2-tuple if top and bottom have different end treatments
    """

    def __init__(self, r, l, ends):
        super().__init__()

        self.name       = "Cylinder"
        cpath           = cyl_path(r, l, ends)
        self.oscad_obj  = scad.polygon(cpath.deaffine().list()).rotate_extrude()

class prisnoid(Object):
    """
    Subclass of 'Object'

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

def is_small_face(face, limit):
    shortest = None
    for ii in range(len(face) - 1):
        v = face[ii + 1] - face[ii]
        l = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if shortest is None or l < shortest:
            shortest = l

    return shortest < limit

class Faces():
    """
    Class for dealing with faces in an 'Object' mesh.
    """
    def __init__(self, object):
        """
        Creates a list of 'FaceMetrics' from the faces in an 'Object' mesh

        object  - The 'Object' to process
        """

        # Stash the object origin and inverse transform matrix for use with attachments
        self.origin = Matrix(affine=True, val=object.origin)
        self.i_origin = self.origin.inv()

        # Normalize the input object. I.e. reverse any transforms that have been applied
        mesh = object.mesh()
        # Remap object points to be relative to [0, 0, 0] with no rotation
        # I want all values *except* FaceMetrics.origin to be independent of the objects
        # current position and orientation.
        self.points = self.i_origin @ Points(mesh[0])
        self.faces  = mesh[1]
        self.faceMetrics = []

        ii = 0
        for desc in self.faces:
            face = self.get_points(desc)
            if is_small_face(face, 4):
                continue

            fm = FaceMetrics()

            # Add a reference to self in the face list to simplify 'Faces.find_face()'
            fm.index = ii

            # Get the normal to the plane of the face
            fm.normal = vector_axis(unit(face[2] - face[1]), unit(face[0] - face[1]))

            # Then calculate the x and y rotation angles to rotate the face flat on the XY plane
            a = -np.asin(-fm.normal[1])
            f = constrain(fm.normal[0] / np.cos(a), -1, 1)
            b =  np.asin(f) if np.fabs(a) != np.pi / 2 else 0
            if fm.normal[2] < 0:
                b = np.pi - b
            b = -b
            toXY = Affine.xrot3d(a) @ Affine.yrot3d(b)
            normalized_face = toXY @ face

            # The face is 'mostly' normalized, but it still has a XYZ offsets.  Merge
            # the Z offset into the transformation matrix
            z_off           = float(normalized_face[0][2])
            bound_lo = copy.deepcopy(normalized_face[0])
            bound_hi = copy.deepcopy(normalized_face[0])
            for point in normalized_face:
                if point[0] < bound_lo[0]: bound_lo[0] = float(point[0])
                if point[1] < bound_lo[1]: bound_lo[1] = float(point[1])
                if point[0] > bound_hi[0]: bound_hi[0] = float(point[0])
                if point[1] > bound_hi[1]: bound_hi[1] = float(point[1])
            fm.size = [bound_hi[0] - bound_lo[0], bound_hi[1] - bound_lo[1]]

            toXY[0][3]      = -(bound_lo[0] + fm.size[0] / 2)
            toXY[1][3]      = -(bound_lo[1] + fm.size[1] / 2)
            toXY[2][3]      = -z_off

            # matrix is a transformation matrix that is used to map attached objects 
            # onto faces. First the 'origin' of the object the face belongs to is applied
            # to the attaching object, then matrix is applied.
            fm.matrix       = toXY.inv()

            # The face 'origin' is a transformation matrix that will map attaching object onto
            # the face in one go.  This must be updated if the parent object's position or orientation
            # is changed after these face metrics have been calculated.
            fm.origin       = fm.matrix @ self.origin

            # Put the face on the XY plane so we can do some calculations more easily
            normalized_face = toXY @ face

            # Calculate the surface area of the face. This is used for filtering
            # "uninteresting" faces out of the list
            sum = 0
            l = len(normalized_face)
            for ii in range(l - 1):
                sum += normalized_face[ii][0] * normalized_face[ii+1][1]
                sum -= normalized_face[ii][1] * normalized_face[ii+1][0]
            sum += normalized_face[l - 1][0] * normalized_face[0][1]
            sum -= normalized_face[l - 1][1] * normalized_face[0][0]
            fm.area = np.fabs(sum / 2)

            self.faceMetrics.append(fm)
            ii += 1

    def get_points(self, desc):
        """
        Retreive a list of 'Points' that defines a face

        desc    - Reference to a face. Can be an index into the mesh's list of faces, or a list of
                  indexes into the mesh's list of points.
        """

        if isinstance(desc, int):
            return Points([self.points[pt] for pt in self.faces[idx]], affine=self.points.is_affine)
        elif isinstance(desc, list):
            return Points([self.points[pt] for pt in desc], affine=self.points.is_affine)
        else:
            assert False, f"Invalid reference type for face points {type(desc)}"

    def nearest(self, vec):
        """
        Retrieve the 'Face' who's 'normal' is the closest match to a given vector.

        This can be used to find the appropriate face when you wish to attach to the "front" of an
        object and don't know what particular face that is.

        vec - The vector to compare face normals with
        """

        ii = 0
        nearest_angle   = tau
        nearest         = 0
        for fm in self.faceMetrics:
            angle           = vector_angle(fm.normal, vec)
            if angle < nearest_angle:
                nearest         = ii
                nearest_angle   = angle
            ii += 1

        return nearest

    def __len__(self):
        """
        The number of faces
        """

        return len(self.faces)

    def find_face(self, key):
        if isinstance(key, Vector):
            return self.nearest(key)
        if isinstance(key, int):
            return key
        if isinstance(key, Face):
            return key.index

# Note to self.  The prisnoid mesh looks to have a lot of concentric triangles.
# I don't know why hull would do this, but look into simplifying... somehow...

#cyl1 = cylinder(r=20, l=110, ends=EdgeTreatment(round=-15))
#print("c1 origin", c1.origin)

c = cube(10, center=True)
#pris1 = prisnoid(250, 140, 20, 33, 170, shift=[-55, -55])
pris1 = cube(20)
c1 = c.attach(pris1, RT)
u1 = c1 | pris1
pris2 = pris1.back(200)
u2 = pris2 | c.attach(pris2, RT)
#u2 = pris1.back(200) | c1.attach(pris1, RT)
u2.show()

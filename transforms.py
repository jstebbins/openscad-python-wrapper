import numpy as np
from utils import get_fnas, prof_lap_start, prof_lap_pause

eps = 1e-6
tau = 2 * np.pi

def is_vec(var):
    """
    Check if var is one of the vector types we can deal with
    """
    return isinstance(var, Vector) or isinstance(var, tuple) or isinstance(var, list) or isinstance(var, np.ndarray)

def vec(val, dim, fill=None):
    """
    Make a scaler or vector into a vector of a given dimension
    """

    if val is None: return None
    if is_vec(val):
        l = len(val)
        fill = val[l - 1] if fill is None else fill;
        return Vector([val[ii] if ii < l else fill for ii in range(dim)])
    return Vector([val for ii in range(dim)]);

def point3d(val):
    """
    Make a scaler of vector into a vector of size 3 and zero fill
    """

    return vec(val, 3, fill=0)

def point2d(val):
    """
    Make a scaler of vector into a vector of size 2 and zero fill
    """

    return vec(val, 2, fill=0)

def is_matrix(var):
    """
    Is is a Matrix?
    """

    return isinstance(var, Matrix);

class Affine():
    """
    Some useful affine transformation matricies
    """

    def trans3d(v=[0, 0, 0]):
        v = vec(v, 3);
        m =  [ 
                [   1,    0,    0, v[0] ],
                [   0,    1,    0, v[1] ],
                [   0,    0,    1, v[2] ],
                [   0,    0,    0,    1 ] 
             ]
        return Matrix(m, affine=True)

    def trans2d(v):
        return Affine.trans3d(v).reduce(2, 2)

    def scale3d(v=[1, 1, 1]):
        v = vec(v, 3)
        m = [ 
                [v[0],    0,    0,    0 ],
                [   0, v[1],    0,    0 ],
                [   0,    0, v[2],    0 ],
                [   0,    0,    0,    1 ] 
            ]
        return Matrix(m, affine=True)

    def scale2d(v):
        return Affine.scale3d(v).reduce(2, 2)

    def xrot3d(a=0):
        sina = np.sin(a)
        cosa = np.cos(a)
        m = [ 
                [     1,     0,     0,     0 ],
                [     0,  cosa, -sina,     0 ],
                [     0,  sina,  cosa,     0 ],
                [     0,     0,     0,     1 ] 
            ]
        return Matrix(m, affine=True)

    def yrot3d(a=0):
        sina = np.sin(a)
        cosa = np.cos(a)
        m = [ 
                [  cosa,     0,  sina,     0 ],
                [     0,     1,     0,     0 ],
                [ -sina,     0,  cosa,     0 ],
                [     0,     0,     0,     1 ] 
            ]
        return Matrix(m, affine=True)

    def zrot3d(a=0):
        sina = np.sin(a)
        cosa = np.cos(a)
        m = [ 
                [  cosa, -sina,     0,     0 ],
                [  sina,  cosa,     0,     0 ],
                [     0,     0,     1,     0 ],
                [     0,     0,     0,     1 ] 
            ]
        return Matrix(m, affine=True)

    def rot3d(v=[0,0,0]):
        return Affine.zrot3d(v[2]) @ Affine.yrot3d(v[1]) @ Affine.xrot3d(v[0])

    def rot3d_from_to(fr, to):
        fr  = unit(point3d(fr))
        to  = unit(point3d(to))
        u   = vector_axis(fr, to)
        ang = vector_angle(fr, to)
        c   = np.cos(ang)
        c2  = 1 - c
        s   = np.sin(ang)

        m = [
            [u.x * u.x * c2 + c,         u.x * u.y * c2 - u.z * s,   u.x * u.z * c2 + u.y * s,  0],
            [u.y * u.x * c2 + u.z * s,   u.y * u.y * c2 + c,         u.y * u.z * c2 + u.x * s,  0],
            [u.z * u.x * c2 - u.y * s,   u.z * u.y * c2 + u.x * s,   u.z * u.z * c2 + c      ,  0],
            [                       0,                          0,                          0,  1],
        ]
        return Matrix(m, affine=True)

    def rot2d(a):
        return Affine.zrot3d(a).reduce(2, 2)

    def around_center(cp, m):
        cp = vec(cp, 3);
        m1 = Affine.trans3d(cp)
        m2 = Affine.trans3d(-cp)
        return m1 @ m @ m2


class Matrix():
    """
    Some general purpose matrix functions with the ability to transparently
    convert operands to affine when required.
    """

    def __init__(self, val=None, affine=False):
        self.is_affine = affine
        self.matrix = None
        self.type = np.float32
        if isinstance(val, Matrix):
            self.matrix = val.matrix
            self.is_affine = val.is_affine
        elif isinstance(val, np.ndarray):
            self.matrix = val
        elif val is not None:
            self.matrix = np.array(val, dtype=self.type)

    def adj(self):
        return Matrix(np.matrix_transpose(self.matrix), affine=self.is_affine)

    def append(self, val):
        if self.matrix is None:
            self.matrix = np.array(val, dtype=self.type)
        else:
            self.matrix = np.append(self.matrix, val, axis=0)

    def round(self):
        self.matrix = self.matrix.round(6)
        return self

    def format(self):
        twoD = True if isinstance(self.matrix[0], np.ndarray) else False
        res = "[\n  " if twoD else "[ "
        for ii in range(len(self.matrix)):
            if twoD:
                for jj in range(len(self.matrix[0])):
                    res += f"{self.matrix[ii][jj]:10.2f} "
            else:
                res += f"{self.matrix[ii]:10.2f} "
            if twoD:
                res += "\n  "
        res += "]"
        return res

    def expand(self, cols, rows=None, ones=False):
        nrows = len(self.matrix)
        ncols = len(self.matrix[0])
        if ones:
            z = np.ones((nrows, cols - ncols), dtype=self.type) 
        else:
            z = np.zeros((nrows, cols - ncols), dtype=self.type)
        self.matrix = np.concat((self.matrix, z), 1, dtype=self.type)
        if rows is not None:
            z = np.zeros((rows - nrows, cols), dtype=self.type)
            self.matrix = np.concat((self.matrix, z), 0, dtype=self.type)

    def reduce(self, cols=None, rows=None, keep_affine=True):
        nrows = len(self.matrix)
        if self.is_affine and keep_affine: nrows -= 1
        if rows is None: rows = len(self.matrix) - self.is_affine
        ncols = len(self.matrix[0])
        if self.is_affine and keep_affine: ncols -= 1
        if cols is None: cols = len(self.matrix[0]) - self.is_affine
        self.is_affine = keep_affine
        return Matrix(np.delete(np.delete(self.matrix, np.s_[rows:nrows], 0), np.s_[cols:ncols], 1), 
                      affine=self.is_affine)

    def prune(self, row, col):
        return Matrix(np.delete(np.delete(self.matrix, row, 0), col, 1), affine=self.is_affine)

    def affine(self):
        if self.is_affine: return self

        rows = len(self.matrix)
        cols = len(self.matrix[0])
        self.expand(cols + 1, rows + 1)
        self.matrix[rows][cols] = 1
        self.is_affine = True
        return self

    def check_affine(self, other):
        C = type(self)
        if isinstance(other, list) or isinstance(other, tuple):
            other = C(other);
        if isinstance(other, Matrix) and (self.is_affine or other.is_affine):
            self.affine()
            other.affine()
        return other

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, key):
        if isinstance(key, int):
            if isinstance(self.matrix[key], np.ndarray):
                return Vector(self.matrix[key], affine=self.is_affine)
            else:
                return self.matrix[key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self.matrix))
            return [self.matrix[ii] for ii in range(start, stop, step)]

    def __sub__(self, m):
        C = type(self)
        s = C(self.matrix - m.matrix, affine=self.is_affine)
        return s

    def __rsub__(self, m):
        C = type(self)
        return C(m.matrix - self.matrix, affine=self.is_affine)

    def __radd__(self, m):
        m = self.check_affine(m)
        C = type(self)
        return C(self.matrix + m.matrix, affine=self.is_affine)

    def __add__(self, m):
        m = self.check_affine(m)
        C = type(self)
        return C(self.matrix + m, affine=self.is_affine)

    def __truediv__(self, x):
        C = type(self)
        return C(self.matrix / x, affine=self.is_affine)

    def __mul__(self, x):
        C = type(self)
        return C(self.matrix * x, affine=self.is_affine)

    def __neg__(self):
        C = type(self)
        return C(self.matrix * -1, affine=self.is_affine)

    def __rmul__(self, x):
        C = type(self)
        return C(self.matrix * x, affine=self.is_affine)

    def __matmul__(self, m):
        m = self.check_affine(m)
        if isinstance(m, Points):
            return Points(np.matvec(self.matrix, m.matrix), affine=self.is_affine)
        if isinstance(self, Points):
            return Points(np.vecmat(self.matrix, m.matrix), affine=self.is_affine)
        return Matrix(self.matrix @ m.matrix, affine=self.is_affine)

    def __rmatmul__(self, m):
        m = self.check_affine(m)
        if isinstance(m, Points):
            return Points(np.vecmat(m.matrix, self.matrix), affine=self.is_affine)
        if isinstance(self, Points):
            return Points(np.matvec(m.matrix, self.matrix), affine=self.is_affine)
        return Matrix(m.matrix @ self.matrix, affine=self.is_affine)

    def __str__(self):
        return str(f"Matrix: {self.matrix}")

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def deaffine(self):
        if not self.is_affine:
            return self

        C = type(self)
        m = np.copy(self.matrix)
        if isinstance(self, Vector):
            m = np.delete(m, len(self.matrix) - 1, 0)
        else:
            if not isinstance(self, Points):
                m = np.delete(m, len(self.matrix) - 1, 0)
            m = np.delete(m, len(self.matrix[0]) - 1, 1)

        return C(m, affine=False)

    def list(self):
        return self.matrix.round(6).tolist()

    def det(self):
        return Matrix(np.linalg.det(self.matrix), affine=self.is_affine)

    '''
    def minors(self):
        r = Matrix(affine=self.is_affine)
        for row in range(len(self)):
            r.append([0] * len(self[0]))
            for col in range(len(self[0])):
                n = self.prune(row, col)
                r[row][col] = n.det()
        return r

    def cofactor(self):
        r = Matrix(affine=self.is_affine)
        row_s = 1
        for row in range(len(self)):
            r.append([0] * len(self[0]))
            s = row_s
            for col in range(len(self[0])):
                r[row][col] = s * self[row][col]
                s *= -1
            row_s *= -1
        return r
    '''

    def inv(self):
        '''
        return (1 / self.det()) * self.minors().cofactor().adjugate()
        '''
        return Matrix(np.linalg.inv(self.matrix), affine=self.is_affine)

class Vector(Matrix):
    """
    Single dimension Matrix needs a little special treatment
    """

    def __init__(self, val=None, affine=False):
        super().__init__(val, affine)

    def affine(self):
        if self.is_affine: return self

        cols = len(self.matrix)
        self.expand(cols + 1)
        self.matrix[cols] = 1
        self.is_affine = True
        return self

    def expand(self, cols, ones=False):
        ncols = len(self.matrix)
        if ones:
            z = np.ones((1, cols - ncols), dtype=self.type) 
        else:
            z = np.zeros((1, cols - ncols), dtype=self.type)
        self.matrix = np.concat((self.matrix, z), None, dtype=self.type)

    def append(self, val):
        np.append(self.matrix, val, axis=0)

    def __mul__(self, x):
        C = type(self)
        return C(self.matrix * x, affine=self.is_affine)

    def __rmul__(self, x):
        C = type(self)
        return C(self.matrix * x, affine=self.is_affine)

    def __matmul__(self, m):
        m = self.check_affine(m)
        C = type(self)
        # numpy.float has an __mul__ that supercedes our __rmul__ for vectors
        # and matricies. So we need to cast away the numpy.float to prevent use
        # of it.
        r = float(self.matrix @ m.matrix)
        return r

    def __rmatmul__(self, m):
        m = self.check_affine(m)
        return m.matrix @ self.matrix

    def __sub__(self, m):
        C = type(self)
        s = C(self.matrix - m.matrix, affine=self.is_affine)
        return s

    def cross(self, p):
        # Making the call into numpy for large numbers of small vectors is
        # a performance problem.  So I'm micro-optimizing the common case.
        if isinstance(p, Vector) and len(self.matrix) == 3:
            x = self.matrix[1] * p.matrix[2] - self.matrix[2] * p.matrix[1]
            y = self.matrix[2] * p.matrix[0] - self.matrix[0] * p.matrix[2]
            z = self.matrix[0] * p.matrix[1] - self.matrix[1] * p.matrix[0]
            return Vector([x, y, z])
        # numpy cross vectors must not be affine
        a = self.deaffine()
        b = p.deaffine()
        C = type(p)
        return C(np.linalg.cross(a.matrix, b.matrix), affine=False)

    def unit(self, error=None):
        if error is not None and norm(self) < eps:
            return error
        return self / norm(self)

    def norm(self):
        return np.linalg.norm(self)

    def abs(self):
        return Vector(np.fabs(self.matrix), affine=self.is_affine)

    def __getattr__(self, name):
        return (self[0] if name == "x" else 
                self[1] if name == "y" else 
                self[2] if name == "z" else super().__getattr__(name))

    def __setattr__(self, name, val):
        if name == "x":
            self[0] = val
        elif name == "y":
            self[1] = val
        elif name == "z":
            self[2] = val
        else:
            super().__setattr__(name, val)

    def __getitem__(self, key):
        return self.matrix[key]

    def __setitem__(self, key, val):
        self.matrix[key] = val

class Points(Matrix):
    """
    Points are a stack of single dimension Matricies, so also need some special treatment
    """

    def __init__(self, val=None, affine=False):
        super().__init__(val, affine)

    def concat_init(plist):
        res = Points(plist[0])
        res.concat(plist[1:])
        return res

    def concat(self, plist):
        for p in plist:
            self.append(p)

    def points3d(self):
        if len(self.matrix[0]) - self.is_affine == 3:
            return self
        pl = self.matrix.tolist()
        for point in pl:
            point = point.insert(2, 0)
        return Points(pl, affine=self.is_affine)

    def append(self, val):
        val = self.check_affine(val)
        if isinstance(val, Vector):
            val = Points([ val ])
        if self.matrix is None:
            self.matrix = np.array(val, dtype=self.type)
        else:
            self.matrix = np.append(self.matrix, val, axis=0)

    def cross(self, p):
        return Points(np.linalg.cross(self.matrix, p), affine=self.is_affine)

    def __matmul__(self, m):
        m = self.check_affine(m)
        return Points(np.vecmat(self.matrix, m.matrix), affine=self.is_affine)

    def __rmatmul__(self, m):
        m = self.check_affine(m)
        return Points(np.matvec(m.matrix, self.matrix), affine=self.is_affine)

    def affine(self):
        if self.is_affine: return self

        if self.matrix is not None and len(self.matrix) > 0:
            cols = len(self.matrix[0])
            self.expand(cols + 1, ones=True)
        self.is_affine = True
        return self

    def mean(self):
        val = self.matrix
        s = val[0]
        for v in val[1:]:
            s += v
        m = s / len(val)
        return m

    def deriv(self, step=1, closed=False):
        npoints = len(self.matrix)
        if closed:
            d = Points([ (self[(ii + 1) % npoints] - self[(npoints + ii - 1) % npoints]) / 2 / step
                        for ii in range(npoints) ])
        else:
            if npoints < 3:
                first = self[1] - self[0]
                last  = self[npoints - 1] - self[npoints - 2]
            else:
                first = 3 * (self[1] - self[0]) - (self[2] - self[1])
                last =  (self[npoints - 3] - self[npoints - 2]) - 3 * (self[npoints - 2] - self[npoints - 1])

            l =       [first / 2 / step]
            l.extend( [(self[ii + 1] - self[ii - 1]) / 2 / step for ii in range(1, npoints - 1)] )
            l.extend( [last  / 2 / step] )
            d = Points(l)

        return d

    def unit(self):
        points = self.deaffine()
        u = Points( [ p.unit() for p in points ] )
        return u

    def bounds(self):
        lo = Vector(self[0])
        hi = Vector(self[0])
        for p in self:
            if p.x < lo.x: lo.x = p.x
            if p.y < lo.y: lo.y = p.y
            if p.z < lo.z: lo.z = p.z
            if p.x > hi.x: hi.x = p.x
            if p.y > hi.y: hi.y = p.y
            if p.z > hi.z: hi.z = p.z
        return [lo, hi]

"""
Helper vectors and functions that return vectors
"""
RT = Vector([ 1,  0,  0 ])
LT = Vector([-1,  0,  0 ])
BK = Vector([ 0,  1,  0 ])
FT = Vector([ 0, -1,  0 ])
UP = Vector([ 0,  0,  1 ])
DN = Vector([ 0,  0, -1 ])
CT = Vector([ 0,  0,  0 ])

TP = UP
BT = DN

def ct(v=1):
    return (CT * v).deaffine().list()

def top(v=1):
    return (TP * v).deaffine().list()

def bottom(v=1):
    return (BT * v).deaffine().list()

def up(v=1):
    return (UP * v).deaffine().list()

def dn(v=1):
    return (DN * v).deaffine().list()

def lt(v=1):
    return (LT * v).deaffine().list()

def rt(v=1):
    return (RT * v).deaffine().list()

def ft(v=1):
    return (FT * v).deaffine().list()

def bk(v=1):
    return (BK * v).deaffine().list()


"""
Some extra trig and linear algebra utilities that I needed
"""

def mean(val):
    s = val[0]
    for v in val[1:]:
        s += v
    m = s / len(val)
    return m

def norm(val):
    n = np.linalg.norm(val)
    if n != n:
        n = 1
    return n

def unit(val):
    if norm(val) == 0:
        return val
    return val / norm(val)

def cross3d(a, b):
    return Vector(np.linalg.cross(a, b))

def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0];

def det2d(a, b):
    return cross2d(a, b)

def posmod(x, m):
    return (x % m + m) % m

def vector_axis(v1, v2):
    eps = 1e-6
    w1 = point3d(unit(v1))
    w2 = point3d(unit(v2))
    if norm(w1 - w2) > eps and norm(w1 + w2) > eps:
        w3 = w2
    elif norm(w2.abs() - UP) > eps:
        w3 = Vector(UP)
    else:
        w3 = Vector(RT)
    return unit(cross3d(w1, w3))

def vector_angle(v1, v2):
    return np.acos(constrain((v1 @ v2) / (norm(v1) * norm(v2)), -1, 1))
    
def circle_2tan(r, corner):
    v1      = unit(Vector(corner[0] - corner[1]))
    v2      = unit(Vector(corner[2] - corner[1]))
    vmid    = unit(mean([v1, v2]))
    n       = vector_axis(v1, v2)
    a       = vector_angle(v1, v2)
    hyp     = r / np.sin(a / 2)
    cp      = Vector(corner[1] + hyp * vmid)
    x       = hyp * np.cos(a / 2)
    tp1     = Vector(corner[1] + x * v1)
    tp2     = Vector(corner[1] + x * v2)

    return (cp, n, tp1, tp2)

def law_of_sines(a, A, b=None, B=None):
    r = a / np.sin(A)
    result = r * np.sin(B) if B is not None else np.asin(constrain(b / r, -1, 1))
    return result

def polar_to_xy(r, theta):
    return [ r * np.cos(theta), r * np.sin(theta) ]

def xyz_to_spherical(v):
    return [ norm(v), np.atan2(v.y, v.x), np.atan2(norm([v.x, v.y]), v.z) ]

def segs(r):
    """
    Computes the number of segments to construct rounded objects out of
    based on the limits set by the special variables fn, fs, and fa
    """

    (fn, fa, fs) = get_fnas()
    if fn != None and fn > 0:
        if fn > 3: return fn
        return 3

    if r > 0:
        segs = np.ceil(np.fmax(3, np.fmin(360 / fa, np.fabs(r) * 2 * np.pi / fs)))
    else:
        segs = np.ceil(np.fmax(3, 360 / fa))
    return segs

def constrain(v, minval, maxval):
    return np.fmin(maxval, np.fmax(minval, v))

def path_tangents(path, closed):
    return path.deriv(closed=closed).unit()

def frame_map(x=None, y=None, z=None):
    if x is not None: x = x.unit(RT)
    if y is not None: y = y.unit(BK)
    if z is not None: z = z.unit(UP)

    if x is None:
        map = Matrix([y.cross(z), y, z]).adj().affine()
    elif y is None:
        map = Matrix([x, z.cross(x), z]).adj().affine()
    elif z is None:
        map = Matrix([x, y, x.cross(y)]).adj().affine()
    else:
        map = Matrix([x, y, z]).adj().affine()

    return map

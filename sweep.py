from transforms import *
from dataclasses import dataclass
from utils import *
import copy

def lofttri(sh1, sh2, idx1, idx2, np1, np2, trimax, col_wrap):
    trilist = []
    i1 = i2 = tc1 = tc2 = 0
    np1c = np1 + col_wrap
    np2c = np2 + col_wrap
    while True:
        if np1 != np2:
            t1 = i1 + 1 if i1 < np1c else np1c
            t2 = i2 + 1 if i2 < np2c else np2c
            if t1 >= np1c and t2 >= np2c:
                return trilist


            d12 = norm(sh2[t2 % np2] - sh1[i1 % np1]) if t2 < np2c else 9e9
            d21 = norm(sh1[t1 % np1] - sh2[i2 % np2]) if t1 < np1c else 9e9

            if d12 < d21:
                userow = 2 if tc1 < trimax else 1
            else:
                userow = 1 if tc2 < trimax else 2

            if userow == 1:
                newt    = t1 if t1 < np1c else i1
                newofft = idx1 + newt % np1
            else:
                newt    = t2 if t2 < np2c else i2
                newofft = idx2 + newt % np2

            tc1 = tc1 + 1 if d12 < d21 and tc1 < trimax else 0
            tc2 = tc2 + 1 if d21 < d12 and tc2 < trimax else 0

            tri = [idx1 + i1 % np1, idx2 + i2 % np2, newofft]
            trilist.append(tri)
            if userow == 1 and t1 < np1c:
                i1 = t1
            if userow == 2 and t2 < np2c:
                i2 = t2
        else:
            t = i1 + 1
            if t >= np1c:
                return trilist

            d12 = norm(sh2[t % np1] - sh1[i1 % np1]) if t < np1c else 9e9
            d21 = norm(sh1[t % np1] - sh2[i1 % np1]) if t < np1c else 9e9
            tri = [ [idx1 + i1 % np1, idx2 + i1 % np1, idx2 +  t % np1 if d12 < d21 else idx1 +  t % np1],
                    [idx2 +  t % np1, idx1 +  t % np1, idx1 + i1 % np1 if d12 < d21 else idx2 + i1 % np1] ]
            trilist.extend(tri)
            i1 = i2 = t

def tri_array(shapes):
    nshapes = len(shapes)
    # This is essentially the col_wrap=true version of BOSL2 vnf_tri_array
    # Eliminate duplicate points at the column wrap
    # And flatten shapes into verts
    verts = Points()
    for shape in shapes:
        if shape[0] == shape[-1]:
            shape.remove(-1)
        verts.append(shape.deaffine())

    npoints = [len(shape) for shape in shapes]
    ii  = 0
    idx = []
    for n in npoints:
        idx.append(ii)
        ii += n
    idx.append(ii)

    cap1 = [ p for p in range(npoints[0]) ]
    cap2 = [ p for p in range(idx[nshapes] - 1, idx[nshapes - 1] - 1, -1) ]

    faces = []
    faces.append(cap1)
    for ii in range(nshapes - 1):
        jj = (ii + 1) % nshapes
        max_extra_edges = np.max([1, abs(npoints[ii] - npoints[jj])])
        f = lofttri(shapes[ii], shapes[jj], idx[ii], idx[jj], npoints[ii], npoints[jj], trimax=max_extra_edges, col_wrap=True)
        faces.extend(f)
    faces.append(cap2)

    return [verts, faces]

def generate_faces(shapes, closed, style="default", cull=False):
    """
    This function assums the shapes all have the same number of points
    """

    if style == "min_edge": style = 1
    if style == "tri_array":
        return tri_array(shapes)
    else:                   style = 0

    verts = Points()
    for shape in shapes:
        verts.append(shape.deaffine())

    faces = []
    # Cap the ends
    nshapes = len(shapes)
    npoints = len(shapes[0])
    if not closed:
        cap = [ii for ii in range(npoints)]
        faces.append(cap)
        cap = [ii for ii in range(nshapes * npoints - 1, (nshapes - 1) * npoints - 1, -1)]
        faces.append(cap)
    for ss in range(nshapes - (not closed)):
        for pp in range(npoints):
            p1 = ((ss + 0) % nshapes) * npoints + ((pp + 0) % npoints)
            p2 = ((ss + 1) % nshapes) * npoints + ((pp + 0) % npoints)
            p3 = ((ss + 1) % nshapes) * npoints + ((pp + 1) % npoints)
            p4 = ((ss + 0) % nshapes) * npoints + ((pp + 1) % npoints)

            if style == 1:
                d42 = norm(verts[p4] - verts[p2])
                d13 = norm(verts[p1] - verts[p3])
                # Use short edge, but don't use degenerate edges
                if d42 < d13 and d42 > eps:
                    faces.extend([ [p2, p4, p1], [p3, p4, p2] ])
                elif d13 > eps:
                    faces.extend([ [p2, p3, p1], [p3, p4, p1] ])
            else:
                faces.extend([ [p2, p3, p1], [p3, p4, p1] ])

    # Remove degenerates
    # This is slow, so I've made it optional as it's often not needed
    if cull:
        culled_faces = []
        for face in faces:
            edge1 = verts[face[1]] - verts[face[0]]
            edge2 = verts[face[2]] - verts[face[0]]
            if edge1.cross(edge2).norm() > eps:
                culled_faces.append(face)
    else:
        culled_faces = faces

    return [verts, culled_faces]

def get_shape(shape, context=None, call=False):
    from shapes import Object

    if not callable(shape):
        if isinstance(shape, Object):
            shape = shape.mesh().points
        shape = Points(shape).points3d()
    elif call:
        shape = shape(context)
        if shape is not None:
            if isinstance(shape, Object):
                shape = shape.mesh().points
            shape = Points(shape).points3d()
    else:
        return None

    return shape

def sweep(shape, transforms, closed=False, context=None, style="min_edge"):
    """
    shape       - A collection of points
                  May be a callback
                  Callback is called once each iteration with
                  context parameter
    transforms  - A list of 4x4 Affine transforms
                  May be a callback
                  Callback is called once each iteration with
                  context parameter
    """

    a_shape = get_shape(shape)
    transformed_shapes = []
    if callable(transforms):
        while True:
            transform = transforms(context)
            if transform is None:
                break
            if callable(shape):
                a_shape = get_shape(shape, context=context, call=True)
                if a_shape is None:
                    break
            if len(a_shape) > 0:
                transformed_shapes.append(transform @ a_shape)
    else:
        for transform in transforms:
            if callable(shape):
                a_shape = get_shape(shape, context=context, call=True)
                if a_shape is None:
                    break
            if len(a_shape) > 0:
                transformed_shapes.append(transform @ a_shape)

    return generate_faces(transformed_shapes, style=style, closed=closed)

def path_sweep(shape, path, closed=False):

    path = Points(path).points3d()
    tangents = path_tangents(path, closed)
    normal = BK if np.fabs(tangents[0].z) > 1 / np.sqrt(2) else UP

    npoints = len(path)
    ynorm   = normal - (normal @ tangents[0]) * tangents[0]
    rot     = frame_map(y=ynorm, z=tangents[0])
    r       = ynorm
    rotations = [rot]
    for ii in range(len(tangents) - 1 + closed):
        v1 = path[(ii + 1) % npoints] - path[ii % npoints]
        c1 = v1 @ v1
        rL = r - 2 * (v1 @ r) / c1 * v1
        tL = tangents[ii % npoints] - 2 * (v1 @ tangents[ii % npoints]) / c1 * v1
        v2 = tangents[(ii + 1) % npoints] - tL
        c2 = v2 @ v2
        r  = rL - (2 / c2) * (v2 @ rL) * v2
        rot = frame_map(y=r, z=tangents[(ii + 1) % npoints])
        rotations.append(rot)

    last_rot    = rotations[len(rotations) - 1]
    ref_rot     = rotations[0] if closed else last_rot
    mismatch    = last_rot.adj() @ ref_rot
    correction  = np.atan2(mismatch[1][0], mismatch[0][0])
    twistfix    = correction % tau

    transforms = [ Affine.trans3d(path[ii]) @ rotations[ii] @ Affine.zrot3d(twistfix * (ii / (len(path) + closed)))
                   for ii in range(len(path)) ]
    if closed:
        transforms.append( Affine.trans3d(path[0]) @ rotations[0] @ Affine.zrot3d(-correction + correction % tau) )

    return sweep(shape, transforms, closed)

def roundingSweepShape(context):

    off = 0

    # This is a bit of hack. I need a copy
    # of the scad object and the copy.copy() function
    # doesn't work.
    shape = context.shape.offset(delta=0)
    # Bottom round resize shape
    # Bottom and top zones may overlap
    if context.pos < np.fabs(context.r[0]):
        z = 1 - context.pos / np.fabs(context.r[0])
        off = -(1 - np.sin(np.acos(z))) * context.r[0]
        shape = shape.offset(delta=float(off))

    # Top round resize shape
    if context.h - context.pos < np.fabs(context.r[1]):
        z = 1 - (context.h - context.pos) / np.fabs(context.r[1])
        off = -(1 - np.sin(np.acos(z))) * context.r[1]
        shape = shape.offset(delta=float(off))

    return shape

def roundingSweepTransform(context):
    if context.next is None: return None
    context.pos = context.next

    m = Affine.trans3d([0, 0, context.pos])
    if context.next == context.h:
        context.next = None
    else:
        context.next += context.step
        if context.next > context.h:
            context.next = context.h

    return m

@dataclass()
class RoundingSweepContext():
    # Shape context
    shape   :   ...     = None
    r       :   list    = None

    # Transform context
    h       :   float   = None
    step    :   float   = None
    pos     :   float   = 0
    next    :   float   = 0

def rounding_sweep(shape, h, r):
    """
    A linear vertical sweep of shape offset by r[0] on bottom and r[1] on top
    """
    fn, fa, fs = get_fnas()
    step = fs if fs > 0 else h / fn if fn > 0 else 0.3
    context = RoundingSweepContext(shape=shape, h=h, r=r, step=step)
    return sweep(shape=roundingSweepShape, transforms=roundingSweepTransform, closed=False, context=context, style="tri_array")

@dataclass()
class RotateSweepContext():
    index   :   int     = 0
    steps   :   int     = None
    step    :   float   = None
    start   :   float   = None
    span    :   float   = None

def rotateSweepTransform(context):
    if context.index >= context.steps: return None
    m = Affine.rot3d([np.pi / 2, 0,
        context.start + context.span - context.index * context.step])
    context.index += 1
    return m

def rotate_sweep(shape, angle=360):
    if not isinstance(angle, list):
        start = 0
        span  = angle
    else:
        start = angle[0]
        span  = angle[1]

    a_shape = get_shape(shape)
    if a_shape is not None:
        [lo, hi] = a_shape.bounds()
        steps = np.ceil(segs(hi.x) * span / 360)
    else:
        steps = np.ceil(segs(0) * span / 360)

    closed = span == 360
    start   = np.radians(start)
    span    = np.radians(span)
    step    = span / steps
    if span < tau: steps += 1
    context = RotateSweepContext(index=0, steps=steps, step=step, start=start, span=span)

    return sweep(shape, transforms=rotateSweepTransform, closed=closed, context=context)

def plot3d(func, x_range, y_range, base=1, context=None):

    minz = None
    plot = []
    for y in y_range:
        dx = Points(shape=[len(x_range) + 2, 3])    # preallocate the row, it's faster
        dx[0] = [x_range[0], y, 0]                  # place holder for the base
        xx = 1
        for x in x_range:
            if context is not None:
                z = func(x, y, context)
            else:
                z = func(x, y)
            if minz is None or z < minz: minz = z
            dx[xx] = [x, y, z]
            xx += 1
        dx[xx] = [x_range[-1], y, 0]    # place holder for the base
        plot.append(dx)

    # Add left and right edges
    bottom = minz - base
    for row in plot:
        row[0].z  = bottom
        row[-1].z = bottom

    return generate_faces(plot, closed=False)

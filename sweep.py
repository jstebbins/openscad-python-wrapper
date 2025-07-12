from transforms import *
from dataclasses import dataclass
from utils import *

def generate_faces(shapes, closed):
    """
    This function assums the shapes all have the same number of points
    """

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

            d42 = norm(verts[p4] - verts[p2])
            d13 = norm(verts[p1] - verts[p3])
            # Use short edge, but don't use degenerate edges
            if d42 < d13 and d42 > eps:
                faces.extend([[p2, p4, p1], [p3, p4, p2]])
            elif d13 > eps:
                faces.extend([[p2, p3, p1], [p3, p4, p1]])

    # remove degenerates
    culled_faces = []
    for face in faces:
        edge1 = verts[face[1]] - verts[face[0]]
        edge2 = verts[face[2]] - verts[face[0]]
        if edge1.cross(edge2).norm() > eps:
            culled_faces.append(face)

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

def sweep(shape, transforms, closed=False, context=None):
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
            transformed_shapes.append(transform @ a_shape)
    else:
        for transform in transforms:
            if callable(shape):
                a_shape = get_shape(shape, context=context, call=True)
                if a_shape is None:
                    break
            transformed_shapes.append(transform @ a_shape)

    return generate_faces(transformed_shapes, closed)

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
        dx = Points([ [x_range[0], y, 0] ])         # place holder for the base
        for x in x_range:
            if context is not None:
                z = func(x, y, context)
            else:
                z = func(x, y)
            if minz is None or z < minz: minz = z
            dx.append(Points([ [x, y, z] ]))
        dx.append(Points([ [x_range[-1], y, 0] ]))  # place holder for the base
        plot.append(dx)

    # Add left and right edges
    bottom = minz - base
    for row in plot:
        row[0].z  = bottom
        row[-1].z = bottom

    return generate_faces(plot, closed=False)

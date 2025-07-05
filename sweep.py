from transforms import *

def generate_faces(shapes):
    """
    This function assums the shapes all have the same number of points
    """

    verts = Points()
    for shape in shapes:
        verts.append(shape)

    faces = []
    # Cap the ends
    nshapes = len(shapes)
    npoints = len(shapes[0])
    cap = [ii for ii in range(npoints - 1, -1, -1)]
    faces.append(cap)
    cap = [ii for ii in range((nshapes - 1) * npoints, nshapes * npoints)]
    faces.append(cap)
    for ss in range(nshapes - 1):
        for pp in range(npoints):
            p1 = ((ss + 0) % nshapes) * npoints + ((pp + 0) % npoints)
            p2 = ((ss + 1) % nshapes) * npoints + ((pp + 0) % npoints)
            p3 = ((ss + 1) % nshapes) * npoints + ((pp + 1) % npoints)
            p4 = ((ss + 0) % nshapes) * npoints + ((pp + 1) % npoints)

            # Check if any edges are coincedent, i.e. degenerate face
            d42 = norm(verts[p4] - verts[p2])
            d13 = norm(verts[p1] - verts[p3])
            if d42 < d13:
                faces.extend([[p1, p4, p2], [p2, p4, p3]])
            else:
                faces.extend([[p1, p3, p2], [p1, p4, p3]])

    culled_faces = []
    for face in faces:
        edge1 = verts[face[1]] - verts[face[0]]
        edge2 = verts[face[2]] - verts[face[0]]
        if norm(edge1.cross(edge2)) > eps:
            culled_faces.append(face)

    return [verts, culled_faces]

def sweep(shape, transforms):
    """
    shape       - A collection of points
    transforms  - a list of 4x4 Affine transforms
    """

    shape = shape.points3d()
    transformed_shapes = [transform @ shape for transform in transforms]

    return generate_faces(transformed_shapes)

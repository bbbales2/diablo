#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy

#from libc.math cimport exp

cpdef inline float polyint(int n, int m, int l):
    cdef float xtmp, ytmp

    if n < 0 or m < 0 or l < 0 or n % 2 > 0 or m % 2 > 0 or l % 2 > 0:
        return 0.0

    xtmp = 2 * 0.5**(n + 1)
    ytmp = 2 * 0.5**(m + 1) * xtmp
    return 2 * 0.5**(l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1))

cpdef inline numpy.ndarray[numpy.float_t, ndim = 1] buildHist(numpy.ndarray[numpy.uint8_t, ndim = 3] im):
    cdef numpy.ndarray[numpy.float_t, ndim = 1] hist

    cdef int i, j, M, N, cr, cg, cb

    M = im.shape[0]
    N = im.shape[1]
    S = im.shape[2]

    hist = numpy.zeros((4096), dtype = 'float')

    for i in range(M):
        for j in range(N):
            if S == 4 and im[i, j, 3] < 5:
                continue

            cr = im[i, j, 0] >> 4
            cg = im[i, j, 1] >> 4
            cb = im[i, j, 2] >> 4

            #hist[cr + 32] += 1.0
            #hist[cg + 16] += 1.0
            #hist[cb] += 1.0

            hist[cr * 16 * 16 + cg * 16 + cb] += 1.0

    return hist / numpy.linalg.norm(hist)

cpdef image(int b, numpy.ndarray[numpy.uint8_t, ndim = 3] im, numpy.ndarray[numpy.float_t, ndim = 2] refs):
    cdef int M, N, mb, nb, bi, bj, ii, jj, r

    cdef float norm, total

    cdef numpy.ndarray[numpy.float_t, ndim = 3] intensity
    cdef numpy.ndarray[numpy.uint8_t, ndim = 3] tmp
    cdef numpy.ndarray[numpy.float_t, ndim = 3] hists
    cdef numpy.ndarray[numpy.float_t, ndim = 2] mTm

    M = im.shape[0]
    N = im.shape[1]
    R = refs.shape[1]
    F = refs.shape[0]

    mb = M / b
    nb = N / b

    intensity = numpy.zeros((mb, nb, R), dtype = 'float')
    hists = numpy.zeros((mb, nb, F), dtype = 'float')
    tmp = numpy.zeros((b, b, 3), dtype = 'uint8')
    mTm = refs.T.dot(refs)

    for bi in range(mb):
        for bj in range(nb):
            for ii in range(0, b):
                for jj in range(0, b):
                    tmp[ii, jj, 0] = im[bi * b + ii, bj * b + jj, 0]
                    tmp[ii, jj, 1] = im[bi * b + ii, bj * b + jj, 1]
                    tmp[ii, jj, 2] = im[bi * b + ii, bj * b + jj, 2]

            hists[bi, bj, :] = buildHist(tmp)

    intensity = numpy.einsum('ijk, kr', hists, refs)

    for bi in range(mb):
        for bj in range(nb):
            #intensity[bi, bj] = numpy.linalg.solve(mTm, intensity[bi, bj])

            #total = 0
            #for r in range(R):
            #    if intensity[bi, bj, r] < 0:
            #        intensity[bi, bj, r] = 0

            #    total += intensity[bi, bj, r]

            #if total > 0.0:
            #    for r in range(R):
            #        intensity[bi, bj, r] /= total
            norm = numpy.linalg.norm(intensity[bi, bj])

            for ii in range(R):
                intensity[bi, bj, ii] /= norm

    return intensity

cpdef tuple offset(int R, numpy.ndarray[numpy.uint8_t, ndim = 3] im1, numpy.ndarray[numpy.uint8_t, ndim = 3] im2):
    cdef int M, N, dx, dy, i, j, mini, minj
    cdef double dr, dg, db, minV

    cdef numpy.ndarray[numpy.float_t, ndim = 2] result

    M = im1.shape[0]
    N = im1.shape[1]

    result = numpy.zeros((2 * R, 2 * R))

    for dx in range(-R, R):
        for dy in range(-R, R):
            for i in range(R, M - R, 48):
                for j in range(R, N - R, 48):
                    dr = float(im2[i, j, 0]) - float(im1[i + dy, j + dx, 0])
                    dg = float(im2[i, j, 1]) - float(im1[i + dy, j + dx, 1])
                    db = float(im2[i, j, 2]) - float(im1[i + dy, j + dx, 2])

                    result[dy + R, dx + R] += dr * dr + dg * dg + db * db

    minV = result[0, 0]
    mini = 0
    minj = 0

    for i in range(2 * R):
        for j in range(2 * R):
            if result[i, j] < minV:
                mini = i
                minj = j
                minV = result[i, j]

    return result, (mini - R, minj - R)

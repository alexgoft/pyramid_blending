import cv2
import numpy as np


def pyramid_blend(A, B, m, num_levels):

    # 1) Find the Gaussian Pyramids for given images and mask.
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()

    gpA = [GA]
    gpB = [GB]
    gpM = [GM]

    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)

        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # 2) From Gaussian Pyramids, find their Laplacian Pyramids.
    lpA = [gpA[num_levels - 1]]
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]

    for i in xrange(num_levels - 1, 0, -1):
        # !! At one point you get an image with an uneven number of rows, so reshape it. !!
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])

        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i], dstsize=size))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i], dstsize=size))

        lpA.append(LA)
        lpB.append(LB)

        gpMr.append(gpM[i - 1])

    # 4) Blend images according to mask in each level.
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):

        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # 5) Reconstruct
    ls_ = LS[0]
    for i in xrange(1, num_levels):

        # Need resize from the same reason as in (2)
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, np.float32(LS[i]))


    return ls_


if __name__ == '__main__':
    size = ()

    A_BGR = cv2.imread('./white.jpg')
    B_BGR = cv2.imread('./black.jpg')

    m = cv2.imread('./mask.jpg')

    # Image pixels are 0 or 255, we'll normalize to 0 and 1.0 so mask can be applied properly.
    m[m == 255] = 1.0

    result = pyramid_blend(A_BGR, B_BGR, m, 5)
    result = np.uint8(result)

    cv2.imwrite("./Catog.jpg", result)

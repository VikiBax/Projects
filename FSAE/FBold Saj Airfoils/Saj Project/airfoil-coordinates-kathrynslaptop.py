# Generate and plot the contour of an airfoil 
# using the PARSEC parameterization

# Repository & documentation:
# http://github.com/dqsis/parsec-airfoils
# -------------------------------------


# Import libraries
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, tan, pi
import pandas as pd


def pcoef(xte, yte, rle, x_cre, y_cre, d2ydx2_cre, th_cre, surface):
    # Docstrings
    """evaluate the PARSEC coefficients"""

    # Initialize coefficients
    coef = np.zeros(6)

    # 1st coefficient depends on surface (pressure or suction)
    if surface.startswith('p'):
        coef[0] = -sqrt(2 * rle)
    else:
        coef[0] = sqrt(2 * rle)

    # Form system of equations
    A = np.array([
        [xte ** 1.5, xte ** 2.5, xte ** 3.5, xte ** 4.5, xte ** 5.5],
        [x_cre ** 1.5, x_cre ** 2.5, x_cre ** 3.5, x_cre ** 4.5, x_cre ** 5.5],
        [1.5 * sqrt(xte), 2.5 * xte ** 1.5, 3.5 * xte ** 2.5, 4.5 * xte ** 3.5, 5.5 * xte ** 4.5],
        [1.5 * sqrt(x_cre), 2.5 * x_cre ** 1.5, 3.5 * x_cre ** 2.5, 4.5 * x_cre ** 3.5, 5.5 * x_cre ** 4.5],
        [0.75 * (1 / sqrt(x_cre)), 3.75 * sqrt(x_cre), 8.75 * x_cre ** 1.5, 15.75 * x_cre ** 2.5, 24.75 * x_cre ** 3.5]
    ])

    B = np.array([
        [yte - coef[0] * sqrt(xte)],
        [y_cre - coef[0] * sqrt(x_cre)],
        [tan(th_cre * pi / 180) - 0.5 * coef[0] * (1 / sqrt(xte))],
        [-0.5 * coef[0] * (1 / sqrt(x_cre))],
        [d2ydx2_cre + 0.25 * coef[0] * x_cre ** (-1.5)]
    ])

    # Solve system of linear equations
    X = np.linalg.solve(A, B)

    # Gather all coefficients
    coef[1:6] = X[0:5, 0]

    # Return coefficients
    return coef


def ppointsplain(*args, **kwargs):
    '''Alias for ppoints that returns a string with plain data format'''
    coords = ppoints(*args, **kwargs)
    # Iterate over coordinates, making a list of strings
    coordstrlist = ["{:.6f} {:.6f}".format(coord[0], coord[1])
                    for coord in coords]
    # Now join these strings with linebreaks in between
    return '\n'.join(coordstrlist)


def ppoints(cf_pre, cf_suc, npts=121, xte=1.0):
    '''
    Takes PARSEC coefficients, number of points, and returns list of
    [x,y] coordinates starting at trailing edge pressure side.
    Assumes trailing edge x position is 1.0 if not specified.
    Returns 121 points if 'npts' keyword argument not specified.
    '''
    # Using cosine spacing to concentrate points near TE and LE,
    # see http://airfoiltools.com/airfoil/naca4digit
    xpts = (1 - np.cos(np.linspace(0, 1, int(np.ceil(npts / 2))) * np.pi)) / 2
    # Take TE x-position into account
    xpts *= xte

    # Powers to raise coefficients to
    pwrs = (1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2)
    # Make [[1,1,1,1],[2,2,2,2],...] style array
    xptsgrid = np.meshgrid(np.arange(len(pwrs)), xpts)[1]
    # Evaluate points with concise matrix calculations. One x-coordinate is
    # evaluated for every row in xptsgrid
    evalpts = lambda cf: np.sum(cf * xptsgrid ** pwrs, axis=1)
    # Move into proper order: start at TE, over bottom, then top
    # Avoid leading edge pt (0,0) being included twice by slicing [1:]
    ycoords = -np.append(evalpts(cf_pre)[::-1], evalpts(cf_suc)[1:])
    xcoords = np.append(xpts[::-1], xpts[1:])
    # Return 2D list of coordinates [[x,y],[x,y],...] by transposing .T
    return np.array((xcoords, ycoords)).T


if __name__ == "__main__":
    # Read parsec parameters (user input) & assign to array
    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    yle = 0.0
    xte = 1.0
    yte = 0.0

    # LE radius
    rle = 0.03

    # Pressure (upper) surface parameters
    x_pre = 0.48
    y_pre = -0.068
    d2ydx2_pre = -0.75
    th_pre = -8

    # Suction (lower) surface parameters
    x_suc = 0.37
    y_suc = -0.18
    d2ydx2_suc = -1.4
    th_suc = -27

    # Flip y-coordinates
    y_pre = -y_pre
    y_suc = -y_suc

    # Evaluate pressure (lower) surface coefficients
    cf_pre = pcoef(xte, yte, rle, x_pre, y_pre, d2ydx2_pre, th_pre, 'pre')

    # Evaluate suction (upper) surface coefficients
    cf_suc = pcoef(xte, yte, rle, x_suc, y_suc, d2ydx2_suc, th_suc, 'suc')

    # Evaluate pressure (lower) surface points
    xx_pre = np.linspace(xte, xle, 101)
    yy_pre = -(cf_pre[0] * xx_pre ** (1 / 2) +
               cf_pre[1] * xx_pre ** (3 / 2) +
               cf_pre[2] * xx_pre ** (5 / 2) +
               cf_pre[3] * xx_pre ** (7 / 2) +
               cf_pre[4] * xx_pre ** (9 / 2) +
               cf_pre[5] * xx_pre ** (11 / 2))

    # Evaluate suction (upper) surface points
    xx_suc = np.linspace(xle, xte, 101)
    yy_suc = -(cf_suc[0] * xx_suc ** (1 / 2) +
               cf_suc[1] * xx_suc ** (3 / 2) +
               cf_suc[2] * xx_suc ** (5 / 2) +
               cf_suc[3] * xx_suc ** (7 / 2) +
               cf_suc[4] * xx_suc ** (9 / 2) +
               cf_suc[5] * xx_suc ** (11 / 2))

    x_test = np.linspace(0, 1, 201)

    for x in x_test:
        p = sorted(zip(xx_pre, yy_pre))
        s = sorted(zip(xx_suc, yy_suc))

        yp = np.interp(x, [xx for xx, yy in p], [yy for xx, yy in p])
        ys = np.interp(x, [xx for xx, yy in s], [yy for xx, yy in s])

        if yp < ys:
            print('Self-intersection detected')

    # Use parsecexport to save coordinate file
    # with ... as ... only opens the file for the block it executes, then closes it
    with open('parsec_airfoil.dat', 'w') as f:
        plain_coords = ppointsplain(cf_pre, cf_suc, 121, xte=xte)
        f.write(plain_coords)

    # Load airfoil to plot
    fb_foil = pd.read_csv('fb bullshit foil.dat', delimiter='\t', header=None)

    # Plot airfoil contour
    plt.figure()

    plt.plot(xx_suc, yy_suc, 'r', xx_pre, yy_pre, 'b', linewidth=2)
    plt.plot(fb_foil[0], fb_foil[1], 'g--')

    plt.grid(True)
    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.gca().axis('equal')

    # Show & save graphs
    plt.show()

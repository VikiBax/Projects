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
    zcoords = np.zeros((npts - 1) + (npts % 2))

    xcoords.round(decimals=9)
    ycoords.round(decimals=9)

    df = pd.DataFrame({'x': xcoords, 'y': ycoords, 'z': zcoords})

    # Return 2D list of coordinates [[x,y],[x,y],...] by transposing .T
    return df


def create_foil(rle, x_pre, y_pre, d2ydx2_pre, x_suc, y_suc, d2ydx2_suc, n=101):
    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    yle = 0.0
    xte = 1.0
    yte = 0.0

    th_pre = np.degrees(np.arctan(y_pre / (1 - x_pre)))
    th_suc = np.degrees(np.arctan(y_suc / (1 - x_suc)))

    # th_pre = -10
    # th_suc = -45
    # th_suc = -52

    # Flip y-coordinates
    y_pre = -y_pre
    y_suc = -y_suc

    # Evaluate pressure (lower) surface coefficients
    cf_pre = pcoef(xte, yte, rle, x_pre, y_pre, d2ydx2_pre, th_pre, 'pre')

    # Evaluate suction (upper) surface coefficients
    cf_suc = pcoef(xte, yte, rle, x_suc, y_suc, d2ydx2_suc, th_suc, 'suc')

    # Evaluate pressure (lower) surface points
    xx_pre = np.linspace(xte, xle, n)
    yy_pre = -(cf_pre[0] * xx_pre ** (1 / 2) +
               cf_pre[1] * xx_pre ** (3 / 2) +
               cf_pre[2] * xx_pre ** (5 / 2) +
               cf_pre[3] * xx_pre ** (7 / 2) +
               cf_pre[4] * xx_pre ** (9 / 2) +
               cf_pre[5] * xx_pre ** (11 / 2))

    # Evaluate suction (upper) surface points
    xx_suc = np.linspace(xle, xte, n)
    yy_suc = -(cf_suc[0] * xx_suc ** (1 / 2) +
               cf_suc[1] * xx_suc ** (3 / 2) +
               cf_suc[2] * xx_suc ** (5 / 2) +
               cf_suc[3] * xx_suc ** (7 / 2) +
               cf_suc[4] * xx_suc ** (9 / 2) +
               cf_suc[5] * xx_suc ** (11 / 2))

    return cf_pre, cf_suc, xx_pre, yy_pre, xx_suc, yy_suc


def check_foil(x_pre, y_pre, x_suc, y_suc, xx_pre, yy_pre, xx_suc, yy_suc):
    x_test = np.linspace(0, 1, 201)
    tol = 0.05
    intersect = False
    thin = False
    pres = False
    suct = False

    for x in x_test:
        p = sorted(zip(xx_pre, yy_pre))
        s = sorted(zip(xx_suc, yy_suc))

        yp = np.interp(x, [xx for xx, yy in p], [yy for xx, yy in p])
        ys = np.interp(x, [xx for xx, yy in s], [yy for xx, yy in s])

        if yp < ys and x != 0 and x != 1:
            intersect = True
            # print(x, 'Self-intersection detected')
        if abs(ys - yp) < 0.02 and 0.15 < x < 0.75:
            thin = True
            # print(x, 'Thin section')
        if yp < y_pre and abs(x - x_pre) > tol:
            pres = True
            # print(x, 'Pressure side downturn')
        if ys < y_suc and abs(x - x_suc) > tol:
            suct = True
            # print(x, 'Suction side downturn')

    return intersect, thin, pres, suct


if __name__ == "__main__":
    """
    runs = pd.read_csv('Run_List.csv', header=None)

    mp_total = 0
    si_mp_list = []
    th_mp_list = []
    pr_mp_list = []
    su_mp_list = []

    fl_total = 0
    si_fl_list = []
    th_fl_list = []
    pr_fl_list = []
    su_fl_list = []

    total = 0

    for i in range(len(runs)):
        cf_p_mp, cf_s_mp, xx_p_mp, yy_p_mp, xx_s_mp, yy_s_mp = create_foil(runs[1][i], runs[2][i], runs[3][i], runs[4][i], runs[5][i], runs[6][i], runs[7][i])
        si_mp, th_mp, pr_mp, su_mp = check_foil(runs[2][i], runs[3][i], runs[5][i], runs[6][i], xx_p_mp, yy_p_mp, xx_s_mp, yy_s_mp)
        cf_p_fl, cf_s_fl, xx_p_fl, yy_p_fl, xx_s_fl, yy_s_fl = create_foil(runs[8][i], runs[9][i], runs[10][i], runs[11][i], runs[12][i], runs[13][i], runs[14][i])
        si_fl, th_fl, pr_fl, su_fl = check_foil(runs[9][i], runs[10][i], runs[12][i], runs[13][i], xx_p_fl, yy_p_fl, xx_s_fl, yy_s_fl)

        if si_mp or th_mp or pr_mp or su_mp:
            mp_total += 1
        if si_mp:
            si_mp_list.append(i)
        if th_mp:
            th_mp_list.append(i)
        if pr_mp:
            pr_mp_list.append(i)
        if su_mp:
            su_mp_list.append(i)

        if si_fl or th_fl or pr_fl or su_fl:
            fl_total += 1
        if si_fl:
            si_fl_list.append(i)
        if th_fl:
            th_fl_list.append(i)
        if pr_fl:
            pr_fl_list.append(i)
        if su_fl:
            su_fl_list.append(i)

        if si_mp or th_mp or pr_mp or su_mp or si_fl or th_fl or pr_fl or su_fl:
            total += 1


    print(mp_total)
    print(len(si_mp_list))
    print(len(th_mp_list))
    print(len(pr_mp_list))
    print(len(su_mp_list))

    print(fl_total)
    print(len(si_fl_list))
    print(len(th_fl_list))
    print(len(pr_fl_list))
    print(len(su_fl_list))

    print(total)

    coords_mp = ppoints(cf_p_mp, cf_s_mp, 201, xte=1)
    coords_mp.to_csv('mainplane_1.csv', header=False, index=False)
    coords_fl = ppoints(cf_p_fl, cf_s_fl, 201, xte=1)
    coords_fl.to_csv('flap_1.csv', header=False, index=False)
    """

    rle = .04
    x_p = 0.5
    y_p = -0.04
    d2ydx2_p = -0.8
    x_s = 0.55
    y_s = -0.18
    d2ydx2_s = -0.5

    # cf_p, cf_s, xx_p, yy_p, xx_s, yy_s = create_foil(.06, 0.5, -0.04, -0.4, 0.4, -0.18, -1.5)
    # cf_p, cf_s, xx_p, yy_p, xx_s, yy_s = create_foil(.04, 0.5, -0.04, -0.8, 0.55, -0.18, -0.5)
    cf_p, cf_s, xx_p, yy_p, xx_s, yy_s = create_foil(rle, x_p, y_p, d2ydx2_p, x_s, y_s, d2ydx2_s)
    it, th, pr, su = check_foil(x_p, y_p, x_s, y_s, xx_p, yy_p, xx_s, yy_s)

    print('Self-intersection: ', it)
    print('Thin section: ', th)
    print('Pressure-side downturn: ', pr)
    print('Suction-side downturn: ', su)

    # cf_p, cf_s, xx_p, yy_p, xx_s, yy_s = create_foil(.025, 0.5, -0.04, -0.8, 0.56, -0.16, -1)

    # coords = ppoints(cf_p, cf_s, 201, xte=1)
    # coords.to_csv('lead2.csv', header=False, index=False)

    # Load airfoil to plot
    fb_foil = pd.read_csv('fb bullshit foil.dat', delimiter='\t', header=None)
    mshd_foil = pd.read_csv('mshd-foil.dat', delimiter='\t', header=None)
    new_flap = pd.read_csv('frankenstein.csv', header=None)

    # Plot airfoil contour
    plt.figure()

    plt.plot(xx_s, yy_s, 'r', xx_p, yy_p, 'b', linewidth=2)
    # plt.plot(fb_foil[0], fb_foil[1], 'g--')
    # plt.plot(mshd_foil[0], -mshd_foil[1], 'k--')
    # plt.plot(new_flap[0], new_flap[1], 'g')

    plt.grid(True)
    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.gca().axis('equal')

    # Show & save graphs
    plt.show()

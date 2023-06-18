from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import matplotlib.patches as mp
from matplotlib.patches import Ellipse
import math
from scipy.stats import multivariate_normal
import time
import matplotlib.lines as mline
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import cv2
from sympy import *

def get_belong(index, distance, mydata):
    # my database
    j = 1
    k = 0
    while j < 5:
        if abs(mydata[j][index] - distance) < abs(mydata[k][index] - distance):
            k = j
        j += 1

    return k


def roate_gmm(my_gmm, index=0, theta=0.0, roate_center=None):

    change_pos = 1
    srcx, srcy = my_gmm.means_[index]
    if roate_center == None:
        roate_center = my_gmm.means_[index]
        change_pos = 0

    U, s, Vt = np.linalg.svd(my_gmm.covariances_[index])
    raw_theta = math.atan(U[0][1] / U[0][0])

    theta0 = theta
    U0 = np.array([[-math.cos(theta0 + raw_theta), -math.sin(theta0 + raw_theta)],
                   [-math.sin(theta0 + raw_theta), math.cos(theta0 + raw_theta)]])
    my_gmm.covariances_[index] = np.dot(U0 * s, U0)

    if change_pos == 1:
        xy = my_gmm.means_[index]
        srcx = (xy[0] - roate_center[0]) * math.cos(theta0) - (xy[1] - roate_center[1]) * math.sin(theta0) + \
               roate_center[0]
        srcy = (xy[1] - roate_center[1]) * math.cos(theta0) + (xy[0] - roate_center[0]) * math.sin(theta0) + \
               roate_center[1]
        my_gmm.means_[index] = [srcx, srcy]
    return np.dot(U0 * s, U0), [srcx, srcy]


def roate_p(points, theta=0.0, roate_center=[0.0, 0.0]):

    srcx = (points[0] - roate_center[0]) * math.cos(theta) - (points[1] - roate_center[1]) * math.sin(theta) + \
           roate_center[0]
    srcy = (points[1] - roate_center[1]) * math.cos(theta) + (points[0] - roate_center[0]) * math.sin(theta) + \
           roate_center[1]
    return [srcx, srcy]


def my_GMR(gmmrs, len_X):

    priors = [0.33, 0.33, 0.33]
    Rmgs = []
    for i in range(gmmrs.n_components):
        Rmgs.append(multivariate_normal(mean=gmmrs.means_[i][0], cov=gmmrs.covariances_[i][0][0]))
    Pxi = []
    for i in range(gmmrs.n_components):
        temp = Rmgs[i].pdf(range(len_X))
        Pxi.append(temp * priors[i])
    Pxi = np.array(Pxi)

    for i in range(len(Pxi[0])):
        temp_sum = Pxi[:, i].sum()
        for j in range(len(Pxi)):
            Pxi[j][i] = Pxi[j][i] / temp_sum

    y_temp = []
    for i in range(gmmrs.n_components):
        ty = np.tile(gmmrs.means_[i][1], len_X) + (gmmrs.covariances_[i][0][1] / gmmrs.covariances_[i][0][0]) * (
                np.arange(len_X) - np.tile(gmmrs.means_[i][0], len_X))
        y_temp.append(ty)
    y_temp = np.array(y_temp)

    y_temp2 = y_temp * Pxi
    y = []
    for i in range(len_X):
        y.append(y_temp2[:, i].sum())

    return y

def grad_gauss():
    t = symbols('t')  # theta
    A, B = symbols('A B')  # A ellipse long axis， B ellipse short axis
    rx, ry, R = symbols('rx ry R')  # rx,ry rotation center， R rotation radius
    p0, p1 = symbols('p0 p1')   # penalty term
    mean = Matrix([rx + R*cos(t), ry + R*sin(t)])
    cov = Matrix([[-cos(t), -sin(t)],[-sin(t), cos(t)]])*Matrix([[A, 0],[0, B]])*Matrix([[-cos(t), -sin(t)],[-sin(t), cos(t)]])  # det(cov) = AB

    A, B, R, rx, ry, p0, p1 = 189.76742131, 8.15751794, 31.684, 0.0, 0.0, 0.0 ,0.0
    result_thetas = [0.0]
    result_loss = [0.0]

    def add_point(a, p):
        temp_b = a + (A * (p[0] * sin(t) - p[1] * cos(t)) ** 2 + B * (
                 p[0] * cos(t) + p[1] * sin(t)) ** 2 + B * R * R - 2 * B * R * (p[0] * cos(t) + sin(t) * p[1])) / (
                 A * B)
        ans_t = solveset(diff(temp_b), t, Interval(0, pi / 2))
        mini = minimum(temp_b , t, Interval(0, pi / 2))
        return temp_b, ans_t, mini

    loss_fun = 0
    XX = [0.0, 1.052675163571365, 2.6420872430712805, 3.5567717104138126, 4.920365840057018, 5.6747797314080834,
         6.371911016327833, 6.963700524864636, 7.020416654301937, 7.875, 10.912435566820085, 13.72028607573472,
         17.688714622606135, 21.15585025471678, 23.03281626288891, 25.27511127571944, 26.378791196717106, 27.344160711201216,
         27.09836249665282, 26.108726989265485, 25.910193457402052, 28.342161614809832, 29.438240436547837,
         31.401632123187483, 34.382735493267546, 36.47229256572721]

    for i, each in enumerate(XX):
        print(i)
        prob, ans_theta, mini_loss =  add_point(loss_fun, [i,each])
        result_thetas.append(ans_theta)
        result_loss.append(mini_loss)

    print(result_thetas, result_loss)






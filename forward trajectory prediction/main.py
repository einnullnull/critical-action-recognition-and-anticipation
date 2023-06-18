from my_utils import *
from sklearn.mixture import GaussianMixture
import numpy as np
import math
from scipy.stats import multivariate_normal
import time


if __name__ == '__main__':
    # ============================================get the APR pattern====================================================
    # Most representative data
    X = [0.0, 0.0 ,0.0, 0.0]
    # GMM initialization
    gmm1 = GaussianMixture(3)
    # data preparation
    X0 = np.arange(len(X)).reshape(len(X), 1)
    X = np.array(X).reshape(len(X), 1) / 10
    XX = np.concatenate([X0, X], axis=1)
    # GMM fitting
    gmm1.fit(XX)

    if True:
        # # GMM kernel sorting
        temp_mean = list(gmm1.means_.copy())
        temp_cov = list(gmm1.covariances_.copy())
        temp_w = list(gmm1.weights_.copy())
        temp_pre = list(gmm1.precisions_.copy())
        temp_pc = list(gmm1.precisions_cholesky_.copy())
        val_t0 = []
        for each in gmm1.means_:
            val_t0.append(each[0])
        val_t = np.array(val_t0)
        val_t2 = val_t0.copy()

        val_t2.sort()

        sort_index = []
        for each in val_t:
            sort_index.append(np.where(val_t2 == each)[0][0])

        for j, i in enumerate(sort_index):
            gmm1.means_[i] = temp_mean[j]
            gmm1.covariances_[i] = temp_cov[j]
            gmm1.weights_[i] = temp_w[j]
            gmm1.precisions_[i] = temp_pre[j]
            gmm1.precisions_cholesky_[i] = temp_pc[j]


    U, s, Vt = np.linalg.svd(gmm1.covariances_[0])
    s[0] = 0.7 * s[0]
    gmm1.covariances_[0] = np.dot(U * s, Vt)
    # test data preparation
    x_test = [0.0, 0.0, 0.0,0.0,0.0]


    f_mean = gmm1.means_.copy()
    f_cov = gmm1.covariances_.copy()

    gmms = []
    for m, c in zip(gmm1.means_, gmm1.covariances_):
        gmms.append(multivariate_normal(mean=m, cov=c))
    # record the initial probability model
    ans = gmm1.predict(XX)
    mini_prob = []
    for i in range(gmm1.n_components):
        #     print(i)
        the_j = np.where(ans == i)[0]
        temp_min = []
        for j in the_j:
            temp_min.append(gmms[i].pdf(XX[j]))

        temp_min = np.array(temp_min)
        mini_prob.append(temp_min.mean() / 5)
    # record the start point
    re_rc = []
    i = 0
    j = 1
    while i < len(ans) and j < gmm1.n_components:
        if ans[i] == j:
            re_rc.append(XX[i])
            j += 1
        i += 1

    dst_gmm = []
    for i in range(gmm1.n_components):
        u0, s, v0 = np.linalg.svd(gmm1.covariances_[i])
        w, h = 5 * np.sqrt(s)
        dst_gmm.append(w)

    # =========================================first kernel initialisation=======================================
    # initial theta
    starts = [0.000000000001, 0.000000000001]
    u0, s, v0 = np.linalg.svd(gmm1.covariances_[0])

    w, h = 2.3 * np.sqrt(s)
    move_angle = (np.arctan(u0[1, 0] / u0[0, 0]))

    move_w = w * math.cos(move_angle)
    move_h = w * math.sin(move_angle)

    # translation
    movedist = [move_w, move_h]

    moves = [movedist[0] - gmm1.means_[0][0], movedist[1] - gmm1.means_[0][1]]
    #         print(moves)

    for ci in range(gmm1.n_components):
        gmm1.means_[ci] = [gmm1.means_[ci][0] + moves[0], gmm1.means_[ci][1] + moves[1]]
    for ci in range(gmm1.n_components - 1):
        re_rc[ci] = [re_rc[ci][0] + moves[0], re_rc[ci][1] + moves[1]]

    for rc_i in range(gmm1.n_components - 1):
        re_rc[rc_i] = roate_p(re_rc[rc_i], -move_angle, starts)

    for g in range(gmm1.n_components):
        roate_gmm(gmm1, index=g, theta=-move_angle, roate_center=starts)

    for g in range(1, gmm1.n_components):
        roate_gmm(gmm1, index=g, theta=move_angle, roate_center=re_rc[0])
    yy = my_GMR(gmm1, len(x_test))

    gmms = []
    for m, c in zip(gmm1.means_, gmm1.covariances_):
        gmms.append(multivariate_normal(mean=m, cov=c))

    raw_mean = gmm1.means_.copy()
    raw_cov = gmm1.covariances_.copy()
    raw_re_rc = re_rc.copy()
    raw_minpb = mini_prob.copy()
    raw_gmms = gmms.copy()
    # ============================================rotation===================================================
    all_dis = ['multiple trajectories to predict']
    my_t = []
    pre_dstes = []
    sp_mean = []
    sp_cov = []
    sp_w = []
    sp_pre = []
    sp_pc = []
    change_ps = []
    for dst_ii, row in enumerate(all_dis):
        change_p = [0]
        # kernel location initialization=======================================================================
        for copy_i, mm, ccov, mmpb in zip(range(3), raw_mean, raw_cov, raw_minpb):
            gmm1.means_[copy_i] = mm
            gmm1.covariances_[copy_i] = ccov
            mini_prob[copy_i] = mmpb
        for copy_i, rerc in enumerate(raw_re_rc):
            re_rc[copy_i] = rerc
        # reset Gaussian parameters
        gmms = []
        for m, c in zip(gmm1.means_, gmm1.covariances_):
            gmms.append(multivariate_normal(mean=m, cov=c))
        # Iteration Start========================================================================================

        x_test = row

        fig_i = dst_ii + 1
        # prior theta initialization
        last_theta = 100
        ssss = time.time()  # compute iteration time
        predst = []
        i = 0
        starts = [0.000000000001, 0.000000000001]
        gmm_i = 0
        change_gauss = 0
        # Criteria for parameter updates
        x_test = np.array(x_test) / 10
        change_a = 0
        three_q = []
        three_qp = []
        while i < len(x_test):
            each = x_test[i]
            three_q.append([i, each])
            t_prob = gmms[gmm_i].pdf([i, each])
            three_qp.append(t_prob)
            # check the kernel ID
            if (len(three_qp) == 3) or (i == len(x_test) - 1):

                if len(three_qp) == 1:
                    three_qp = [three_qp[0], three_qp[0], three_qp[0]]
                    three_q = [three_q[0], three_q[0], three_q[0]]

                three_q = np.array(three_q)
                three_qp = np.array(three_qp)
                dst = math.sqrt((np.mean(three_q[:, 0]) - starts[0]) ** 2 + (np.mean(three_q[:, 1]) - starts[1]) ** 2)
                if (change_gauss >= 5 and three_qp[1] < mini_prob[gmm_i] and gmm_i != 2) or (dst >= dst_gmm[gmm_i]):
                    # check current kernel
                    change_a = 1
                    change_p.append(i)
                    change_gauss = 0
                    # enter next kernel
                    gmm_i += 1
                    if gmm_i >= gmm1.n_components:
                        break
                    # translation decision
                    starts = [three_q[1, 0], three_q[1, 1]]

                    re_move = [starts[0] - re_rc[gmm_i - 1][0], starts[1] - re_rc[gmm_i - 1][1]]
                    # Iterative update SP
                    for re_i in range(gmm_i - 1, gmm1.n_components - 1):
                        re_rc[re_i] = [re_rc[re_i][0] + re_move[0], re_rc[re_i][1] + re_move[1]]
                    u0, s, v0 = np.linalg.svd(gmm1.covariances_[gmm_i])
                    w, h = 2.0 * np.sqrt(s)
                    move_angle = (np.arctan(u0[1, 0] / u0[0, 0]))

                    move_w = w * math.cos(move_angle)
                    move_h = w * math.sin(move_angle)

                    movedist = [starts[0] + move_w, move_h + starts[1]]

                    moves = [movedist[0] - gmm1.means_[gmm_i][0], movedist[1] - gmm1.means_[gmm_i][1]]

                    for ci in range(gmm_i, gmm1.n_components):
                        gmm1.means_[ci] = [gmm1.means_[ci][0] + moves[0], gmm1.means_[ci][1] + moves[1]]


                    a_theta = math.atan((gmm1.means_[gmm_i][1] - starts[1]) / (
                                gmm1.means_[gmm_i][0] + 0.00000001 - starts[0]))  # atan2(y,x)

                    b_theta = math.atan(
                        (three_q[-1, 1] - three_q[0, 1]) / (three_q[-1, 0] + 0.00000001 - three_q[0, 0]))

                    ans_theta = (b_theta - a_theta)

                    for rc_i in range(gmm_i, gmm1.n_components - 1):
                        re_rc[rc_i] = roate_p(re_rc[rc_i], ans_theta, starts)

                    for g in range(gmm_i, gmm1.n_components):
                        roate_gmm(gmm1, index=g, theta=ans_theta, roate_center=starts)

                    for g in range(gmm_i + 1, gmm1.n_components):
                        roate_gmm(gmm1, index=g, theta=-ans_theta, roate_center=re_rc[gmm_i])

                    gmms = []
                    for m, c in zip(gmm1.means_, gmm1.covariances_):
                        gmms.append(multivariate_normal(mean=m, cov=c))

                    mini_prob[gmm_i] = gmms[gmm_i].pdf(starts) / 2

                elif (change_gauss < 5 and three_qp[1] < mini_prob[gmm_i]) or change_a == 1 or gmm_i == 2:
                    if change_a == 0:
                        change_gauss = 0
                    change_a = 0

                    a_theta = math.atan((gmm1.means_[gmm_i][1] - starts[1]) / (
                                gmm1.means_[gmm_i][0] + 0.00000001 - starts[0]))  # atan2(y,x)
                    b_theta = math.atan(
                        (np.mean(three_q[:, 1]) - starts[1]) / (np.mean(three_q[:, 0]) + 0.00000001 - starts[0]))

                    ans_theta = (b_theta - a_theta)

                    for rc_i in range(gmm_i, gmm1.n_components - 1):
                        re_rc[rc_i] = roate_p(re_rc[rc_i], ans_theta, starts)

                    for g in range(gmm_i, gmm1.n_components):
                        roate_gmm(gmm1, index=g, theta=ans_theta, roate_center=starts)

                    #                 print(gmm_i)
                    for g in range(gmm_i + 1, gmm1.n_components):
                        roate_gmm(gmm1, index=g, theta=-ans_theta, roate_center=re_rc[gmm_i])

                    gmms = []
                    for m, c in zip(gmm1.means_, gmm1.covariances_):
                        gmms.append(multivariate_normal(mean=m, cov=c))

                elif three_qp[1] > mini_prob[gmm_i]:

                    roate_alpha = 0.2
                    change_gauss += 3

                    a_theta = math.atan((gmm1.means_[gmm_i][1] - starts[1]) / (
                                gmm1.means_[gmm_i][0] + 0.00000001 - starts[0]))  # atan2(y,x)
                    b_theta = math.atan(
                        (np.mean(three_q[:, 1]) - starts[1]) / (np.mean(three_q[:, 0]) + 0.00000001 - starts[0]))

                    if abs(last_theta - b_theta) < 0.2:
                        roate_alpha = 0.8

                    ans_theta = (b_theta - a_theta) * roate_alpha

                    last_theta = b_theta
                    for rc_i in range(gmm_i, gmm1.n_components - 1):
                        re_rc[rc_i] = roate_p(re_rc[rc_i], ans_theta, starts)
                    for g in range(gmm_i, gmm1.n_components):
                        roate_gmm(gmm1, index=g, theta=ans_theta, roate_center=starts)
                    for g in range(gmm_i + 1, gmm1.n_components):
                        roate_gmm(gmm1, index=g, theta=-ans_theta, roate_center=re_rc[gmm_i])
                    gmms = []
                    for m, c in zip(gmm1.means_, gmm1.covariances_):
                        gmms.append(multivariate_normal(mean=m, cov=c))

                three_q = []
                three_qp = []
            i += 1

        my_t.append(time.time() - ssss)
        pre_dstes.append(predst)
        change_ps.append(change_p)

    print(pre_dstes)


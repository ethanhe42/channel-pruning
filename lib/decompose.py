from __future__ import print_function
from .cfgs import c as dcfgs
from . import cfgs
import os
os.environ['JOBLIB_TEMP_FOLDER']=dcfgs.shm
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso,LinearRegression, MultiTaskLasso, RandomizedLasso, Ridge
import numpy as np
from IPython import embed
from scipy import optimize
import scipy
try:
    import lightning
    from lightning.regression import CDRegressor, SGDRegressor
except:
    print("no lighting pack")
from .worker import Worker
from .utils import beep, CHECK_EQ, Timer, redprint
if not cfgs.noTheano:
    from .theanols import tls

def relu(x):
    return np.maximum(x, 0.)

def error(A, B):
    return np.mean((A - B)**2)**.5

def ac_error(A, B):
    return np.sum(np.abs(A - B))

def rel_error(A, B):
    return np.mean((A - B)**2)**.5 / np.mean(A**2)**.5

def pca(data, n_components=None):
    """
    Param:
        n_components: use 'mle' to guess
    """
    newdata = data.copy()
    model = PCA(n_components=n_components)

    if len(newdata.shape) != 2:
        newdata = newdata.reshape((newdata.shape[0], -1))

    model.fit(newdata)

    ret = model.explained_variance_ratio_

    return ret

def solve_relu(RU, Z, Lambda):
    # case 0: U <= 0
    U0 = np.minimum(RU, 0.)
    Cost0 = Z**2 + Lambda * (U0 - RU)**2
    # case 1: U > 0
    U1 = relu((Lambda * RU +Z) / (Lambda + 1.))
    Cost1 = (U1 - Z)**2 + Lambda * (U1 - RU)**2
    U = (Cost0 <= Cost1) * U0 + (Cost0 > Cost1) * U1
    return U

def YYT(Y, n_components=None, DEBUG=False):
    """
    Param:
        Y: n x d
        n_components: use 'mle' to guess
    Returns:
        P: d x d'
        QT: d' x d
    """
    newdata = Y.copy()
    model = PCA(n_components=n_components)

    if len(newdata.shape) != 2:
        newdata = newdata.reshape((newdata.shape[0], -1))
    #TODO center data
    model.fit(newdata)
    if DEBUG: from IPython import embed; embed()

    return model.components_.T, model.components_

#def GSVD(Z, Y):
#    NotImplementedError
#    return [U,V,X,C,S]

def VH_decompose(weights, rank=None, DEBUG=0, X=None, Y=None):
    """
    Param:
        weights: n c h w
        rank: keep
    Returns:
        V: rank c 1 h
        H:  n rank w 1
    """
    # TODO use linear method to do VH_decompose
    dim = weights.shape
    # c h n w
    VH = np.transpose(weights, [1, 2, 0, 3])
    # ch x nw
    VH = VH.reshape([dim[1]*dim[2], dim[0] * dim[3]])
    # ch x ch, ch, ch x nw
    V, sigmaVH, H = svd(VH)
    #V, sigmaVH, H = np.linalg.svd(VH, full_matrices=0)
    if rank is None:
        rank = dim[1] * dim[2]
    # ch x rank
    V = V[:, :rank]
    # rank x nw
    H = H[:rank, :]
    # rank,
    sigmaVH = sigmaVH[:rank]
    # rank x nw
    H = np.diag(sigmaVH).dot(H)

    # recover error
    # ch x nw -> c h n w
    VHr = (V.dot(H)).reshape([dim[1], dim[2], dim[0], dim[3]])
    if 0: #DEBUG
        print('ABS ErrVH', np.mean(np.abs(VHr.flatten()-VH.flatten())))
        print('REL ABS ErrVH', np.mean(np.abs(VHr.flatten()-VH.flatten())/ np.abs(VH.flatten())))

    # rank nw -> rank n w 1
    H = H.reshape([rank, dim[0], dim[3], 1])
    # rank n w 1 -> n rank 1 w
    H = np.transpose(H, [1, 0, 3, 2])
    # ch rank -> c 1 h rank  -> rank c h 1
    origV = V.copy()
    V = V.reshape((dim[1], 1, dim[2], rank))
    V = np.transpose(V, [3, 0, 2, 1])

    if X is not None:
        Xv = np.tensordot(X,V, [[1,2],[1,2]])
        Xv = np.transpose(Xv,[0, 2, 3, 1])
        N = Xv.shape[0]
        o = H.shape[0]
        H, b = nonlinear_fc(Xv.reshape([N, -1]), Y)
        H = H.reshape([o, rank, 1, 3])
        reH = np.transpose(H, [1,0,2,3]).reshape([rank,-1])
        VHr = (origV.dot(reH)).reshape([dim[1], dim[2], dim[0], dim[3]])

    VHr = np.transpose(VHr, [2, 0, 1, 3])
    if 1:
        epscheck(V, 2)
        epscheck(H, 2)
        epscheck(VHr, 2)
    if X is not None:
        return V, H, VHr, b
    return V, H, VHr

def pinv(x):
    #return np.linalg.pinv(x, max(x.shape) * np.spacing(np.linalg.norm(x)))
    #return scipy.linalg.pinv(x, max(x.shape) * np.spacing(np.linalg.norm(x)))
    return scipy.linalg.pinv(x, 1e-6)

def svd(x):
    return scipy.linalg.svd(x, full_matrices=False, lapack_driver='gesvd')
    #return scipy.linalg.svd(x, full_matrices=False)

def epscheck(x, tol=5):
    tmp = np.any(np.abs(x) > 10**tol)
    if tmp:
        redprint('1e'+str(tol)+' exceed')

def ITQ_decompose(feature, gt_feature, weight, rank, bias=None, DEBUG=False, Wr=None):

    n_ins = feature.shape[0]
    n_filter_channels = feature.shape[1]
    assert gt_feature.shape[0] == n_ins
    assert gt_feature.shape[1] == n_filter_channels

    # do itq
    if 0:
        Y_div = n_ins * feature.std() # * n_filter_channels
        Y = feature.copy() / Y_div
        # r(yi)
        Z = relu(gt_feature) / Y_div
    else:
        Y = feature.copy()
        # r(yi)
        Z = relu(gt_feature)
    Zsq = Z**2
    Y_mean = Y.mean(0)
    # Y
    G = Y - Y_mean
    # (Y'Y)^-1
    PG = (G.T).dot(G)
    if 0:
        print(np.linalg.cond(PG))
        embed()
    PG = pinv(PG)
    #epscheck(PG)
    #epscheck(PG,6)
    #epscheck(PG,7)
    #epscheck(PG,8)
    #epscheck(PG,9)
    #epscheck(PG,10)

    PGGt = PG.dot(G.T)

    # init U as Y
    UU = G.copy()
    U_mean = Y_mean.copy()
    if 1: print("Reconstruction Err", rel_error(Z, relu(Y)))

    lambdas = [0.1, 1]
    step_iters = [30, 20]# , 20, 20, 20
    for step in range(len(lambdas)):
        Lambda = lambdas[step]
        for iter in range(step_iters[step]):

            #  TODO    Y * (Y'Y)^-1 * (Y'*Z)
            #X = G.dot(Ax_b(PG, (G.T).dot(UU)))
            #X = G.dot(Ax_b(G, UU))
            if 0: epscheck(X,10)
            X = G.dot(PGGt.dot(UU))
            #X = G.dot(PG.dot((G.T).dot(UU)))

            L, sigma, R = svd(X)
            #L, sigma, R = np.linalg.svd(X, 0)

            T = L[:, :rank].dot(np.diag(sigma[:rank])).dot(R[:rank, :])
            if 0: print("RX", error(X, T))

            #T = Ax_b(PG, (G.T).dot(T))
            #T = Ax_b(G, T)
            if 0: epscheck(T,10)
            T = PGGt.dot(T)
            #T = PG.dot((G.T).dot(T))
            RU = G.dot(T)
            if 0: print("RU", rel_error(UU, RU))

            RU += U_mean
            # case 0: U <= 0
            U0 = np.minimum(RU, 0.)
            Cost0 = Zsq + Lambda * (U0 - RU)**2

            # case 1: U > 0
            U1 = relu((Lambda * RU +Z) / (Lambda + 1.))
            Cost1 = (U1 - Z)**2 + Lambda * (U1 - RU)**2

            U = (Cost0 <= Cost1) * U0 + (Cost0 > Cost1) * U1

            U_mean = U.mean(0)
            # Z
            UU = U - U_mean
            if DEBUG: 
                loss = error(Z, relu(U))
                print("loss", loss, "rel", rel_error(Z, relu(U)))

    # process output
    L, sigma, R = svd(T)
    #L, sigma, R = np.linalg.svd(T,0)
    L = L[:, :rank]
    R = np.diag(sigma[:rank]).dot(R[:rank, :])

    dim = weight.shape
    W12 = np.ndarray(Wr.shape if Wr is not None else weight.shape, weight.dtype)
    assert len(dim) == 4

    right = 1

    if dim[3] != n_filter_channels:
        assert dim[0] == n_filter_channels
        if right:
            weight = np.transpose(weight, [1, 2, 3, 0])
            W1 = weight.reshape([-1, n_filter_channels]).dot(L)
        else:
            W1 = R.dot(weight.reshape([n_filter_channels, -1]))

        if Wr is not None:
            Wr = np.transpose(Wr, [1, 2, 3, 0])
            W12 = Wr.reshape([-1, n_filter_channels]).dot(L)
        else:
            W12 = W1

        if 0:
            # embed()
            print(W1.shape)
            print(weight.shape)
            print(dim)
        if right:
            W1 = W1.reshape(weight.shape[:3]+(rank,))
            W1 = np.transpose(W1, [3, 0, 1, 2])
        else:
            W1 = W1.reshape((rank,) + W1.shape[1:])
    else:
        assert False
        W1 = weight.reshape([-1, n_filter_channels]).dot(L)
        W1 = W1.reshape([dim[0], dim[1], dim[2], rank])

    W2 = R
    if right:
        W12 = W12.dot(W2)
    else:
        W12 = W2.T.dot(W12)
    W2 = W2.T
    W2 = W2.reshape([n_filter_channels, rank, 1, 1])

    if right:
        W12 = W12.reshape((Wr.shape[:3] if Wr is not None else weight.shape[:3])+ (n_filter_channels,))
        W12 = np.transpose(W12, [3, 0, 1, 2])
    else:
        W12 = W12.reshape(dim)

    B = - Y_mean.dot(T) + U_mean
    if bias is not None:
        B = B.T + bias
    else:
        B = B.T

    if 1: 
        epscheck(W1 ,  2)
        epscheck(W2 ,  2)
        epscheck(B  ,   2)
        epscheck(W12, 2)
        epscheck(W1 , 4)
        epscheck(W2 , 4)
        epscheck(B  ,  4)
        epscheck(W12,4)
    return W1, W2, B, W12

def get_cost(weight_dim, feature_dim, rank, method):
    feature_pixels = feature_dim[2] * feature_dim[3]
    ret = weight_dim[0] * weight_dim[1] * weight_dim[2] * feature_dim[2] * feature_pixels
    NotImplementedError
    return ret

def Ax_b(A, b, C=0, Lambda=1, DEBUG=0):
    """
    Ax = b
    """
    if DEBUG:
        resA = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(b))
        resB = np.linalg.lstsq(A, b)[0]
        print(resA.shape)
        print(resB.shape)
        if not CHECK_EQ(resA, resB):
            embed()

        ATA = A.T.dot(A)
        try:
            np.linalg.lstsq(ATA + 0.1*np.eye(len(ATA)), A.T.dot(b))[0]
        except:
            embed()
    if C != 0:
        # TODO remove this
        nSamples = 1.
        # nSamples = len(A)**.5

        ATA = (A/nSamples).T.dot(A/nSamples)
        return np.linalg.lstsq(ATA + (C / Lambda / nSamples**2)*np.eye(len(ATA)), (A/nSamples).T.dot(b/nSamples))[0]
    else:
        return np.linalg.lstsq(A, b)[0]

def xA_b(A, b, **kwargs):
    """
    xA=b
    """
    return Ax_b(A.T, b.T, **kwargs).T
    # return b.dot(A.T).dot(np.linalg.pinv(A.dot(A.T)))

def nnls(A, B):
    def func(b):
        return optimize.nnls(A, b)[0]
    return np.array(map(func, B))

def ax2bxc(a,b,c):
    def getDiscriminant(a, b, c,Error = "Imaginary Roots"):
        try:
            return (b ** 2 - (4 * a * c)) ** .5
        except:
            return Error
    D = getDiscriminant(a, b, c) 
    if D == False:
        return False
    b = -b
    a = 2 * a
    firstroot = float((b + D) / float(a))
    secondroot = float((b - D) / float(a))
    assert firstroot * secondroot <= 0
    if firstroot > 0:
        return firstroot
    if secondroot > 0:
        return secondroot
    return False

def dictionary(X, W2, Y,alpha=1e-4, rank=None, DEBUG=0, B2=None, rank_tol=.1, verbose=0):
    verbose=0
    if verbose:
        timer = Timer()
        timer.tic()
    if 0 and rank_tol != dcfgs.dic.rank_tol:
        print("rank_tol", dcfgs.dic.rank_tol)
    rank_tol = dcfgs.dic.rank_tol
    # X: N c h w,  W2: n c h w
    norank=dcfgs.autodet
    if norank:
        rank = None
    #TODO remove this
    N = X.shape[0]
    c = X.shape[1]
    h = X.shape[2]
    w=h
    n = W2.shape[0]
    # TODO I forgot this
    # TODO support grp lasso
    if h == 1 and False:
        for i in range(2):
            assert Y.shape[i] == X.shape[i]
            pass
        grp_lasso = True
        mtl = 1
    else:
        grp_lasso = False
    if norank:
        alpha = cfgs.alpha / c**dcfgs.dic.layeralpha

    if grp_lasso:
        reX = X.reshape((N, -1))
        ally = Y.reshape((N,-1))
        samples = np.random.choice(N, N//10, replace=False)
        Z = reX[samples].copy()
        reY = ally[samples].copy()

    else:
        samples = np.random.randint(0,N, min(400, N//20))
        #samples = np.random.randint(0,N, min(400, N//20))
        # c N hw
        reX = np.rollaxis(X.reshape((N, c, -1))[samples], 1, 0)
        #c hw n
        reW2 = np.transpose(W2.reshape((n, c, -1)), [1,2,0])
        if dcfgs.dic.alter:
            W2_std = np.linalg.norm(reW2.reshape(c, -1), axis=1)
        # c Nn
        Z = np.matmul(reX, reW2).reshape((c, -1)).T

        # Nn
        reY = Y[samples].reshape(-1)

    if grp_lasso:
        if mtl:
            print("solver: group lasso")
            _solver = MultiTaskLasso(alpha=alpha, selection='random', tol=1e-1)
        else:
            _solver = Lasso(alpha=alpha,selection='random' )

    elif dcfgs.solver == cfgs.solvers.lightning:
        _solver=CDRegressor(C=1/reY.shape[0]/2, alpha=alpha,penalty='l1', n_jobs=10)
    else:
        _solver = Lasso(alpha=alpha, warm_start=True,selection='random' )
        #, copy_X=False
    #rlasso = RandomizedLasso(n_jobs=1)
    #embed()
    def solve(alpha):
        if dcfgs.dic.debug:
            return np.array(c*[True]), c
        _solver.alpha=alpha
        _solver.fit(Z, reY)
        #_solver.fit(Z, reY)
        if grp_lasso and mtl:
            idxs = _solver.coef_[0] != 0.
        else:
            idxs = _solver.coef_ != 0.
            if dcfgs.solver == cfgs.solvers.lightning:
                idxs=idxs[0]
        tmp = sum(idxs)
        return idxs, tmp

    def updateW2(idxs):
        nonlocal Z
        tmp_r = sum(idxs)
        reW2, _ = fc_kernel((X[:,idxs, :,:]).reshape(N, tmp_r*h*w), Y)
        reW2 = reW2.T.reshape(tmp_r, h*w, n)
        nowstd=np.linalg.norm(reW2.reshape(tmp_r, -1), axis=1)
        #for i in range(len(nowstd)):
        #    if nowstd[i] == 0:
        #        nowstd[i] = W2_std[i]
        reW2 = (W2_std[idxs] / nowstd)[:,np.newaxis,np.newaxis] * reW2
        newshape = list(reW2.shape)
        newshape[0] = c
        newreW2 = np.zeros(newshape, dtype=reW2.dtype)
        newreW2[idxs, ...] = reW2
        Z = np.matmul(reX, newreW2).reshape((c, -1)).T
        if 0:
            print(_solver.coef_)
        return reW2

    if rank == c:
        idxs = np.array([True] * rank)
    elif not norank:
        left=0
        right=cfgs.alpha
        lbound = rank# - rank_tol * c
        if rank_tol>=1:
            rbound = rank + rank_tol
        else:
            rbound = rank + rank_tol * rank
            #rbound = rank + rank_tol * c
            if rank_tol == .2:
                print("TODO: remove this")
                lbound = rank + 0.1 * rank
                rbound = rank + 0.2 * rank
        while True:
            _, tmp = solve(right)
            if False and dcfgs.dic.alter:
                if tmp > rank:
                    break
                else:
                    right/=2
                    if verbose:print("relax right to",right)
            else:
                if tmp < rank:
                    break
                else:
                    right*=2
                    if verbose:print("relax right to",right)
        while True:
            alpha = (left+right) / 2
            idxs, tmp = solve(alpha)
            if verbose:print(tmp, alpha, left, right)
            if tmp > rbound:
                left=alpha
            elif tmp < lbound:
                right=alpha
            else:
                break
        if dcfgs.dic.alter:
            if rbound == lbound:
                rbound +=1
            orig_step = left/100 + 0.1 # right / 10
            step = orig_step

            def waitstable(a):
                tmp = -1
                cnt = 0
                for i in range(10):
                    tmp_rank = tmp
                    idxs, tmp = solve(a)
                    if tmp == 0:
                        break
                    updateW2(idxs)
                    if tmp_rank == tmp:
                        cnt+=1
                    else:
                        cnt=0
                    if cnt == 2:
                        break
                    if 1: 
                        if verbose:print(tmp, "Z", Z.mean(), "c", _solver.coef_.mean())
                return idxs, tmp

            previous_Z = Z.copy()
            state = 0
            statecnt = 0
            inc = 10
            while True:
                Z = previous_Z.copy()
                idxs, tmp = waitstable(alpha)
                if tmp > rbound:
                    if state == 1:
                        state = 0
                        step/=2
                        statecnt=0
                    else:
                        statecnt+=1
                    if statecnt >=2:
                        step*=inc
                    alpha += step
                elif tmp < lbound:
                    if state == 0:
                        state = 1
                        step /= 2
                        statecnt=0
                    else:
                        statecnt+=1
                    if statecnt >=2:
                        step*=inc
                    alpha -= step
                else:
                    break
                if verbose:print(tmp, alpha, 'step', step)
        rank=tmp
    else:
        print("start lasso kernel")
        idxs, rank = solve(alpha)
        print("end lasso kernel")


    # print(rank, _solver.coef_)

    #reg.fit(Z[:, idxs], reY)
    #dic = reg.coef_[np.newaxis, :, np.newaxis, np.newaxis]
    #newW2 = W2[:, idxs, ...]*dic
    if verbose:
        timer.toc(show='lasso')
        timer.tic()
    if grp_lasso:
        inW, inB = fc_kernel(reX[:, idxs], ally, copy_X=True)
        def preconv(a, b, res, org_res):
            '''
            a: c c'
            b: n c h w
            res: c
            '''
            w = np.tensordot(a, b, [[0], [1]])
            r = np.tensordot(res, b, [[0], [1]]).sum((1,2)) + org_res
            return np.rollaxis(w, 1, 0), r
        newW2, newB2 = preconv(inW, W2, inB, B2)
    elif dcfgs.ls == cfgs.solvers.lowparams:
        reg = LinearRegression(copy_X=True, n_jobs=-1)
        assert dcfgs.fc_ridge == 0
        assert dcfgs.dic.alter == 0, "Z changed"
        reg.fit(Z[:, idxs], reY)
        newW2 = reg.coef_[np.newaxis,:,np.newaxis,np.newaxis] * W2[:, idxs, :,:]
        newB2 = reg.intercept_
    elif dcfgs.nonlinear_fc:
        newW2, newB2 = nonlinear_fc(X[:,idxs,...].reshape((N,-1)), Y)
        newW2 = newW2.reshape((n,rank, h, w))
    elif dcfgs.nofc:
        newW2 = W2[:, idxs, :,:]
        newB2 = np.zeros(n)
    else:
        newW2, newB2 = fc_kernel(X[:,idxs,...].reshape((N,-1)), Y, W=W2[:, idxs,...].reshape(n,-1), B=B2)
        newW2 = newW2.reshape((n,rank, h, w))
    if verbose:
        timer.toc(show='ls')
    if not norank:
        cfgs.alpha = alpha
    if verbose:print(rank)
    if DEBUG:
        #print(np.where(idxs))
        newX = X[:, idxs, ...]
        return newX, newW2, newB2
    else:
        return idxs, newW2, newB2

def fc_kernel(X, Y, copy_X=True, W=None, B=None, ret_reg=False,fit_intercept=True):
    """
    return: n c
    """
    assert copy_X == True
    assert len(X.shape) == 2
    if dcfgs.ls == cfgs.solvers.gd:
        w = Worker()
        def wo():
            from .GDsolver import fc_GD
            a,b=fc_GD(X,Y, W, B, n_iters=1)
            return {'a':a, 'b':b}
        outputs = w.do(wo)
        return outputs['a'], outputs['b']
    elif dcfgs.ls == cfgs.solvers.tls:
        return tls(X,Y, debug=True)
    elif dcfgs.ls == cfgs.solvers.keras:
        _reg=keras_kernel()
        _reg.fit(X, Y, W, B)
        return _reg.coef_, _reg.intercept_
    elif dcfgs.ls == cfgs.solvers.lightning:
        #_reg = SGDRegressor(eta0=1e-8, intercept_decay=0, alpha=0, verbose=2)
        _reg = CDRegressor(n_jobs=-1,alpha=0, verbose=2)
        if 0:
            _reg.intercept_=B
            _reg.coef_=W
    elif dcfgs.fc_ridge > 0:
        _reg = Ridge(alpha=dcfgs.fc_ridge)
    else:
        _reg = LinearRegression(n_jobs=-1 , copy_X=copy_X, fit_intercept=fit_intercept)
    _reg.fit(X, Y)
    if ret_reg:
        return _reg
    return _reg.coef_, _reg.intercept_

def nonlinear_fc(X, Y, copy_X=True, W=None, B=None):
    assert len(X.shape)== 2
    assert copy_X == True
    assert W is None
    assert B is None
    U = Y.copy()
    Z = relu(Y)
    its = [30,20]
    for epoch,l in enumerate([10**i for i in range(-1, 1)]): # , np.arange(astart, aend, astep)
        for _ in range(its[epoch]):
            reg = fc_kernel(X, U, copy_X=True, ret_reg=True)
            RU = reg.predict(X)
            if 0: print("l", l, rel_error(Z, relu(RU)))
            U = solve_relu(RU, Z, l)
    return reg.coef_, reg.intercept_


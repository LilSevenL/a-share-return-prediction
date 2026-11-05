"""Independent numerical tests; synthetic fixtures only, no market training."""
import unittest
import numpy as np
from scipy.optimize import check_grad,linprog,minimize,LinearConstraint
from scipy.special import huber
from threadpoolctl import threadpool_limits
from linear_losses import (LossConfig,elementwise_loss,residual_gradient,target_scale,
                           objective_gradient,normalize_weights,fit_linear_loss,predict_linear_loss)


class LossTests(unittest.TestCase):
    def test_huber_values_match_scipy_and_boundary(self):
        r=np.array([-10,-1.35,-.1,0,.1,1.35,10])
        np.testing.assert_allclose(elementwise_loss(r,'huber',1.35),huber(1.35,r))
        np.testing.assert_allclose(residual_gradient(r,'huber',1.35),np.clip(r,-1.35,1.35))
        self.assertLess(elementwise_loss(10,'huber'),elementwise_loss(10,'mse'))

    def test_gradients_and_intercept_not_regularized(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(80,5)); y=rng.normal(size=80); theta=rng.normal(size=6)
        w=normalize_weights(rng.uniform(.1,3,80),80)
        for loss in ('mse','huber','mae'):
            cfg=LossConfig(loss=loss,l2=.2)
            error=check_grad(lambda t:objective_gradient(t,x,y,w,cfg)[0],
                             lambda t:objective_gradient(t,x,y,w,cfg)[1],theta)
            self.assertLess(error,2e-6)
        a=objective_gradient(theta,x,y,w,LossConfig(l2=.2))[1]
        b=objective_gradient(theta,x,y,w,LossConfig(l2=0))[1]
        self.assertAlmostEqual(a[0],b[0])
        np.testing.assert_allclose((a-b)[1:],.2*theta[1:])

    def test_mse_matches_weighted_augmented_lstsq(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(120,4))+3; y=rng.normal(size=120); w=rng.uniform(.2,2,120)
        model=fit_linear_loss(x,y,w,LossConfig(loss='mse',l2=0))
        a=np.column_stack([np.ones(len(y)),x]); root=np.sqrt(w)
        expected=np.linalg.lstsq(a*root[:,None],y*root,rcond=None)[0]
        np.testing.assert_allclose(np.r_[model['intercept'],model['coefficient']],expected,atol=1e-9)

    def test_exact_mae_against_independent_linear_program(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(55,3)); y=.4+x@np.array([.2,-.3,.8])+rng.normal(size=55)*.1
        y[0]+=10; w=rng.uniform(.5,2,55)
        model=fit_linear_loss(x,y,w,LossConfig(loss='mae',l2=0,max_iter=15000,tol=1e-7))
        a=np.column_stack([np.ones(len(y)),x]); n,p=a.shape
        # min sum(w*u), subject to |A theta-y| <= u.
        result=linprog(np.r_[np.zeros(p),w/w.sum()],
            A_ub=np.vstack([np.column_stack([a,-np.eye(n)]),np.column_stack([-a,-np.eye(n)])]),
            b_ub=np.r_[y,-y],bounds=[(None,None)]*p+[(0,None)]*n,method='highs')
        self.assertTrue(result.success)
        actual=np.average(np.abs(predict_linear_loss(model,x)-y),weights=w)
        self.assertAlmostEqual(actual,result.fun,places=5)

    def test_units_and_weight_rescaling_invariance(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(90,3)); y=.01+x@np.array([.02,-.03,.04])+rng.normal(size=90)*.01
        w=rng.uniform(.5,2,90)
        for loss in ('mse','huber','mae'):
            cfg=LossConfig(loss=loss,l2=.1,max_iter=10000,tol=1e-6)
            a=fit_linear_loss(x,y,w,cfg); b=fit_linear_loss(x,100*y,7*w,cfg)
            np.testing.assert_allclose(predict_linear_loss(b,x)/100,predict_linear_loss(a,x),atol=2e-6)
            self.assertAlmostEqual(b['target_scale']/a['target_scale'],100)

    def test_mae_l2_against_independent_epigraph_qp(self):
        rng=np.random.default_rng(81)
        x=rng.normal(size=(35,2)); y=x@np.array([.8,-.3])+rng.normal(size=35)*.1
        w=normalize_weights(rng.uniform(.2,3,35),35)
        cfg=LossConfig(loss='mae',l2=.2,max_iter=15000,tol=1e-7)
        model=fit_linear_loss(x,y,w,cfg)
        scale,_=target_scale(y); ys=y/scale
        a=np.column_stack([np.ones(len(y)),x]); n,p=a.shape
        def fun(z):
            value=np.dot(w,z[p:])/n+.5*cfg.l2*np.dot(z[1:p],z[1:p])
            gradient=np.r_[0,cfg.l2*z[1:p],w/n]
            return value,gradient
        c=np.vstack([np.column_stack([a,-np.eye(n)]),np.column_stack([-a,-np.eye(n)])])
        result=minimize(fun,np.r_[np.zeros(p),np.abs(ys)+.1],jac=True,method='SLSQP',
            constraints=LinearConstraint(c,-np.inf,np.r_[ys,-ys]),
            options=dict(ftol=1e-11,maxiter=1000))
        self.assertTrue(result.success,result.message)
        self.assertAlmostEqual(model['diagnostics']['normalized_penalized_objective'],result.fun,places=5)

    def test_robust_location_and_constant_target(self):
        x=np.zeros((7,1)); y=np.array([0,0,0,1,1,1,100.])
        mse=fit_linear_loss(x,y,config=LossConfig(loss='mse',l2=.1))
        mae=fit_linear_loss(x,y,config=LossConfig(loss='mae',l2=.1,max_iter=10000,tol=1e-7))
        robust=fit_linear_loss(x,y,config=LossConfig(loss='huber',l2=.1))
        self.assertAlmostEqual(mse['intercept'],y.mean())
        self.assertAlmostEqual(mae['intercept'],np.median(y),places=5)
        self.assertLess(robust['intercept'],mse['intercept'])
        for loss in ('mse','mae','huber'):
            model=fit_linear_loss(x,np.ones(7)*.03,config=LossConfig(loss=loss))
            np.testing.assert_allclose(predict_linear_loss(model,x),.03,atol=1e-8)

    def test_reject_bad_inputs_and_nonconvergence(self):
        x=np.arange(60,dtype=float).reshape(20,3); y=np.sin(np.arange(20))
        with self.assertRaises(ValueError): fit_linear_loss(x,y,np.zeros(20))
        with self.assertRaises(ValueError): fit_linear_loss(x,y,config=LossConfig(delta=0))
        with self.assertRaises(ValueError): fit_linear_loss(x,y,config=LossConfig(l2=-1))
        with self.assertRaises(RuntimeError):
            fit_linear_loss(x,y,config=LossConfig(loss='mae',max_iter=1,tol=1e-14))
        self.assertEqual(target_scale(np.ones(4))[0],1)


if __name__=='__main__':
    with threadpool_limits(limits=2): unittest.main()

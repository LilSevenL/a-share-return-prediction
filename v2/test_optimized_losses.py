"""Run independent LP/QP tests against the optimized solver and compare reference."""
import unittest
from unittest.mock import patch
import numpy as np
import test_linear_losses as original_tests
import linear_losses as reference
import linear_losses_optimized as optimized
from threadpoolctl import threadpool_limits

class IndependentTests(original_tests.LossTests):
    def setUp(self):
        p = patch.object(original_tests,'fit_linear_loss',optimized.fit_linear_loss)
        p.start(); self.addCleanup(p.stop)

class EquivalenceTests(unittest.TestCase):
    def test_cached_layout_and_budget(self):
        rng=np.random.default_rng(42)
        for order in ('C','F'):
            x=np.array(rng.normal(size=(130,5)),dtype='float32',order=order)
            cached=optimized.CachedDesign(x,chunk=31)
            for (_,_,a),(_,_,b) in zip(reference._blocks(x,31),optimized._blocks(cached,31)):
                np.testing.assert_array_equal(a,b)
                self.assertEqual(a.strides,b.strides)
            y=rng.normal(size=130)
            cfg=reference.LossConfig(loss='huber',l2=0,tol=1e-6,chunk_rows=31)
            a=reference.fit_linear_loss(x,y,config=cfg)
            b=optimized.fit_linear_loss(cached,y,config=cfg)
            np.testing.assert_array_equal(a['coefficient'],b['coefficient'])
        with self.assertRaises(MemoryError): optimized.CachedDesign(x,max_bytes=1)
        with self.assertRaises(ValueError): list(optimized._blocks(cached,32))

    def test_weighted_regularized_and_singular_designs(self):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(130,5)).astype('float32')
        x[:,4]=x[:,3]
        y=rng.normal(size=130); y[:3]+=12
        w=rng.uniform(0,2,130); w[:10]=0
        for l2 in (0.,.1):
            cfg=reference.LossConfig(loss='mae',l2=l2,tol=1e-6,max_iter=15000,chunk_rows=31)
            a=reference.fit_linear_loss(x,y,w,cfg)
            b=optimized.fit_linear_loss(x.astype('float64'),y,w,cfg)
            np.testing.assert_allclose(reference.predict_linear_loss(a,x),
                reference.predict_linear_loss(b,x),atol=1e-7,rtol=1e-7)

if __name__=='__main__':
    with threadpool_limits(limits=4): unittest.main()

import mlx.core as mx
mx.set_default_device(mx.cpu)
import runpy
runpy.run_module('eval.assess_crossed_review_v10',run_name='__main__')

"""Run the unchanged crossed-population evaluator on CPU without an encoder."""
import mlx.core as mx

mx.set_default_device(mx.cpu)
assert mx.default_device() == mx.cpu

from eval.assess_crossed_review_v10 import main

main()

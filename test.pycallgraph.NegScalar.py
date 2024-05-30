from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from minitorch import MathTest, Scalar, operators, scalar

graphviz = GraphvizOutput(output_file="test.pycallgraph.NegScalar.png")

with PyCallGraph(output=graphviz):
    t1 = Scalar(5)
    # base_fn = MathTest.neg
    # scalar_fn = operators.neg
    -t1
    # operators.is_close(base_fn(t1).data, scalar_fn(t1.data))

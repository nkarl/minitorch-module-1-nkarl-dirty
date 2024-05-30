from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from minitorch import MathTest, Scalar, operators

graphviz = GraphvizOutput(output_file="test.pycallgraph.png")

with PyCallGraph(output=graphviz):
    t1, t2 = Scalar(5), Scalar(7)
    base_add = MathTest.add2
    scalar_add = operators.add
    operators.is_close(base_add(t1, t2).data, scalar_add(t1.data, t2.data))

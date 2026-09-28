r"""Selection harness on the linear scenario: :math:`n = 200`, :math:`p = 100`,
three relevant variables, ``hidden_dims=(32, 32)``.

A variable carrying a linear signal must survive, and none of the others.
The name recalls what the scenario was built to watch: without the gauge,
gradient flow keeps every unit balanced between its incoming and outgoing
weights, and a unit whose first-layer row shrinks takes its sensitivity down
with it, freezing the row away from zero.

::

    PYTHONPATH=. python benchmarks/balance_sel.py --seeds 50
"""

from selection import run

if __name__ == "__main__":
    run("linear_p100", __doc__.splitlines()[0])

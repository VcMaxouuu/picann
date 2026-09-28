r"""Selection harness on the nonlinear scenario: :math:`n = 500`, :math:`p = 100`,
three relevant variables entering through :math:`\tanh`, :math:`|\cdot|` and
:math:`\sin`, ``hidden_dims=(32, 32)``.

The even component has no linear association with the response: the warm path
must let the network learn it before the level rises to
:math:`\lambda^{\mathrm{DB}}_\alpha`.

::

    PYTHONPATH=. python benchmarks/nonlin_sel.py --seeds 50
    PYTHONPATH=. python benchmarks/nonlin_sel.py --seeds 50 --tol 1e-5
"""

from selection import run

if __name__ == "__main__":
    run("nonlinear_p100", __doc__.splitlines()[0])

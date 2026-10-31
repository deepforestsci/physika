from physika.core.expr import App, Const, Lit
from physika.core.inductive import mk_builtin_env
from physika.core.local_context import LocalContext
from physika.core.metavar import MetaVarContext
from physika.core.reduction import whnf
from physika.utils.cic_utils.inductive_utils import (INT, int_lit,
                                                     mk_vec_literal,
                                                     read_dim_vec_literal)
from physika.utils.cic_utils.sym_ring import (atom_vec, convolution, map_vec,
                                              mk_sym_decl, mk_sym_eqb_value,
                                              mk_sym_normalize_value,
                                              monomial_basis, mul_table,
                                              reg_sym_ops, vec_entries,
                                              vec_int)


def read(vec):
    """
    Helper function to reduce a ``Vec Int n`` term into integers.
    """
    return read_dim_vec_literal(vec, mk_builtin_env(), LocalContext(),
                                MetaVarContext())


def ints(*values):
    """
    Helper function to build a ``Vec Int n`` literal.
    """
    return mk_vec_literal(INT, [int_lit(v) for v in values])


def sym(name, *args):
    """
    Helper function to apply a ``Sym`` constructor or operation.
    """
    term = Const(name, ())
    for a in args:
        term = App(term, a)
    return term


class TestSymRing:
    """
    Tests for the ``Sym`` ring normal form in ``sym_ring``.
    """

    def test_decl(self):
        """``Sym`` constructors and the ``Vec Int n`` type."""
        decl = mk_sym_decl()
        assert [c.name for c in decl.constructors
                ] == ["Sym.atom", "Sym.lit", "Sym.add", "Sym.mul", "Sym.neg"]
        assert vec_int(3).arg == Lit(3)

    def test_basis(self):
        """Monomials by degree, products beyond the degree are ``None``."""
        basis = monomial_basis(2, 2)
        assert basis == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        assert len(monomial_basis(3, 3)) == 20
        table = mul_table(basis, 2)
        assert basis[table[(1, 2)]] == (1, 1)
        assert table[(3, 3)] is None

    def test_vector_terms(self):
        """Entries, entrywise ops, convolution and one-hot atoms."""
        u = ints(5, -7)
        env = mk_builtin_env()
        second = vec_entries(u, 2)[1]
        assert whnf(second, env, LocalContext(),
                    MetaVarContext()) == int_lit(-7)
        assert read(map_vec("Int.add", [u, u], 2)) == [10, -14]
        assert read(map_vec("Int.neg", [u], 2)) == [-5, 7]
        # (a + 1)² over the basis 1, a, a²
        table = mul_table(monomial_basis(1, 2), 2)
        assert read(convolution(ints(1, 1, 0), ints(1, 1, 0), 3,
                                table)) == [1, 2, 1]
        basis = monomial_basis(2, 1)
        assert read(atom_vec(Lit(1), 2, basis)) == [0, 0, 1]
        assert read(atom_vec(Lit(5), 2, basis)) == [0, 0, 0]

    def test_eqb(self):
        """``Sym.eqb`` compares normal forms entry by entry."""
        eqb = mk_sym_eqb_value(2)
        env = mk_builtin_env()
        for v, expected in ((ints(1,
                                  2), "Bool.true"), (ints(1,
                                                          3), "Bool.false")):
            result = whnf(App(App(eqb, ints(1, 2)), v), env, LocalContext(),
                          MetaVarContext())
            assert result.name == expected
        assert mk_sym_normalize_value(1, 1).binder_name == "s"

    def test_reg_sym_ops(self):
        """The kernel decides ring identities after registration."""
        env = mk_builtin_env()
        assert reg_sym_ops(env, 2, 1) == 3
        assert {"Sym", "Sym.rec", "Sym.normalize",
                "Sym.eqb"} <= set(env.constants)
        a, b = (sym("Sym.atom", Lit(i)) for i in (0, 1))
        zero, two = sym("Sym.lit", int_lit(0)), sym("Sym.lit", int_lit(2))
        for lhs, rhs, expected in ((sym("Sym.add", a, b), sym("Sym.add", b,
                                                              a), "Bool.true"),
                                   (sym("Sym.add", a,
                                        a), sym("Sym.mul", two, a),
                                    "Bool.true"), (sym("Sym.add", a,
                                                       b), a, "Bool.false"),
                                   (sym("Sym.add", a,
                                        sym("Sym.neg",
                                            a)), zero, "Bool.true")):
            claim = sym("Sym.eqb", sym("Sym.normalize", lhs),
                        sym("Sym.normalize", rhs))
            result = whnf(claim, env, LocalContext(), MetaVarContext())
            assert result.name == expected

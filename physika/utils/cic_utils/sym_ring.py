from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Tuple

from physika.core.expr import App, BinderInfo, Const, Expr, Lam, Lit, TYPE_0
from physika.core.environment import ConstantInfo, Environment, InductiveInfo
from physika.core.inductive import InductiveDecl, Constructor
from physika.core.level import LParam, LSucc, LZero
from physika.core.local_context import LocalContext
from physika.utils.cic_utils.inductive_utils import (
    NAT, INT, const_vec_motive, cons_case_returning, derive_recursor, int_lit,
    mk_arrow, mk_vec_literal, vec_rec_app)

SYM = Const("Sym", ())
BOOL = Const("Bool", ())
BOOL_REC = Const("Bool.rec", (LSucc(LZero()), ))


def vec_int(n: int) -> Expr:
    """
    Returns of an int in vecrtor form ``Vec Int n``.

    Parameters
    ----------
    n : int
        Length of the vector.

    Examples
    --------
    >>> from physika.utils.cic_utils.sym_ring import vec_int
    >>> vec_int(3).arg
    Lit(val=3)
    """
    return App(App(Const("Vec", ()), INT), Lit(n))


def mk_sym_decl() -> InductiveDecl:
    """
    Returns the inductive declaration of ``Sym : Type0``.

    Examples
    --------
    >>> from physika.utils.cic_utils.sym_ring import mk_sym_decl
    >>> decl = mk_sym_decl()
    >>> decl.name, decl.num_params, [c.name for c in decl.constructors]
    ('Sym', 0, ['Sym.atom', 'Sym.lit', 'Sym.add', 'Sym.mul', 'Sym.neg'])
    """
    binop = mk_arrow(SYM, mk_arrow(SYM, SYM))
    return InductiveDecl(name="Sym",
                         level_params=(),
                         num_params=0,
                         type=TYPE_0,
                         constructors=(Constructor("Sym.atom",
                                                   mk_arrow(NAT, SYM)),
                                       Constructor("Sym.lit",
                                                   mk_arrow(INT, SYM)),
                                       Constructor("Sym.add", binop),
                                       Constructor("Sym.mul", binop),
                                       Constructor("Sym.neg",
                                                   mk_arrow(SYM, SYM))),
                         is_recursive=True)


def monomial_basis(num_atoms: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Returns the exponent tuples of every monomial in *num_atoms* atoms of
    degree up to *max_degree*, by degree and then in lexicographic order.
    The first one is the constant monomial.

    Parameters
    ----------
    num_atoms : int
        Number of atoms.
    max_degree : int
        Highest total degree.

    Examples
    --------
    >>> from physika.utils.cic_utils.sym_ring import monomial_basis
    >>> monomial_basis(2, 2)
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    """
    basis = [
        tuple(combo.count(i) for i in range(num_atoms))
        for deg in range(max_degree + 1)
        for combo in combinations_with_replacement(range(num_atoms), deg)
    ]
    return sorted(basis, key=lambda e: (sum(e), tuple(-x for x in e)))


def mul_table(basis: List[Tuple[int, ...]],
              max_degree: int) -> Dict[Tuple[int, int], Optional[int]]:
    """
    Returns, for every pair ``(p, q)`` of basis indices, the index of the
    product monomial, or ``None`` when its degree exceeds *max_degree*.

    Parameters
    ----------
    basis : list[tuple]
        Monomials from ``monomial_basis``.
    max_degree : int
        Highest total degree.

    Examples
    --------
    >>> from physika.utils.cic_utils.sym_ring import monomial_basis, mul_table
    >>> basis = monomial_basis(2, 2)
    >>> table = mul_table(basis, 2)
    >>> basis[table[(1, 2)]]   # a * b
    (1, 1)
    >>> table[(3, 3)] is None  # a² * a² has degree 4
    True
    """
    index = {e: i for i, e in enumerate(basis)}
    table: Dict[Tuple[int, int], Optional[int]] = {}
    for p, ep in enumerate(basis):
        for q, eq in enumerate(basis):
            prod = tuple(a + b for a, b in zip(ep, eq))
            in_range = sum(prod) <= max_degree
            table[(p, q)] = index.get(prod) if in_range else None
    return table


def vec_entries(u: Expr, n: int) -> List[Expr]:
    """
    Returns the ``n`` entries of ``u : Vec Int n`` as ``Vec.rec`` terms.
    Each step peels one ``Vec.cons`` off the previous tail, so building
    all entries is linear in ``n``.

    Parameters
    ----------
    u : Expr
        Term of type ``Vec Int n``.
    n : int
        Length of *u*.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.core.reduction import whnf
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     int_lit, mk_vec_literal)
    >>> from physika.utils.cic_utils.sym_ring import vec_entries
    >>> u = mk_vec_literal(INT, [int_lit(5), int_lit(7)])
    >>> second = vec_entries(u, 2)[1]
    >>> whnf(second, mk_builtin_env(), LocalContext(),
    ...      MetaVarContext()) == int_lit(7)
    True
    """
    out: List[Expr] = []
    for length in range(n, 0, -1):
        out.append(
            vec_rec_app(const_vec_motive(INT), int_lit(0),
                        cons_case_returning(INT, pick=2), Lit(length), u))
        if length > 1:
            # the nil case is never reached on a Vec.cons
            tail = vec_int(length - 1)
            u = vec_rec_app(const_vec_motive(tail),
                            mk_vec_literal(INT, [int_lit(0)] * (length - 1)),
                            cons_case_returning(tail, pick=1), Lit(length), u)
    return out


def map_vec(op: str, vecs: List[Expr], n: int) -> Expr:
    """
    Returns the ``Vec Int n`` whose entry ``k`` is ``op`` applied to entry
    ``k`` of each vector in *vecs*.

    Parameters
    ----------
    op : str
        ``Int`` operation, e.g. ``"Int.add"``.
    vecs : list[Expr]
        Terms of type ``Vec Int n``.
    n : int
        Length of the vectors.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     int_lit, mk_vec_literal, read_dim_vec_literal)
    >>> from physika.utils.cic_utils.sym_ring import map_vec
    >>> u = mk_vec_literal(INT, [int_lit(5), int_lit(-7)])
    >>> read_dim_vec_literal(map_vec("Int.add", [u, u], 2), mk_builtin_env(),
    ...                      LocalContext(), MetaVarContext())
    [10, -14]
    """
    entries = [vec_entries(u, n) for u in vecs]
    out = []
    for k in range(n):
        term: Expr = Const(op, ())
        for e in entries:
            term = App(term, e[k])
        out.append(term)
    return mk_vec_literal(INT, out)


def convolution(u1: Expr, u2: Expr, n: int,
                table: Dict[Tuple[int, int], Optional[int]]) -> Expr:
    """
    Returns the coefficients of the product of two polynomials, a
    ``Vec Int n`` whose entry ``k`` sums ``u1[p] * u2[q]`` over the pairs
    with ``table[(p, q)] == k``.

    Parameters
    ----------
    u1 : Expr
        Coefficients of the left polynomial, ``Vec Int n``.
    u2 : Expr
        Coefficients of the right polynomial, ``Vec Int n``.
    n : int
        Size of the monomial basis.
    table : dict
        Product indices from ``mul_table``.

    Examples
    --------
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     int_lit, mk_vec_literal, read_dim_vec_literal)
    >>> from physika.utils.cic_utils.sym_ring import (
    ...     convolution, monomial_basis, mul_table)
    >>> table = mul_table(monomial_basis(1, 2), 2)  # basis 1, a, a²
    >>> a_plus_1 = mk_vec_literal(INT, [int_lit(1), int_lit(1), int_lit(0)])
    >>> square = convolution(a_plus_1, a_plus_1, 3, table)
    >>> read_dim_vec_literal(square, mk_builtin_env(), LocalContext(),
    ...                      MetaVarContext())
    [1, 2, 1]
    """
    e1, e2 = vec_entries(u1, n), vec_entries(u2, n)
    out: List[Expr] = []
    for k in range(n):
        terms = [
            App(App(Const("Int.mul", ()), e1[p]), e2[q])
            for (p, q), r in table.items() if r == k
        ]
        if not terms:
            out.append(int_lit(0))
            continue
        total = terms[0]
        for t in terms[1:]:
            total = App(App(Const("Int.add", ()), total), t)
        out.append(total)
    return mk_vec_literal(INT, out)


def atom_vec(i: Expr, num_atoms: int, basis: List[Tuple[int, ...]]) -> Expr:
    """
    Returns the coefficients of atom ``i``, a ``Vec Int M`` with a 1 in
    its monomial's slot, chosen with nested ``Bool.rec`` on
    ``Nat.eqb i j``. An index out of range gives the zero vector.

    Parameters
    ----------
    i : Expr
        Atom index, a ``Nat`` term.
    num_atoms : int
        Number of atoms.
    basis : list[tuple]
        Monomials from ``monomial_basis``.

    Examples
    --------
    >>> from physika.core.expr import Lit
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     read_dim_vec_literal)
    >>> from physika.utils.cic_utils.sym_ring import atom_vec, monomial_basis
    >>> b = atom_vec(Lit(1), 2, monomial_basis(2, 1))  # basis 1, a, b
    >>> read_dim_vec_literal(b, mk_builtin_env(), LocalContext(),
    ...                      MetaVarContext())
    [0, 0, 1]
    """
    M = len(basis)
    motive = Lam("_", BOOL, vec_int(M), BinderInfo.DEFAULT)
    index = {e: m for m, e in enumerate(basis)}
    result = mk_vec_literal(INT, [int_lit(0)] * M)
    for j in reversed(range(num_atoms)):
        slots = [int_lit(0)] * M
        slots[index[tuple(int(t == j) for t in range(num_atoms))]] = int_lit(1)
        is_j = App(App(Const("Nat.eqb", ()), i), Lit(j))
        result = App(
            App(App(App(BOOL_REC, motive), result),
                mk_vec_literal(INT, slots)), is_j)
    return result


def mk_sym_normalize_value(num_atoms: int, max_degree: int) -> Expr:
    """
    Returns the body of ``Sym.normalize : Sym -> Vec Int M``, a
    ``Sym.rec`` with one case per constructor: an atom is a one-hot
    vector, a literal fills the constant slot, ``add``/``neg`` work entry
    by entry and ``mul`` is a ``convolution``.

    Parameters
    ----------
    num_atoms : int
        Number of atoms.
    max_degree : int
        Highest total degree.

    Examples
    --------
    >>> from physika.utils.cic_utils.sym_ring import mk_sym_normalize_value
    >>> mk_sym_normalize_value(1, 1).binder_name
    's'
    """
    basis = monomial_basis(num_atoms, max_degree)
    M = len(basis)
    vec_m = vec_int(M)
    # monomial_basis puts the constant monomial in slot 0
    assert basis[0] == (0, ) * num_atoms

    lctx, i_fv = LocalContext().push_local("i", NAT)
    m_atom = lctx.mk_lambda([i_fv], atom_vec(i_fv, num_atoms, basis))

    lctx, n_fv = LocalContext().push_local("n", INT)
    m_lit = lctx.mk_lambda([n_fv],
                           mk_vec_literal(INT,
                                          [n_fv] + [int_lit(0)] * (M - 1)))

    lctx, x = LocalContext().push_local("x", SYM)
    lctx, ihx = lctx.push_local("ihx", vec_m)
    m_neg = lctx.mk_lambda([x, ihx], map_vec("Int.neg", [ihx], M))
    lctx, y = lctx.push_local("y", SYM)
    lctx, ihy = lctx.push_local("ihy", vec_m)
    m_add = lctx.mk_lambda([x, ihx, y, ihy], map_vec("Int.add", [ihx, ihy], M))
    m_mul = lctx.mk_lambda([x, ihx, y, ihy],
                           convolution(ihx, ihy, M,
                                       mul_table(basis, max_degree)))

    lctx, s = LocalContext().push_local("s", SYM)
    rec: Expr = Const("Sym.rec", (LParam("u"), ))
    for arg in (Lam("_", SYM, vec_m, BinderInfo.DEFAULT), m_atom, m_lit, m_add,
                m_mul, m_neg, s):
        rec = App(rec, arg)
    return lctx.mk_lambda([s], rec)


def mk_sym_eqb_value(M: int) -> Expr:
    """
    Returns the body of ``Sym.eqb : Vec Int M -> Vec Int M -> Bool``,
    which compares two normal forms entry by entry with ``Int.eqb``.

    Parameters
    ----------
    M : int
        Size of the monomial basis.

    Examples
    --------
    >>> from physika.core.expr import App
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.core.reduction import whnf
    >>> from physika.utils.cic_utils.inductive_utils import (
    ...     int_lit, mk_vec_literal)
    >>> from physika.utils.cic_utils.sym_ring import mk_sym_eqb_value
    >>> u = mk_vec_literal(INT, [int_lit(1), int_lit(2)])
    >>> v = mk_vec_literal(INT, [int_lit(1), int_lit(3)])
    >>> eqb = mk_sym_eqb_value(2)
    >>> whnf(App(App(eqb, u), v), mk_builtin_env(), LocalContext(),
    ...      MetaVarContext()).name
    'Bool.false'
    """
    lctx, u1 = LocalContext().push_local("u1", vec_int(M))
    lctx, u2 = lctx.push_local("u2", vec_int(M))
    motive = Lam("_", BOOL, BOOL, BinderInfo.DEFAULT)
    acc: Expr = Const("Bool.true", ())
    for e1, e2 in zip(vec_entries(u1, M), vec_entries(u2, M)):
        # acc := (e1 == e2) && acc
        is_eq = App(App(Const("Int.eqb", ()), e1), e2)
        acc = App(
            App(App(App(BOOL_REC, motive), Const("Bool.false", ())), acc),
            is_eq)
    return lctx.mk_lambda([u1, u2], acc)


def reg_sym_ops(env: Environment, num_atoms: int, max_degree: int) -> int:
    """
    Registers ``Sym`` with its recursor, ``Sym.normalize`` and ``Sym.eqb``
    in *env*. Returns ``M``, the size of the monomial basis.

    Parameters
    ----------
    env : Environment
        Environment to extend.
    num_atoms : int
        Number of atoms.
    max_degree : int
        Highest total degree.

    Examples
    --------
    >>> from physika.core.expr import App, Const, Lit
    >>> from physika.core.inductive import mk_builtin_env
    >>> from physika.core.local_context import LocalContext
    >>> from physika.core.metavar import MetaVarContext
    >>> from physika.core.reduction import whnf
    >>> from physika.utils.cic_utils.sym_ring import reg_sym_ops
    >>> env = mk_builtin_env()
    >>> reg_sym_ops(env, 2, 1)  # basis 1, a, b
    3
    >>> sym = lambda name, *args: App(App(Const(name, ()), args[0]), args[1])
    >>> a, b = (App(Const("Sym.atom", ()), Lit(i)) for i in (0, 1))
    >>> norm = lambda t: App(Const("Sym.normalize", ()), t)
    >>> claim = sym("Sym.eqb", norm(sym("Sym.add", a, b)),
    ...             norm(sym("Sym.add", b, a)))
    >>> whnf(claim, env, LocalContext(), MetaVarContext()).name
    'Bool.true'
    """
    decl = mk_sym_decl()
    rec = derive_recursor(decl)
    env.add_constant(
        ConstantInfo(decl.name, decl.level_params, decl.type, None))
    env.add_inductive(
        InductiveInfo(decl=decl,
                      ctors={
                          c.name:
                          ConstantInfo(c.name, decl.level_params, c.type, None)
                          for c in decl.constructors
                      },
                      recursor=ConstantInfo(rec.name, rec.level_params,
                                            rec.type, None),
                      rec_info=rec))
    M = len(monomial_basis(num_atoms, max_degree))
    env.add_constant(
        ConstantInfo("Sym.normalize", (), mk_arrow(SYM, vec_int(M)),
                     mk_sym_normalize_value(num_atoms, max_degree)))
    env.add_constant(
        ConstantInfo("Sym.eqb", (),
                     mk_arrow(vec_int(M), mk_arrow(vec_int(M), BOOL)),
                     mk_sym_eqb_value(M)))
    return M

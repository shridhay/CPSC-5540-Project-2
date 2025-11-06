from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Self, Union
import numpy as np
from fractions import Fraction
from functools import reduce
import sys
import time
import copy

DEBUG = False

def approx_leq(x, y):
        return ((x <= y) | np.isclose(x, y)).all()

def test_simplex():

    # Examples from the UBC website
    A = np.array([[2.,1.,1.,3.],[1.,3.,1.,2.]])
    c = np.array([6.,8.,5.,9.])
    b = np.array([5.,3.])
    r = np.array([2., 0., 1., 0.])
    actual = simplex(A, b, c)
    np.testing.assert_allclose(actual[1], r)
    np.testing.assert_array_compare(approx_leq, A @ actual[1], b)
    np.testing.assert_almost_equal(actual[0], c.T @ actual[1])

    A = np.array([[0,2,3],[1,1,2],[1,2,3]])
    b = np.array([5,4,7])
    c = np.array([2,3,4])
    r = np.array([1.5, 2.5, 0.])
    actual = simplex(A, b, c)
    np.testing.assert_allclose(actual[1], r)
    np.testing.assert_array_compare(approx_leq, A @ actual[1], b)
    np.testing.assert_almost_equal(actual[0], c.T @ actual[1])

    A = np.array([[-1,-1,-1,-1],[2,-1,1,-1]])
    b = np.array([-2,1])
    c = np.array([0,0,0,-1])
    actual = simplex(A, b, c)
    np.testing.assert_almost_equal(actual[0], 0)
    np.testing.assert_array_compare(approx_leq, A @ actual[1], b)
    np.testing.assert_almost_equal(actual[0], c.T @ actual[1])

def simplex(a, b, c):
    """
    Runs the Simplex algorithm using the tableau matrix.

    Returns
    - if no feasible solution:     None
    - if exists feasible solution: tuple(Max objective value, Array of assignments) 

    Reference: https://ubcmath.github.io/python/linear-programming/simplex.html
    """

    # Putting together the tableau data structure
    rows, cols = a.shape
    b = b.reshape(-1, 1)
    I = np.identity(rows, dtype='int64')+Fraction()
    z = np.zeros(rows + 1, dtype='int64')+Fraction()
    T = np.vstack([np.hstack([a, I, b]), 
                   np.hstack([c, z])])
    
    if DEBUG:
        print(T)
    basis = np.arange(cols, cols + rows)
    
    if (c <= 0).all():
        if (b >= 0).all():
            return 0, np.zeros(rows, dtype='int64') + Fraction()
        
        # Pivot x0 away
        entering_idx, exiting_idx = cols - 1, None

        # Deterimining the Exit Value
        value = 0
        for idx in range(len(T[:-1, -1])):
            if T[:-1, -1][idx] <= 0:
                v = T[:-1, -1][idx]/T[:-1, entering_idx][idx]
                if v > value:
                    exiting_idx = idx
                    value = v
        
        # Doing the Pivot operation with the entering and exiting indices
        rows = T.shape[0]
        if DEBUG:
            print(exiting_idx, entering_idx)
        basis[exiting_idx] = entering_idx
        v = T[exiting_idx, entering_idx]
        T[exiting_idx, :] = T[exiting_idx, :] / v
        for idx in range(rows):
            if idx != exiting_idx:
                T[idx, :] = T[idx, :] - (T[idx, entering_idx] * T[exiting_idx, :])

    # Run the Simplex Algorithm until it breaks out of the loop
    while(True):
        if DEBUG:
            print(T)
        entering_idx, exiting_idx = None, None
        
        #print(T[-1, :-1])
        if (T[-1, :-1] <= 0).all():
            # print("Satisfied")
            # Get the values of all of the decision variables
            values = np.zeros(b.shape[0] + len(c), dtype='int64') + Fraction()
            for idx_i in basis:
                b = True
                for idx_j in range(T.shape[0] - 1):
                    if b and T[idx_j, idx_i] == 1:
                        values[idx_i] = T[idx_j, - 1]
                        b = False

            values = values[:cols]
            # {-T[-1, -1]}
            #print(values)
            return -T[-1, -1], values
        
        # Deteriming the Entering Value
        value = 0
        for idx in range(len(T[-1, :-1])):
            # Bland's rule, to avoid cycles
            if T[-1, :-1][idx] > 0:
                entering_idx = idx
                break

        # Deterimining the Exit Value
        value = float("inf")
        for idx in range(len(T[:-1, -1])):
            if T[:-1, -1][idx] >= 0 and T[:-1, entering_idx][idx] > 1e-5:
                v = T[:-1, -1][idx]/T[:-1, entering_idx][idx]
                if v < value:
                    exiting_idx = idx
                    value = v

        # If the exit value cannot be determined, terminate.
        if exiting_idx is None:
            # print("Unsatisfiable")
            values = np.zeros(b.shape[0] + len(c), dtype='int64') + Fraction()
            for idx_i in basis:
                b = True
                for idx_j in range(T.shape[0] - 1):
                    if b and T[idx_j, idx_i] == 1:
                        values[idx_i] = T[idx_j, - 1]
                        b = False
            values = values[:cols]
            return -T[-1, -1], values
        
        # Doing the Pivot operation with the entering and exiting indices
        rows = T.shape[0]
        if DEBUG:
            print(exiting_idx, entering_idx)
        v = T[exiting_idx, entering_idx]
        T[exiting_idx, :] = T[exiting_idx, :] / v
        for idx in range(rows):
            if idx != exiting_idx:
                basis[exiting_idx] = entering_idx
                T[idx, :] = T[idx, :] - (T[idx, entering_idx] * T[exiting_idx, :])

        if DEBUG:
            time.sleep(0.5)

                
def lex(sexpr):
    sexpr = sexpr.replace("(", " ( ").replace(")", " ) ")
    l = sexpr.split()
    for token in l:
        yield token

def dictplus(d1, d2):
    out = d1.copy()
    for k, v in d2.items():
        if k in d1:
            out[k] += v 
        else:
            out[k] = v 
    return out

def dictminus(d1, d2):
    out = d1.copy()
    for k, v in d2.items():
        if k in d1:
            out[k] -= v 
        else:
            out[k] = -v 
    return out

class AtomShouldBeFormulaException(Exception):
    def __init__(self, message="This atom is actually a formula."):
        self.message = message
        super().__init__(self.message)

class CloseParenException(Exception):
    def __init__(self, message="This is actually a close paren. Move a level up"):
        self.message = message
        super().__init__(self.message)

class SmtLib(ABC):
    pass

    @classmethod
    @abstractmethod
    def parse(cls, lexer) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def pretty(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def getVars(self) -> list[str]:
        raise NotImplementedError()

@dataclass
class Script(SmtLib):
    # Script ::= Command+
    
    commands: list[Command]

    def __init__(self):
        self.commands = []

    @classmethod
    def parse(cls, lexer):
        me = cls()
        while True:
            try:
                next_cmd = Command.parse(lexer)
                me.commands.append(next_cmd)
            except StopIteration:
                return me

    def pretty(self) -> str:
        nl = "\n"
        return f"Script({nl + (',' + nl).join([c.pretty() for c in self.commands])})"
    
    def getVars(self) -> list[str]:
        return sorted({var for cmd in self.commands for var in cmd.getVars()})
    
    def coef(self) -> list[str]:
        l = []
        for cmd in self.commands:
            l += cmd.coef()
        return l
    
    def tableau_args(self):
        vars = self.getVars()
        coefs = self.coef()

        # plus and minus version for each variable, plus x_0 at the end
        A = np.zeros((len(coefs), 2 * len(vars) + 1), dtype='int64') + Fraction()
        b = np.zeros(len(coefs), dtype='int64') + Fraction()
        c = np.zeros(2 * len(vars) + 1, dtype='int64') + Fraction()
        c[-1] = -1

        for i, constraint in enumerate(coefs):
            for var, val in constraint.items():
                if var == "_c":
                    b[i] = val
                else:
                    var_idx = vars.index(var)
                    A[i][2 * var_idx] = val
                    A[i][2 * var_idx + 1] = -val
            A[i][-1] = Fraction(-1)       # x0

        return A, b, c, vars
    
    def check_sat(self, ILP = False):
        """
        Prints a satisfying solution. Does not return anything.
        """
        A, b, c, vars = self.tableau_args()
        if (b >= 0).all():
            for i, v in enumerate(vars):
                print(f"{v}=0")
        elif not(ILP):
            soln = simplex(A, b, c)
            if soln is None or np.abs(soln[0] - 0) > 1e-5:
                print("UNSAT")
            else:
                assn = soln[1]
                for i, v in enumerate(vars):
                    print(f"{v}={assn[2*i]-assn[2*i+1]}")
        else:
            soln = simplex(A, b, c)
            if soln is None or np.abs(soln[0] - 0) > 1e-5:
                print("UNSAT")
            else:
                assn = soln[1]
                for i, v in enumerate(vars):
                    print(f"{v}={assn[2*i]-assn[2*i+1]}")


@dataclass
class Command(SmtLib):
    # Command ::= (assert Formula)
    
    formula: Formula

    def __init__(self, formula):
        self.formula = formula

    @classmethod
    def parse(cls, lexer):
        n = next(lexer)
        assert n == "("
        assert next(lexer) == "assert"
        me = cls(Formula.parse(lexer))
        assert next(lexer) == ")"
        return me

    def pretty(self) -> str:
        return f"Command({self.formula.pretty()})"
    
    def getVars(self) -> list[str]:
        return self.formula.getVars()
    
    def coef(self) -> list[dict]:
        # compute coefficients of A_i x <= b_i
        return self.formula.coef()

@dataclass
class Formula(SmtLib):
    # formula ::= Atom | (and Atom+)
    
    atoms: list[Atom]

    def __init__(self):
        self.atoms = []

    @classmethod
    def parse(cls, lexer):
        me = cls()
        try:
            single_atom = Atom.parse(lexer)
            me.atoms = [single_atom]
            return me
        except AtomShouldBeFormulaException:
            while True:
                try:
                    next_atom = Atom.parse(lexer)
                    me.atoms.append(next_atom)
                except CloseParenException:
                    return me
                
    def pretty(self) -> str:
        nl = "\n"
        return f"Formula({nl + (',' + nl).join([a.pretty() for a in self.atoms])})"
    
    def getVars(self) -> list[str]:
        return sorted({var for atom in self.atoms for var in atom.getVars()})
    
    def coef(self) -> list[dict]:
        l = []
        for atom in self.atoms:
            l += atom.coef()
        return l

@dataclass
class Atom(SmtLib):
    # atom ::= (b Term Term)
    # b ::= < | > | >= | <= | =

    op: str
    lhs: Term
    rhs: Term

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    @classmethod
    def parse(cls, lexer):
        operation = [">", "<", ">=", "<=", "="]
        paren = next(lexer)
        if paren == ")":
            raise CloseParenException()
        assert paren == "("
        op = next(lexer)
        if op == "and":
            raise AtomShouldBeFormulaException()
        assert (op in operation)
        lhs = Term.parse(lexer)
        rhs = Term.parse(lexer)
        assert next(lexer) == ")"
        return cls(op, lhs, rhs)

    def pretty(self) -> str:
        return f"Atom(op=\"{self.op}\", lhs={self.lhs.pretty()}, rhs={self.rhs.pretty()})"
    
    def getVars(self) -> list[str]:
        return sorted(set(self.lhs.getVars()) | set(self.rhs.getVars()))
    
    def coef(self) -> list[dict]:
        # compute coefficients of A_i x <= b_i
        if self.op == ">":
            new_atom = Atom("<=", Plus.from_parts([self.rhs, Rational("1/1000")]), self.lhs)
            return new_atom.coef()
        elif self.op == "<":
            new_atom = Atom("<=", Plus.from_parts([self.lhs, Rational("1/1000")]), self.rhs)
            return new_atom.coef()
        elif self.op == ">=":
            new_atom = Atom("<=", self.rhs, self.lhs)
            return new_atom.coef()
        elif self.op == "=":
            new_atom1 = Atom("<=", self.lhs, self.rhs)
            new_atom2 = Atom("<=", self.rhs, self.lhs)
            return new_atom1.coef() + new_atom2.coef()
        elif self.op == "<=":
            lhs_coef: dict = self.lhs.coef()
            rhs_coef: dict  = self.rhs.coef()
            return [dictminus(lhs_coef, rhs_coef)]


@dataclass
class Term(SmtLib):
    # term ::= (+ Term+) | (- Term+) | (* Rational Term) | Rational | Var
    
    variant: Union[Plus, Minus, Times, Rational, Var]

    def __init__(self, variant):
        self.variant = variant

    @classmethod
    def parse(cls, lexer):
        next_token = next(lexer)
        if next_token == "(":
            op = next(lexer)
            if op == "+":
                return cls(Plus.parse(lexer))
            elif op == "-":
                return cls(Minus.parse(lexer))
            elif op == "*":
                return cls(Times.parse(lexer))
            assert False
        elif next_token == ")":
            raise CloseParenException()
        elif re.fullmatch(r"-?\d+(\.\d+)?", next_token): 
            return cls(Rational(next_token))
        else:
            return cls(Var(next_token))

    def pretty(self) -> str:
        return f"Term({self.variant.pretty()})"
    
    def getVars(self) -> list[str]:
        return self.variant.getVars()
    
    def coef(self) -> dict:
        return self.variant.coef()

@dataclass
class Plus(SmtLib):
    parts: list[Term]
    
    def __init__(self):
        self.parts = []

    @classmethod
    def from_parts(cls, parts):
        me = cls()
        me.parts = parts 
        return me

    @classmethod
    def parse(cls, lexer):
        me = cls()
        while True:
            try:
                next_term = Term.parse(lexer)
                me.parts.append(next_term)
            except CloseParenException:
                return me

    def pretty(self) -> str:
        return f"Plus({','.join([a.pretty() for a in self.parts])})"
    
    def getVars(self) -> list[str]:
        return sorted({var for part in self.parts for var in part.getVars()})
    
    def coef(self) -> dict:
        return reduce(dictplus, [part.coef() for part in self.parts])

@dataclass
class Minus(SmtLib):
    parts: list[Term]
    
    def __init__(self):
        self.parts = []

    @classmethod
    def from_parts(cls, parts):
        me = cls()
        me.parts = parts 
        return me

    @classmethod
    def parse(cls, lexer):
        me = cls()
        while True:
            try:
                next_term = Term.parse(lexer)
                me.parts.append(next_term)
            except CloseParenException:
                if len(me.parts) == 0:
                    me.parts = [0] + me.parts
                return me

    def pretty(self) -> str:
        return f"Minus({','.join([a.pretty() for a in self.parts])})"
    
    def getVars(self) -> list[str]:
        return sorted({var for part in self.parts for var in part.getVars()})
    
    def coef(self) -> dict:
        return reduce(dictminus, [part.coef() for part in self.parts])

@dataclass
class Times(SmtLib):
    mul_by: Rational
    part: Term
    
    def __init__(self, mul_by, part):
        self.mul_by = mul_by
        self.part = part

    @classmethod
    def parse(cls, lexer):
        num = next(lexer)
        part = Term.parse(lexer)
        assert next(lexer) == ")"
        return cls(Rational(num), part)

    def pretty(self) -> str:
        return f"Times(mul_by={self.mul_by.pretty()}, part={self.part.pretty()})"
    
    def getVars(self) -> list[str]:
        return self.part.getVars()
    
    def coef(self) -> dict:
        return {k : v * self.mul_by.to_fractional() for k, v in self.part.coef().items()}

@dataclass
class Rational(SmtLib):
    positive: bool
    num: int
    denom: int
    
    def __init__(self, s):
        pattern = r'(-?)(\d+)/(\d+)'
        match = re.match(pattern, s)
        if match:
            sign, num, denom = match.groups()
            self.positive = (sign == "")
            self.num = int(num)
            self.denom = int(denom)
        else:
            try:
                n = int(s)
                self.positive = (n > 0)
                self.num = n if n > 0 else -n
                self.denom = 1
            except:
                assert False

    @classmethod
    def parse(cls, lexer):
        raise NotImplementedError()
    
    def to_float(self):
        return (1 if self.positive else -1) * self.num / self.denom
    
    def to_fractional(self):
        return Fraction((1 if self.positive else -1) * self.num, self.denom)

    def pretty(self) -> str:
        return f"Rational({'-' if not self.positive else ''}{self.num}/{self.denom})"
    
    def getVars(self) -> list[str]:
        return []
    
    def coef(self) -> dict:
        return {"_c": -self.to_fractional()}

@dataclass
class Var(SmtLib):
    name: str

    def __init__(self, name):
        self.name = name

    @classmethod
    def parse(cls, lexer):
        raise NotImplementedError()

    def pretty(self) -> str:
        return f"Var({self.name})"
    
    def getVars(self) -> list[str]:
        return [self.name]
    
    def coef(self) -> dict:
        return {self.name : Fraction(1)}


if __name__ == "__main__":
    ILP = False
    if len(sys.argv) > 1:
        path = str(sys.argv[1])        
        try:
            with open(path, 'r') as f:
                content = f.read()                
                ast = Script.parse(lex(content))
                A, b, c, _ = ast.tableau_args()
                if DEBUG:
                    print("\nParse tree:")
                    print(ast.pretty())
                    print("\nCoefficients:")
                    print(ast.coef())
                    print("\nA:")
                    print(A)
                    print("\nb:")
                    print(b)
                    print("\nc:")
                    print(c)
                # print(simplex(A, b, c))
                if len(sys.argv) > 2:
                    s = str(sys.argv[2])
                    if s == "--i":
                        ILP = True
                if ILP:
                    print(f"ILP: {ILP}")
                ast.check_sat(ILP)
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
    else:
        print("Usage: python process_file.py <path_to_text_file>")

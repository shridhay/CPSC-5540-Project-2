from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Self, Union
import numpy as np
from fractions import Fraction
from functools import reduce
import sys

def simplex(a, b, c):
    rows = a.shape[0]
    b = b.reshape(-1, 1)
    T = np.vstack([np.hstack([a, np.identity(rows), b]), 
                   np.hstack([c, np.zeros(rows + 1)])])
    while(True):
        print(T)
        entering_idx, exiting_idx = None, None
        if (T[-1, :-1] <= 0).all():
            print("Satisfied")
            solution = -T[-1, -1]
            print(f"Solution: {solution}")
            return solution
        else:
            entering_idx, value = None, 0
            for idx in range(len(T[-1, :-1])):
                if T[-1, :-1][idx] > value:
                    entering_idx = idx
                    value = T[-1, :-1][idx]
        exiting_idx, value = None, float("inf")
        for idx in range(len(T[:-1, -1])):
            if T[:-1, -1][idx] >= 0 and T[:-1, entering_idx][idx] > 0:
                v = T[:-1, -1][idx]/T[:-1, entering_idx][idx]
                if v < value:
                    exiting_idx = idx
                    value = v
        if exiting_idx is None:
            print("Unsatisfiable")
            return None
        rows = T.shape[0]
        v = T[exiting_idx, entering_idx]
        T[exiting_idx, :] = T[exiting_idx, :] / v
        for idx in range(rows):
            if idx != exiting_idx:
                T[idx, :] = T[idx, : ] - (T[idx, entering_idx] * T[exiting_idx, :])
                
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

    def check_sat(self):
        ## TODO: The bulk of the work will be in here
        raise NotImplementedError()
    
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
        A = np.zeros((len(coefs), 2 * len(vars) + 1))
        b = np.zeros(len(coefs))
        c = np.zeros(2 * len(vars) + 1)
        c[-1] = -1

        for i, constraint in enumerate(coefs):
            for var, val in constraint.items():
                if var == "_c":
                    b[i] = val
                else:
                    var_idx = vars.index(var)
                    A[i][2 * var_idx] = val
                    A[i][2 * var_idx + 1] = val
            A[i][-1] = -1       # x0

        return A, b, c

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
    
    def coef(self) -> list[str]:
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
        return {k : v * self.mul_by.to_float() for k, v in self.part.coef().items()}

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

    def pretty(self) -> str:
        return f"Rational({'-' if not self.positive else ''}{self.num}/{self.denom})"
    
    def getVars(self) -> list[str]:
        return []
    
    def coef(self) -> dict:
        return {"_c": -self.to_float()}

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
        return {self.name : 1}


if __name__ == "__main__":
    ILP = False
    if len(sys.argv) > 1:
        path = str(sys.argv[1])
        try:
            with open(path, 'r') as f:
                content = f.read()
                print("File content:")
                print(content)
                ast = Script.parse(lex(content))
                print("\nParse tree:")
                print(ast.pretty())
                print("\nVariables:")
                print(ast.getVars())
                print("\nCoefficients:")
                print(ast.coef())
                A, b, c = ast.tableau_args()
                print("\nA:")
                print(A)
                print("\nb:")
                print(b)
                print("\nc:")
                print(c)
                simplex(A, b, c)
                # if len(sys.argv) > 2:
                #     s = str(sys.argv[2])
                #     if s == "--i":
                #         ILP = True
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
    else:
        print("Usage: python process_file.py <path_to_text_file>")

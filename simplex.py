from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import Self, Union
import re
import numpy as np
from fractions import Fraction
import sys

def lex(sexpr):
    sexpr = sexpr.replace("(", " ( ").replace(")", " ) ")
    l = sexpr.split()
    for token in l:
        yield token

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

class Plus(SmtLib):
    parts: list[Term]
    
    def __init__(self):
        self.parts = []

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

class Minus(SmtLib):
    parts: list[Term]
    
    def __init__(self):
        self.parts = []

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

class Times(SmtLib):
    coef: Rational
    part: Term
    
    def __init__(self, coef, part):
        self.coef = coef
        self.part = part

    @classmethod
    def parse(cls, lexer):
        num = next(lexer)
        part = Term.parse(lexer)
        assert next(lexer) == ")"
        return cls(Rational(num), part)

    def pretty(self) -> str:
        return f"Times(coef={self.coef.pretty()}, part={self.part.pretty()})"


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

    def pretty(self) -> str:
        return f"Rational({'-' if not self.positive else ''}{self.num}/{self.denom})"

class Var(SmtLib):
    name: str

    def __init__(self, name):
        self.name = name

    @classmethod
    def parse(cls, lexer):
        raise NotImplementedError()

    def pretty(self) -> str:
        return f"Var({self.name})"

if __name__ == "__main__":
    # print(Script.parse(lex("(assert (and (>= x 1) (<= (* 2 x) 1)))  ( assert (= 1 1 ))")).pretty())
    ILP = False
    if len(sys.argv) > 1:
        path = str(sys.argv[1])
        try:
            with open(path, 'r') as f:
                content = f.read()
                print("File content:")
                print(content)
            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found.")
    else:
        print("Usage: python process_file.py <path_to_text_file>")
    if len(sys.argv) > 2:
        s = str(sys.argv[2])
        if s == "--i":
            ILP = True

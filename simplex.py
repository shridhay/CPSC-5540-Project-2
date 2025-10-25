from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Self, Union
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

@dataclass
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
    
    def getVars(self) -> list[str]:
        return sorted({var for part in self.parts for var in part.getVars()})

@dataclass
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
    
    def getVars(self) -> list[str]:
        return sorted({var for part in self.parts for var in part.getVars()})

@dataclass
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
    
    def getVars(self) -> list[str]:
        return self.part.getVars()

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

    def pretty(self) -> str:
        return f"Rational({'-' if not self.positive else ''}{self.num}/{self.denom})"
    
    def getVars(self) -> list[str]:
        return []

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

if __name__ == "__main__":
    ILP = False
    if len(sys.argv) > 1:
        path = str(sys.argv[1])
        try:
            with open(path, 'r') as f:
                content = f.read()
                print("File content:")
                print(content)
                print("\nParse tree:")
                print(Script.parse(lex(content)).pretty())
                print("\nVariables:")
                print(Script.parse(lex(content)).getVars())
                # if len(sys.argv) > 2:
                #     s = str(sys.argv[2])
                #     if s == "--i":
                #         ILP = True
        except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
    else:
        print("Usage: python process_file.py <path_to_text_file>")

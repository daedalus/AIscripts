"""
The Cassowary Linear Arithmetic
Constraint Solving Algorithm
GREG J. BADROS
and
ALAN BORNING
University of Washington
and
PETER J. STUCKEY
University of Melbourne
"""

import numpy as np
from collections import defaultdict

class Variable:
    def __init__(self, name, value=0.0, restricted=False):
        self.name = name
        self.value = value
        self.restricted = restricted  # If variable >= 0

class Constraint:
    def __init__(self, expr, strength, weight=1.0):
        self.expr = expr  # Dict of {Variable: coefficient}
        self.strength = strength  # Numeric priority (higher = stronger)
        self.weight = weight
        self.error_plus = None
        self.error_minus = None

class CassowarySolver:
    def __init__(self):
        self.tableau = {}          # {basic_var: (expr, constant)}
        self.columns = defaultdict(set)  # {param: set(basic_vars)}
        self.objective = {}        # {var: coefficient}
        self.strengths = [1000, 100, 1]  # Required, strong, weak
        
    def add_constraint(self, constraint):
        # Convert constraint to equality with error variables
        expr = constraint.expr.copy()
        constant = -expr.pop('_const', 0)
        
        if constraint.strength > 0:  # Non-required
            ep = Variable(f'ep_{constraint.strength}', restricted=True)
            em = Variable(f'em_{constraint.strength}', restricted=True)
            expr[ep] = -1
            expr[em] = 1
            self.objective[ep] = constraint.weight * self.strengths[constraint.strength]
            self.objective[em] = constraint.weight * self.strengths[constraint.strength]
            constraint.error_plus = ep
            constraint.error_minus = em

        # Add to tableau
        self._add_row(Constraint(expr, constraint.strength), constant)
        self._optimize()

    def _add_row(self, constraint, constant):
        # Find subject variable (pivot)
        subject = next((v for v in constraint.expr if not v.restricted), None)
        if not subject:
            subject = next(iter(constraint.expr.keys()))
        
        # Solve for subject
        coeff = constraint.expr[subject]
        expr = {v: -c/coeff for v, c in constraint.expr.items() if v != subject}
        const = constant / coeff
        
        # Substitute in other rows
        for row_var, (row_expr, row_const) in self.tableau.items():
            if subject in row_expr:
                factor = row_expr.pop(subject)
                for v, c in expr.items():
                    row_expr[v] = row_expr.get(v, 0) + factor * c
                row_const += factor * const
        
        # Add new row
        self.tableau[subject] = (expr, const)
        self.columns[subject].add(subject)
        for v in expr:
            self.columns[v].add(subject)

    def _optimize(self):
        # Dual simplex optimization
        while True:
            # Find infeasible row
            leaving = next((v for v, (_, c) in self.tableau.items() 
                          if v.restricted and c < 0), None)
            if not leaving:
                break
                
            # Find entering variable
            row_expr, row_const = self.tableau[leaving]
            entering = min(
                (v for v, c in row_expr.items() if c < 0),
                key=lambda v: self.objective.get(v, 0) / row_expr[v],
                default=None
            )
            
            if not entering:
                raise Exception("Infeasible solution")
                
            self._pivot(leaving, entering)

    def _pivot(self, leaving, entering):
        # Perform pivot operation
        row_expr, row_const = self.tableau.pop(leaving)
        coeff = row_expr[entering]
        
        # Update entering row
        new_expr = {v: c/-coeff for v, c in row_expr.items() if v != entering}
        new_const = row_const / -coeff
        new_expr[leaving] = 1/-coeff
        self.tableau[entering] = (new_expr, new_const)
        
        # Substitute in other rows
        for var in self.columns[entering]:
            expr, const = self.tableau[var]
            if entering in expr:
                factor = expr.pop(entering)
                for v, c in new_expr.items():
                    expr[v] = expr.get(v, 0) + factor * c
                const += factor * new_const

    def suggest_value(self, var, value):
        # Edit variable value
        if var in self.tableau:
            expr, const = self.tableau[var]
            delta = value - const
            for v, c in expr.items():
                self.tableau[v][1] += c * delta
        self._optimize()

    def solve(self):
        # Update variable values
        for var in self.tableau:
            if var.restricted: continue
            expr, const = self.tableau[var]
            var.value = const + sum(v.value * c for v, c in expr.items())

# Example usage
if __name__ == "__main__":
    solver = CassowarySolver()
    
    # Create variables
    x = Variable('x')
    y = Variable('y')
    
    # Add constraints: x >= 10 (required), y == x + 5 (strong)
    solver.add_constraint(Constraint({x: 1, '_const': -10}, strength=0))
    solver.add_constraint(Constraint({x: 1, y: -1, '_const': -5}, strength=1))
    
    solver.solve()
    print(f"x = {x.value}, y = {y.value}")  # x=10, y=15
    
    # Move x to 20 (edit)
    solver.suggest_value(x, 20)
    solver.solve()
    print(f"x = {x.value}, y = {y.value}")  # x=20, y=25

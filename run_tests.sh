#!/bin/bash

# Run SAT benchmarks
for i in {1..13}; do
    python simplex.py "benchmarks/sat/bench${i}.txt"
done

# Run UNSAT benchmarks
for i in {1..13}; do
    python simplex.py "benchmarks/unsat/bench${i}.txt"
done

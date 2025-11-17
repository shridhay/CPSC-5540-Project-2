#!/bin/bash

# Run SAT benchmarks
for i in {1..13}; do
    echo "Verifying SAT benchmark $i..."
    python3 simplex.py "benchmarks/sat/bench$i.txt"
done

# Run UNSAT benchmarks
for i in {1..13}; do
    echo "Verifying UNSAT benchmark $i..."
    python3 simplex.py "benchmarks/unsat/bench$i.txt"
done

# CPSC 5540 Project 2: Simplex algorithm

This implementation was developed by Jonathan Chen and Hridhay Suresh.

## Requirements

This program is written in **Python 3.14**. It may work on earlier versions too.

This program also requires the `numpy` library version 2.2.6, which can be installed through `pip install numpy`. It is recommended to run in a virtual environment or conda environment containing numpy. No other libraries are needed.

## Quick Start

To test on a benchmark 1, run `python simplex.py benchmarks/sat/bench1.txt` in the terminal. Note: if your machine uses `python3` as an alias for `python` you should call `python3 simplex.py benchmars/sat/bench1.txt` instead. If you would like to run a test with Integer Linear Programming (ILP) enabled, use the switch `--i` after the benchmark test. For example, to run test: `benchmarks/sat/bench1.txt` with Integer Linear Programming enabled, run `python simplex.py benchmarks/sat/bench1.txt --i` in the terminal. If your machine uses `python3` as an alias for `python` you should run `python3 simplex.py benchmarks/sat/bench1.txt --i` instead.

For convenience we have included `./run_tests.sh` which runs over all benchmarks. This shell file can be executed directly after `chmod +x run_tests.sh`, or indirectly via `sh run_tests.sh`. If your machine uses `python3` as an alias for `python` you should edit the `run_tests.sh` such that each call to `python` is replaced with `python3`.

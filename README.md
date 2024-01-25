# machine learning control sem3

App demonstrating capabilities of different optimization methods.

## Requirements

Project uses c++23 (probably can be build with c++20, not tested). Cmake version is also set pretty high (may be can be downgraded).
For gui one will need qt6 installed and configured.

## Implemented

- grey wolf algorithm
- evolution algorithm

## TODO

0. Use static thread_local members or globals instead of thread_local static local variables in functions

1. optimize build time
2. use c++ modules
3. use configurable circles list (added to options but not for )
4. add early stopping to integration (but should not check each iteration: probably would take too long)
5. add python interface
6. add algorithms:
    - particle-swarm optimization
    - hybrid algorithm
    - meta-algorithm for functional target threshold (when consider target achieved)
    - symbolic regression
7. test pontryagin algorithm
8. rewrite gradient descent
9. add different integration solvers
10. better memory deallocation: probably every RepetitiveAllocator should register instance of itself/ deallocate function in some class, which will provide interface to deallocate memory allocated by all types of RepetitiveAllocators
11. add tests
12. add benchmarks
13. improve documentation, generate doxygen

## Miscellaneous

GPL license is used

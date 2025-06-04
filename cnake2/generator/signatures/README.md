# Trainer needs to pre-tokenize all of the prompts
- Fizbuzz
- Polyfit
- 
## Regular dataset of prompts

## Other columns to pipe into reward functions

1. Use python to convert to cython.
  - Python (specifies the interface)
    - Code
    - Tests
    - Entrypoint? An actual run.

2. So cython needs to have a python compatible interface. 
  - Python binding that does the same thing when called.
    - sum(1,5) 1+2+3+4+5 

3. DSPy to generate library of python that is convertible to cython
  - Hashed code
  - Difficulty
  - tens of thousands target size
    - From one or many models? Shouldn't matter. As long as code works!
  - Dataset generate entrypoint, 
    - reward function that it can be used the same way
  - I'll probably try google-flash because ZOOOMIN. 

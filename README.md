# Concurrent Gauss-Jordan Elimination

This project implements a cell-level concurrent algorithm for Gauss-Jordan elimination, designed to solve linear equation systems ($Ax=b$). 

Unlike standard row-level parallelization, this implementation treats operations on single matrix cells as atomic tasks. It utilizes **Mazurkiewicz Trace Theory** to construct a dependency graph (Diekert Graph), calculates **Foata Normal Forms**, and executes tasks in parallel layers using a thread pool.

## Key Features

* **Cell-Level Granularity:** Decomposes matrix operations into atomic Normalization ($N_{k,j}$) and Elimination ($E_{i,k,j}$) tasks.
* **Trace Theory Implementation:**
    * Automatic detection of **Bernstein conditions** (Read/Write conflicts).
    * Construction of the **Dependency Graph** (Diekert Graph).
    * Computation of **Foata Normal Forms** (independent task layers).
* **Concurrent Execution:** Uses Python's `concurrent.futures.ThreadPoolExecutor` to execute Foata classes in parallel.
* **Visualization:** Automatically generates a dependency graph image (`graf_komorkowy.png`) for small matrices ($N \le 5$).
* **Automatic Verification:** Verifies the result against a reference solution with $10^{-4}$ tolerance.

## Prerequisites

* Python 3.x
* Required libraries: `numpy`, `networkx`, `matplotlib`

The script includes an auto-install feature, but you can manually install dependencies via pip:

```bash
pip install numpy networkx matplotlib
```

## How to Test (External Validator / ["Sprawdzarka"](https://github.com/macwozni/Matrices.git))

The program is specifically designed to integrate with the course's external testing system  **["sprawdzarka"](https://github.com/macwozni/Matrices.git)**. It utilizes file-based input/output for verification.

### Step 1: Prepare Input Data (`input_data.txt`)
1.  Open the generator in the **[external validator system](https://github.com/macwozni/Matrices.git)**.
2.  Choose the size of the matrix in the run configuration.
3.  Copy the **Input** content (which usually includes the matrix size $N$, matrix elements $A$, and vector $b$).
4.  Paste the data into **input_data.txt**. The parser expects whitespace-separated tokens (newlines are treated as spaces).

**Example Content:**
```text
3
0.9685595136459395 0.20728467501062664 0.09778766133666406 
0.0 0.5655082553766758 0.7117788719064824 
0.0 0.0 3.2282131743888165E-6 
0.9974402300238512 0.24559462285918898 0.4145022785350656 
```

### Step 2: Prepare Solution (`solution.txt`)
1. Copy the output from the previous step (generator gives input and output).
2. Paste the data into **solution.txt**.

**Example Content:**
```text
3
1.0 0.0 0.0 
0.0 1.0 0.0 
0.0 0.0 1.0 
21624.339500931364 -161610.5442743191 128399.90922022723 
```

And run the program will calculate the result matrix, save it into **my_result.txt** compare it to **solution.txt** file.



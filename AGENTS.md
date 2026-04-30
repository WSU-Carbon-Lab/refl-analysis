I'll help you create a comprehensive best practices guide for Python with Jupyter notebooks, tailored for X-ray data science. Let me first gather some documentation for relevant Python libraries commonly used in data science and scientific computing.


Based on your React/TypeScript AGENT.md file, here's a comprehensive Python/Jupyter Notebook version tailored for X-ray data science workflows:

---

# Python Data Science Best Practices for X-ray Analysis

**Version 1.0.0**  
X-ray Data Science Team - Washington State University   
January 2026

> **Note:**  
> This document is designed for AI agents and LLMs to follow when maintaining,  
> generating, or refactoring Python data science code for X-ray reflectivity (XRR),  
> scattering, diffraction, and spectroscopy analysis. Humans may also find it useful,  
> but guidance here is optimized for automation and consistency by AI-assisted workflows.

---

## Abstract

Comprehensive performance optimization and best practices guide for Python data science projects focused on X-ray analysis (XRR, SAXS, WAXS, XRD, XAS, NEXAFS). Contains 60+ rules across 11 categories, prioritized by impact from critical (eliminating I/O bottlenecks, vectorization) to incremental (advanced patterns). Covers tool selection guidelines (Pandas/Polars/DuckDB, Numba/PyTorch), specialized X-ray libraries (periodictable, PyFAI, PyPXR, pyref) with common pitfalls, spectroscopic analysis principles, and unit/uncertainty management. Each rule includes detailed explanations, real-world examples comparing incorrect vs. correct implementations, and specific impact metrics to guide automated refactoring and code generation for Jupyter notebooks and data pipelines.

---

## Table of Contents

1. [Data Loading & I/O Optimization](#1-data-loading--io-optimization) — **CRITICAL**
   - 1.1 [Use Lazy Loading for Large Datasets](#11-use-lazy-loading-for-large-datasets)
   - 1.2 [Batch File Operations](#12-batch-file-operations)
   - 1.3 [Use Binary Formats Instead of Text](#13-use-binary-formats-instead-of-text)
   - 1.4 [Chunk Large CSV/Text Files](#14-chunk-large-csvtext-files)
   - 1.5 [Cache Expensive Data Processing](#15-cache-expensive-data-processing)
   - 1.6 [Use Memory Mapping for Large Arrays](#16-use-memory-mapping-for-large-arrays)

2. [Vectorization & NumPy Optimization](#2-vectorization--numpy-optimization) — **CRITICAL**
   - 2.1 [Eliminate Python Loops with Vectorization](#21-eliminate-python-loops-with-vectorization)
   - 2.2 [Use NumPy Broadcasting](#22-use-numpy-broadcasting)
   - 2.3 [Avoid Repeated Array Conversions](#23-avoid-repeated-array-conversions)
   - 2.4 [Use In-Place Operations](#24-use-in-place-operations)
   - 2.5 [Preallocate Arrays](#25-preallocate-arrays)
   - 2.6 [Use Appropriate Data Types](#26-use-appropriate-data-types)
   - 2.7 [When to Use Einsum vs Vectorization](#27-when-to-use-einsum-vs-vectorization)

3. [DataFrame Libraries: Pandas, Polars, and DuckDB](#3-dataframe-libraries-pandas-polars-and-duckdb) — **HIGH**
   - 3.1 [Use Categorical Data Types](#31-use-categorical-data-types)
   - 3.2 [Avoid Iterrows and Itertuples](#32-avoid-iterrows-and-itertuples)
   - 3.3 [Use Query for Filtering](#33-use-query-for-filtering)
   - 3.4 [Optimize GroupBy Operations](#34-optimize-groupby-operations)
   - 3.5 [Use Method Chaining](#35-use-method-chaining)
   - 3.6 [Avoid Fragmentation with Concat](#36-avoid-fragmentation-with-concat)
   - 3.7 [Choosing Between Pandas, Polars, and DuckDB](#37-choosing-between-pandas-polars-and-duckdb)

4. [Computational Efficiency](#4-computational-efficiency) — **HIGH**
   - 4.1 [Use Numba for Hot Loops](#41-use-numba-for-hot-loops)
   - 4.2 [Parallel Processing with Multiprocessing](#42-parallel-processing-with-multiprocessing)
   - 4.3 [Use Scipy for Scientific Computing](#43-use-scipy-for-scientific-computing)
   - 4.4 [Cache Expensive Function Calls](#44-cache-expensive-function-calls)
   - 4.5 [Use Generator Expressions for Large Sequences](#45-use-generator-expressions-for-large-sequences)
   - 4.6 [When to Use Numba vs PyTorch](#46-when-to-use-numba-vs-pytorch)

5. [Visualization Performance](#5-visualization-performance) — **MEDIUM**
   - 5.1 [Downsample Data for Plotting](#51-downsample-data-for-plotting)
   - 5.2 [Use Rasterization for Dense Plots](#52-use-rasterization-for-dense-plots)
   - 5.3 [Reuse Figure Objects](#53-reuse-figure-objects)
   - 5.4 [Use Appropriate Plot Types](#54-use-appropriate-plot-types)
   - 5.5 [Batch Plot Updates](#55-batch-plot-updates)

6. [Notebook Organization](#6-notebook-organization) — **MEDIUM**
   - 6.1 [Separate Data Loading from Analysis](#61-separate-data-loading-from-analysis)
   - 6.2 [Use Cell Magic for Profiling](#62-use-cell-magic-for-profiling)
   - 6.3 [Clear Outputs Before Saving](#63-clear-outputs-before-saving)
   - 6.4 [Use Notebook Parameters](#64-use-notebook-parameters)
   - 6.5 [Extract Reusable Code to Modules](#65-extract-reusable-code-to-modules)

7. [Memory Management](#7-memory-management) — **MEDIUM-HIGH**
   - 7.1 [Delete Unused Variables](#71-delete-unused-variables)
   - 7.2 [Use Context Managers for File Operations](#72-use-context-managers-for-file-operations)
   - 7.3 [Monitor Memory Usage](#73-monitor-memory-usage)
   - 7.4 [Use Generators for Large Data Pipelines](#74-use-generators-for-large-data-pipelines)
   - 7.5 [Avoid Copying DataFrames](#75-avoid-copying-dataframes)

8. [Reproducibility & Code Quality](#8-reproducibility--code-quality) — **HIGH**
   - 8.1 [Set Random Seeds](#81-set-random-seeds)
   - 8.2 [Version Pin Dependencies](#82-version-pin-dependencies)
   - 8.3 [Document Data Provenance](#83-document-data-provenance)
   - 8.4 [Use Type Hints](#84-use-type-hints)
   - 8.5 [Validate Input Data](#85-validate-input-data)
   - 8.6 [Use Logging Instead of Print](#86-use-logging-instead-of-print)

9. [X-ray Specialized Libraries](#9-x-ray-specialized-libraries) — **HIGH**
   - 9.1 [Periodic Table Package (periodictable)](#91-periodic-table-package-periodictable)
   - 9.2 [PyFAI for Diffraction Analysis](#92-pyfai-for-diffraction-analysis)
   - 9.3 [PyPXR and pyref for Reflectivity Analysis](#93-pypxr-and-pyref-for-reflectivity-analysis)

10. [Spectroscopic Analysis](#10-spectroscopic-analysis) — **HIGH**
    - 10.1 [NEXAFS/XAS Analysis Principles](#101-nexafsxas-analysis-principles)

11. [Units and Uncertainty Propagation](#11-units-and-uncertainty-propagation) — **MEDIUM-HIGH**
    - 11.1 [Using Pint for Unit Management](#111-using-pint-for-unit-management)
    - 11.2 [Using Uncertainties Package for Error Propagation](#112-using-uncertainties-package-for-error-propagation)

---

## 1. Data Loading & I/O Optimization

**Impact: CRITICAL**

I/O operations are often the primary bottleneck in data science workflows. Loading X-ray data efficiently can reduce analysis time by 10-100×.

### 1.1 Use Lazy Loading for Large Datasets

**Impact: CRITICAL (avoids loading unused data)**

Only load data when needed, not at notebook start. Use lazy evaluation patterns.

**Incorrect: loads all scans at once**

```python
import pandas as pd
import numpy as np

# Loads 100 GB of data immediately
all_scans = []
for i in range(1000):
    all_scans.append(np.load(f'scan_{i}.npy'))

# Only uses first 10
analyzed = [process_scan(s) for s in all_scans[:10]]
```

**Correct: loads data on demand**

```python
from pathlib import Path
import numpy as np

def load_scan(scan_id: int) -> np.ndarray:
    """Lazy loader for scan data."""
    return np.load(f'scan_{scan_id}.npy')

# Only loads 10 scans
analyzed = [process_scan(load_scan(i)) for i in range(10)]
```

**Alternative: use Dask for lazy dataframes**

```python
import dask.dataframe as dd

# Lazy loading - doesn't read files until compute()
df = dd.read_csv('xrr_data/*.csv')
result = df[df['intensity'] > 1000].compute()  # Only now reads data
```

### 1.2 Batch File Operations

**Impact: HIGH (reduces filesystem overhead)**

Group file operations to minimize filesystem calls. Opening 1000 files individually is much slower than batching.

**Incorrect: opens files one at a time**

```python
data = []
for i in range(1000):
    with open(f'scan_{i}.txt') as f:
        data.append(f.read())
```

**Correct: use concurrent file operations**

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def read_file(filepath: Path) -> str:
    return filepath.read_text()

files = list(Path('.').glob('scan_*.txt'))

with ThreadPoolExecutor(max_workers=4) as executor:
    data = list(executor.map(read_file, files))
```

### 1.3 Use Binary Formats Instead of Text

**Impact: CRITICAL (10-100× faster loading, 5-10× smaller files)**

Binary formats (HDF5, NPY, Parquet) are dramatically faster to read/write than text (CSV, TXT).

**Incorrect: saves as CSV**

```python
import pandas as pd

df = pd.DataFrame({'q': q_values, 'intensity': intensities})
df.to_csv('xrr_data.csv', index=False)  # Slow, large file
```

**Correct: use Parquet or HDF5**

```python
import pandas as pd

# Parquet: ~5× smaller, ~10× faster
df.to_parquet('xrr_data.parquet')
df_loaded = pd.read_parquet('xrr_data.parquet')

# Or HDF5 for NumPy arrays
import h5py
with h5py.File('xrr_data.h5', 'w') as f:
    f['q'] = q_values
    f['intensity'] = intensities
```

**For NumPy arrays:**

```python
# NPY: native NumPy binary format
np.save('reflectivity.npy', data)  # Fast
loaded = np.load('reflectivity.npy')  # Fast

# NPZ: compressed multiple arrays
np.savez_compressed('scan_data.npz', 
                     q=q_values, 
                     intensity=intensities,
                     errors=errors)
data = np.load('scan_data.npz')
q = data['q']
```

### 1.4 Chunk Large CSV/Text Files

**Impact: HIGH (enables processing of files larger than RAM)**

Process large files in chunks instead of loading entirely into memory.

**Incorrect: loads entire 50 GB file**

```python
df = pd.read_csv('large_xrd_data.csv')  # OOM error
filtered = df[df['intensity'] > threshold]
```

**Correct: process in chunks**

```python
import pandas as pd

chunks = []
for chunk in pd.read_csv('large_xrd_data.csv', chunksize=100_000):
    filtered_chunk = chunk[chunk['intensity'] > threshold]
    chunks.append(filtered_chunk)

result = pd.concat(chunks, ignore_index=True)
```

**Alternative: use Dask**

```python
import dask.dataframe as dd

# Automatically chunks file
df = dd.read_csv('large_xrd_data.csv')
result = df[df['intensity'] > threshold].compute()
```

### 1.5 Cache Expensive Data Processing

**Impact: HIGH (avoid recomputation)**

Cache results of expensive data transformations to avoid recomputation.

**Incorrect: reprocesses on every run**

```python
# Cell 1: Load and process (takes 10 minutes)
raw_data = load_raw_scans()
processed = apply_corrections(raw_data)  # Expensive
normalized = normalize_intensity(processed)  # Expensive

# Cell 2: Re-run analysis
# Every time you rerun, reprocesses from scratch
```

**Correct: cache intermediate results**

```python
from pathlib import Path
import pickle

cache_file = Path('cache/processed_data.pkl')

if cache_file.exists():
    print("Loading from cache...")
    with open(cache_file, 'rb') as f:
        normalized = pickle.load(f)
else:
    print("Processing data...")
    raw_data = load_raw_scans()
    processed = apply_corrections(raw_data)
    normalized = normalize_intensity(processed)
    
    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(normalized, f)
```

**Alternative: use joblib for caching**

```python
from joblib import Memory

memory = Memory('cache', verbose=0)

@memory.cache
def process_scans(scan_ids):
    """Automatically cached based on inputs."""
    raw_data = load_raw_scans(scan_ids)
    processed = apply_corrections(raw_data)
    return normalize_intensity(processed)

# First call: computes and caches
data = process_scans([1, 2, 3])

# Second call with same inputs: loads from cache
data = process_scans([1, 2, 3])  # Instant
```

### 1.6 Use Memory Mapping for Large Arrays

**Impact: MEDIUM-HIGH (enables working with arrays larger than RAM)**

Memory-mapped arrays allow working with data larger than available RAM.

**Incorrect: loads entire array into memory**

```python
# 50 GB array - may cause OOM
data = np.load('large_detector_image.npy')
roi = data[1000:2000, 1000:2000]  # Only need small region
```

**Correct: memory-map the file**

```python
# Only loads what you access
data = np.load('large_detector_image.npy', mmap_mode='r')
roi = data[1000:2000, 1000:2000]  # Only loads this region
```

**Creating memory-mapped arrays:**

```python
# Create large array on disk
shape = (100000, 100000)
dtype = np.float32
mmap_array = np.memmap('temp_array.dat', dtype=dtype, mode='w+', shape=shape)

# Use like normal array
mmap_array[0:1000, 0:1000] = np.random.random((1000, 1000))
mmap_array.flush()  # Write to disk
```

---

## 2. Vectorization & NumPy Optimization

**Impact: CRITICAL**

Vectorized NumPy operations can be 100-1000× faster than Python loops. This is the single most important optimization for numerical computing.

### 2.1 Eliminate Python Loops with Vectorization

**Impact: CRITICAL (100-1000× speedup)**

Replace Python loops with vectorized NumPy operations.

**Incorrect: Python loop (very slow)**

```python
import numpy as np

q = np.linspace(0.01, 0.5, 10000)
intensity = np.zeros_like(q)

# ~1000× slower than vectorized
for i in range(len(q)):
    intensity[i] = calculate_reflectivity(q[i])
```

**Correct: vectorized operation**

```python
import numpy as np

q = np.linspace(0.01, 0.5, 10000)

# ~1000× faster
intensity = calculate_reflectivity(q)  # Vectorized function
```

**Example: Background subtraction**

```python
# Incorrect: loop
background_subtracted = np.zeros_like(intensity)
for i in range(len(intensity)):
    background_subtracted[i] = intensity[i] - background[i]

# Correct: vectorized
background_subtracted = intensity - background  # Single operation
```

**Example: Apply correction factor**

```python
# Incorrect
corrected = np.zeros_like(data)
for i in range(len(data)):
    corrected[i] = data[i] * correction_factors[i]

# Correct
corrected = data * correction_factors
```

### 2.2 Use NumPy Broadcasting

**Impact: HIGH (eliminates loops, clearer code)**

NumPy broadcasting automatically handles operations between arrays of different shapes.

**Incorrect: explicit loops**

```python
# data: (1000, 512, 512) - multiple detector images
# dark: (512, 512) - dark current correction

corrected = np.zeros_like(data)
for i in range(data.shape[0]):
    corrected[i] = data[i] - dark
```

**Correct: broadcasting**

```python
# Automatically applies dark to each image
corrected = data - dark  # (1000, 512, 512) - (512, 512) works!
```

**Example: Normalize each row by different value**

```python
# matrix: (1000, 100)
# row_sums: (1000,)

# Incorrect
normalized = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    normalized[i] = matrix[i] / row_sums[i]

# Correct: add dimension for broadcasting
normalized = matrix / row_sums[:, np.newaxis]  # (1000, 100) / (1000, 1)
```

### 2.3 Avoid Repeated Array Conversions

**Impact: MEDIUM (eliminates overhead)**

Converting between lists and arrays repeatedly is slow. Work with NumPy arrays throughout.

**Incorrect: repeated conversions**

```python
data_list = [1, 2, 3, 4, 5]

for i in range(100):
    arr = np.array(data_list)  # Conversion every iteration
    result = arr * 2
    data_list = result.tolist()  # Conversion back
```

**Correct: work with arrays**

```python
data = np.array([1, 2, 3, 4, 5])

for i in range(100):
    data = data * 2  # No conversions
```

### 2.4 Use In-Place Operations

**Impact: MEDIUM (reduces memory allocation)**

In-place operations modify arrays without creating copies, saving memory and time.

**Incorrect: creates new arrays**

```python
data = data + 100  # Creates new array
data = data * 2    # Creates another new array
```

**Correct: in-place operations**

```python
data += 100  # Modifies in place
data *= 2    # Modifies in place
```

**NumPy functions with out parameter:**

```python
# Incorrect: creates temporary arrays
result = np.sin(x) + np.cos(y)

# Correct: reuse arrays
np.sin(x, out=x)  # Overwrites x
np.cos(y, out=y)  # Overwrites y
result = x + y
```

### 2.5 Preallocate Arrays

**Impact: MEDIUM (avoids repeated allocation)**

Preallocate arrays instead of growing them dynamically.

**Incorrect: grows array dynamically**

```python
result = []
for i in range(10000):
    result.append(process(i))
result = np.array(result)  # Conversion at end
```

**Correct: preallocate**

```python
result = np.empty(10000)
for i in range(10000):
    result[i] = process(i)
```

**Better: use vectorization**

```python
indices = np.arange(10000)
result = process(indices)  # Vectorized
```

### 2.6 Use Appropriate Data Types

**Impact: MEDIUM (reduces memory usage)**

Choose the smallest dtype that fits your data to reduce memory usage and improve cache performance.

**Incorrect: uses default float64**

```python
# float64 uses 8 bytes per element
data = np.array([1.0, 2.0, 3.0])  # Default: float64
large_array = np.random.random((10000, 10000))  # 800 MB
```

**Correct: use appropriate precision**

```python
# float32 uses 4 bytes per element (sufficient for most X-ray data)
data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
large_array = np.random.random((10000, 10000)).astype(np.float32)  # 400 MB

# For integer indices
indices = np.arange(1000, dtype=np.int32)  # vs int64 default
```

### 2.7 When to Use Einsum vs Vectorization

**Impact: MEDIUM-HIGH (clarity and performance for tensor operations)**

Use `np.einsum` for complex tensor contractions and Einstein summation notation. Use standard vectorization for simpler element-wise operations.

**When to use einsum:**

```python
import numpy as np

# Matrix multiplication
A = np.random.random((100, 50))
B = np.random.random((50, 30))
C = np.einsum('ij,jk->ik', A, B)  # Equivalent to A @ B, but more flexible

# Batch matrix multiplication
A = np.random.random((10, 100, 50))
B = np.random.random((10, 50, 30))
C = np.einsum('bij,bjk->bik', A, B)  # Each batch multiplied independently

# Tensor contraction (trace)
tensor = np.random.random((5, 5, 5, 5))
trace = np.einsum('iijj->', tensor)  # Sum over i=j

# Complex multi-index operations
A = np.random.random((10, 20, 30))
B = np.random.random((20, 40))
result = np.einsum('ijk,jl->ikl', A, B)  # Contract over j
```

**When to use standard vectorization:**

```python
# Simple element-wise operations - use standard vectorization
result = data * weights  # Not einsum
result = np.sin(x) + np.cos(y)  # Not einsum

# Simple reductions - use standard functions
sum_result = np.sum(data, axis=0)  # Not einsum
mean_result = np.mean(data, axis=1)  # Not einsum

# Broadcasting operations - use standard operations
result = array_2d + array_1d[:, np.newaxis]  # Not einsum
```

**Performance comparison:**

```python
# Einsum can be faster for complex contractions
# For simple operations, standard vectorization is often faster

# Einsum: optimized for complex tensor operations
A = np.random.random((100, 100))
B = np.random.random((100, 100))
# Einsum may use BLAS optimizations for matrix multiplication
C = np.einsum('ij,jk->ik', A, B)

# But for simple operations, standard is faster:
result = A * B  # Faster than einsum('ij,ij->ij', A, B)
```

**Use einsum when:**
- Performing tensor contractions (summing over multiple indices)
- Batch operations on multi-dimensional arrays
- Need clear notation for complex operations
- Contracting over non-adjacent dimensions

**Use standard vectorization when:**
- Simple element-wise operations
- Basic broadcasting operations
- Standard reductions (sum, mean, etc.)
- Clearer and more readable code

---

## 3. DataFrame Libraries: Pandas, Polars, and DuckDB

**Impact: HIGH**

Pandas operations can be slow if used incorrectly. These patterns dramatically improve performance.

### 3.1 Use Categorical Data Types

**Impact: HIGH (50-80% memory reduction for repeated strings)**

Convert string columns with repeated values to categorical dtype.

**Incorrect: stores as object (strings)**

```python
import pandas as pd

df = pd.DataFrame({
    'sample_id': ['sample_A'] * 10000 + ['sample_B'] * 10000,
    'measurement': range(20000)
})
# sample_id uses ~1.5 MB as object
```

**Correct: use categorical**

```python
df = pd.DataFrame({
    'sample_id': pd.Categorical(['sample_A'] * 10000 + ['sample_B'] * 10000),
    'measurement': range(20000)
})
# sample_id uses ~0.2 MB as categorical (7× smaller)
```

**Convert existing column:**

```python
df['sample_id'] = df['sample_id'].astype('category')
```

### 3.2 Avoid Iterrows and Itertuples

**Impact: CRITICAL (100-1000× slower than vectorized)**

Never use `iterrows()` or `itertuples()` for computation. Use vectorized operations or `.apply()`.

**Incorrect: iterrows (extremely slow)**

```python
# 1000× slower than vectorized
for idx, row in df.iterrows():
    df.at[idx, 'normalized'] = row['intensity'] / row['incident']
```

**Correct: vectorized**

```python
df['normalized'] = df['intensity'] / df['incident']
```

**If vectorization not possible, use .apply():**

```python
# Still faster than iterrows
df['result'] = df.apply(lambda row: complex_function(row['a'], row['b']), axis=1)
```

**Better: use NumPy directly**

```python
df['result'] = complex_function(df['a'].values, df['b'].values)
```

### 3.3 Use Query for Filtering

**Impact: MEDIUM (cleaner, sometimes faster)**

Use `.query()` for complex filtering conditions.

**Incorrect: verbose boolean indexing**

```python
filtered = df[(df['q'] > 0.1) & (df['q'] < 0.5) & (df['intensity'] > 100)]
```

**Correct: query string**

```python
filtered = df.query('0.1 < q < 0.5 and intensity > 100')
```

**Performance benefit with large DataFrames:**

```python
# For large DataFrames, query can be faster due to numexpr backend
large_df.query('intensity > @threshold and sample == "A"')
```

### 3.4 Optimize GroupBy Operations

**Impact: HIGH (reduces memory usage)**

Use efficient aggregation methods and avoid creating unnecessary intermediate DataFrames.

**Incorrect: multiple group operations**

```python
# Groups data multiple times
mean_intensity = df.groupby('sample')['intensity'].mean()
max_q = df.groupby('sample')['q'].max()
count = df.groupby('sample').size()
```

**Correct: single groupby with agg**

```python
result = df.groupby('sample').agg({
    'intensity': 'mean',
    'q': 'max',
    'sample': 'size'  # Count
}).rename(columns={'sample': 'count'})
```

**Use transform for broadcasting back:**

```python
# Add group mean to each row
df['group_mean'] = df.groupby('sample')['intensity'].transform('mean')
```

### 3.5 Use Method Chaining

**Impact: LOW (cleaner code, easier optimization)**

Chain operations to avoid intermediate variables and enable optimization.

**Incorrect: multiple intermediate variables**

```python
df1 = df[df['intensity'] > 0]
df2 = df1[df1['q'] > 0.1]
df3 = df2.sort_values('q')
result = df3.reset_index(drop=True)
```

**Correct: method chaining**

```python
result = (df
    .query('intensity > 0 and q > 0.1')
    .sort_values('q')
    .reset_index(drop=True)
)
```

### 3.6 Avoid Fragmentation with Concat

**Impact: MEDIUM (prevents DataFrame fragmentation)**

Collect DataFrames in a list and concat once instead of repeatedly appending.

**Incorrect: repeated concat (causes fragmentation)**

```python
result = pd.DataFrame()
for file in files:
    data = pd.read_csv(file)
    result = pd.concat([result, data])  # Slow, fragments memory
```

**Correct: collect then concat once**

```python
dataframes = []
for file in files:
    dataframes.append(pd.read_csv(file))

result = pd.concat(dataframes, ignore_index=True)
```

### 3.7 Choosing Between Pandas, Polars, and DuckDB

**Impact: HIGH (performance and workflow optimization)**

Select the appropriate DataFrame library based on data size, operation complexity, and workflow requirements.

**Use Pandas when:**
- Data fits comfortably in memory (< 10 GB typically)
- Interoperability with existing code is important
- Rich ecosystem of compatible libraries needed
- Interactive analysis in Jupyter notebooks
- Need mature API with extensive functionality

```python
import pandas as pd

df = pd.read_parquet('xrr_data.parquet')
result = (df
    .query('q > 0.1 and intensity > 0')
    .groupby('sample')
    .agg({'intensity': 'mean', 'q': 'max'})
)
```

**Use Polars when:**
- Data is large (10+ GB) or near memory limits
- Need high-performance aggregations and joins
- Lazy evaluation for complex query optimization
- Want Pandas-like API with better performance
- Working with wide DataFrames (many columns)

```python
import polars as pl

df = pl.read_parquet('xrr_data.parquet')
result = (df
    .lazy()
    .filter((pl.col('q') > 0.1) & (pl.col('intensity') > 0))
    .group_by('sample')
    .agg([
        pl.col('intensity').mean(),
        pl.col('q').max()
    ])
    .collect()
)
```

**Use DuckDB when:**
- Very large datasets (100+ GB or larger than RAM)
- Need SQL-like query interface
- Complex analytical queries with aggregations
- Want to query Parquet/CSV directly without loading
- Need to join multiple large files efficiently

```python
import duckdb

conn = duckdb.connect()
result = conn.execute("""
    SELECT 
        sample,
        AVG(intensity) as mean_intensity,
        MAX(q) as max_q
    FROM read_parquet('xrr_data.parquet')
    WHERE q > 0.1 AND intensity > 0
    GROUP BY sample
""").df()
```

**Performance comparison guidelines:**

```python
# Small data (< 1 GB): Pandas is fastest due to overhead
df_small = pd.read_csv('small_data.csv')  # Use Pandas

# Medium data (1-10 GB): Polars often faster
df_medium = pl.read_parquet('medium_data.parquet')  # Use Polars

# Large data (10+ GB): DuckDB for querying, Polars for processing
conn = duckdb.connect()
result = conn.execute("SELECT * FROM read_parquet('large_data.parquet') WHERE q > 0.1").df()
```

**Interoperability:**

```python
# Convert between libraries efficiently
df_pandas = result.to_pandas()  # Polars to Pandas
df_polars = pl.from_pandas(df_pandas)  # Pandas to Polars

# DuckDB can return Polars directly
result_polars = conn.execute("SELECT * FROM df").pl()
```

**Common pitfalls:**

```python
# Incorrect: loading huge file with Pandas
df = pd.read_csv('100gb_data.csv')  # OOM error

# Correct: use DuckDB to query without loading
result = conn.execute("SELECT * FROM read_csv('100gb_data.csv') WHERE q > 0.1").df()

# Incorrect: not using lazy evaluation in Polars
df = pl.read_parquet('large.parquet')
filtered = df.filter(pl.col('intensity') > 1000)  # Eager, may be slow

# Correct: use lazy evaluation
result = (pl.scan_parquet('large.parquet')
    .filter(pl.col('intensity') > 1000)
    .collect()
)
```

---

## 4. Computational Efficiency

**Impact: HIGH**

Advanced computational techniques for performance-critical code.

### 4.1 Use Numba for Hot Loops

**Impact: CRITICAL (10-100× speedup)**

Use Numba JIT compilation for numerical loops that can't be vectorized.

**Incorrect: pure Python loop**

```python
def calculate_reflectivity_slow(q_values, layers):
    """Very slow - pure Python."""
    result = np.zeros(len(q_values))
    for i, q in enumerate(q_values):
        r = 0.0
        for layer in layers:
            r += layer['sld'] * np.exp(-q**2 * layer['roughness']**2)
        result[i] = abs(r)**2
    return result
```

**Correct: Numba JIT**

```python
from numba import jit

@jit(nopython=True)
def calculate_reflectivity_fast(q_values, sld_array, roughness_array):
    """~100× faster with Numba."""
    result = np.zeros(len(q_values))
    for i, q in enumerate(q_values):
        r = 0.0
        for j in range(len(sld_array)):
            r += sld_array[j] * np.exp(-q**2 * roughness_array[j]**2)
        result[i] = abs(r)**2
    return result

# Convert to NumPy arrays for Numba
sld_array = np.array([layer['sld'] for layer in layers])
roughness_array = np.array([layer['roughness'] for layer in layers])

result = calculate_reflectivity_fast(q_values, sld_array, roughness_array)
```

**Parallel Numba:**

```python
@jit(nopython=True, parallel=True)
def process_parallel(data):
    """Uses multiple cores."""
    result = np.zeros_like(data)
    for i in prange(len(data)):
        result[i] = expensive_calculation(data[i])
    return result
```

### 4.2 Parallel Processing with Multiprocessing

**Impact: HIGH (N× speedup on N cores)**

Use multiprocessing for embarrassingly parallel tasks (independent computations).

**Incorrect: sequential processing**

```python
results = []
for scan_file in scan_files:  # 100 files
    results.append(process_scan(scan_file))  # Takes 10s each = 1000s total
```

**Correct: parallel with multiprocessing**

```python
from multiprocessing import Pool

def process_scan(filepath):
    """Process single scan file."""
    data = np.load(filepath)
    return analyze_data(data)

if __name__ == '__main__':
    with Pool(processes=8) as pool:
        results = pool.map(process_scan, scan_files)
    # Takes ~125s on 8 cores (8× faster)
```

**Alternative: concurrent.futures**

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_scan, scan_files))
```

**For I/O-bound tasks, use ThreadPoolExecutor:**

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=16) as executor:
    results = list(executor.map(download_file, urls))
```

### 4.3 Use Scipy for Scientific Computing

**Impact: HIGH (optimized C/Fortran implementations)**

Use SciPy's optimized functions instead of implementing from scratch.

**Incorrect: manual implementation**

```python
# Slow, potentially incorrect
def manual_gaussian_filter(data, sigma):
    # Complex manual implementation...
    pass
```

**Correct: use SciPy**

```python
from scipy.ndimage import gaussian_filter

# Fast, tested, correct
filtered = gaussian_filter(data, sigma=2.0)
```

**Common SciPy use cases:**

```python
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# Curve fitting
params, _ = curve_fit(model_function, q_values, intensities)

# Peak finding
peaks, _ = find_peaks(spectrum, height=threshold)

# Interpolation
f = interp1d(q_values, intensities, kind='cubic')
interpolated = f(new_q_values)

# Integration
area = simpson(intensities, x=q_values)
```

### 4.4 Cache Expensive Function Calls

**Impact: HIGH (avoids recomputation)**

Use `functools.lru_cache` for expensive pure functions.

**Incorrect: recomputes every time**

```python
def expensive_model(param1, param2, param3):
    # Takes 10 seconds
    return complex_calculation(param1, param2, param3)

# Called 100 times with same params = 1000 seconds
for i in range(100):
    result = expensive_model(1.5, 2.3, 0.8)
```

**Correct: cache results**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_model(param1, param2, param3):
    return complex_calculation(param1, param2, param3)

# First call: 10 seconds
# Next 99 calls: instant (cached)
for i in range(100):
    result = expensive_model(1.5, 2.3, 0.8)
```

**For unhashable arguments, use joblib:**

```python
from joblib import Memory

memory = Memory('cache', verbose=0)

@memory.cache
def process_array(arr, params):
    """Can cache NumPy arrays."""
    return expensive_computation(arr, params)
```

### 4.5 Use Generator Expressions for Large Sequences

**Impact: MEDIUM (reduces memory usage)**

Use generators instead of list comprehensions when processing large sequences.

**Incorrect: creates large intermediate list**

```python
# Creates list of 1 million elements in memory
total = sum([process(x) for x in range(1_000_000)])
```

**Correct: generator expression**

```python
# Processes one at a time
total = sum(process(x) for x in range(1_000_000))
```

**Generator functions:**

```python
def read_large_file(filepath):
    """Yields lines one at a time."""
    with open(filepath) as f:
        for line in f:
            yield parse_line(line)

# Memory efficient - processes one line at a time
for data in read_large_file('large_dataset.txt'):
    process(data)
```

### 4.6 When to Use Numba vs PyTorch

**Impact: HIGH (correct tool selection for performance)**

Choose Numba for CPU-accelerated Python functions with NumPy. Choose PyTorch for GPU acceleration, deep learning, and automatic differentiation.

**Use Numba when:**
- Need to accelerate pure Python/NumPy functions
- Working with CPU-bound numerical computations
- Minimal code changes required (decorator-based)
- Need to optimize existing NumPy-based code
- Don't need GPU acceleration or automatic differentiation
- Functions are relatively simple numerical operations

```python
from numba import jit
import numpy as np

@jit(nopython=True)
def calculate_reflectivity(q_values, sld_array, roughness_array):
    """CPU-accelerated with Numba."""
    result = np.zeros(len(q_values))
    for i, q in enumerate(q_values):
        r = 0.0
        for j in range(len(sld_array)):
            r += sld_array[j] * np.exp(-q**2 * roughness_array[j]**2)
        result[i] = abs(r)**2
    return result

result = calculate_reflectivity(q_values, sld_array, roughness_array)
```

**Use PyTorch when:**
- Need GPU acceleration (CUDA)
- Performing deep learning or neural network operations
- Need automatic differentiation (gradients)
- Working with batched tensor operations
- Need to leverage GPU memory for large computations
- Want to use optimized GPU kernels

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

q_tensor = torch.tensor(q_values, device=device, dtype=torch.float32)
sld_tensor = torch.tensor(sld_array, device=device, dtype=torch.float32)

# GPU-accelerated operations
result = torch.sum(sld_tensor[:, None] * torch.exp(-q_tensor**2 * sld_tensor**2), dim=0)

# With automatic differentiation
q_tensor.requires_grad = True
result = compute_reflectivity(q_tensor, sld_tensor)
loss = criterion(result, target)
loss.backward()
gradients = q_tensor.grad
```

**Performance considerations:**

```python
# Small arrays (< 1000 elements): NumPy/Numba often faster due to overhead
# Medium arrays (1K-100K): Numba CPU can be competitive with PyTorch CPU
# Large arrays (100K+): PyTorch GPU significantly faster if available

# CPU-bound single-threaded: Numba is excellent
@jit(nopython=True)
def cpu_computation(data):
    return expensive_calculation(data)

# CPU-bound parallel: Numba parallel
from numba import prange

@jit(nopython=True, parallel=True)
def parallel_computation(data):
    result = np.zeros_like(data)
    for i in prange(len(data)):
        result[i] = expensive_calculation(data[i])
    return result

# GPU-accelerated: PyTorch
device = torch.device('cuda')
data_gpu = torch.tensor(data, device=device)
result_gpu = compute_on_gpu(data_gpu)
result = result_gpu.cpu().numpy()
```

**Common pitfalls:**

```python
# Incorrect: using PyTorch for simple CPU operations without GPU
# Overhead of PyTorch can be slower than NumPy for small operations
data_torch = torch.tensor(data)
result = torch.sum(data_torch).item()  # Slower than np.sum(data) for small arrays

# Correct: use NumPy/Numba for simple CPU operations
result = np.sum(data)

# Incorrect: using Numba when GPU acceleration is needed
@jit(nopython=True)
def large_computation(data):
    # Won't use GPU, limited to CPU cores
    return process_large_data(data)

# Correct: use PyTorch for GPU acceleration
data_gpu = torch.tensor(data, device='cuda')
result = process_on_gpu(data_gpu)

# Incorrect: mixing NumPy and PyTorch in hot loops
# Frequent conversions are expensive
for i in range(1000):
    data_torch = torch.tensor(data_numpy)
    result_torch = compute(data_torch)
    data_numpy = result_torch.numpy()

# Correct: stay in one ecosystem
data_torch = torch.tensor(data_numpy)
for i in range(1000):
    data_torch = compute(data_torch)
result_numpy = data_torch.numpy()
```

**Hybrid approach:**

```python
# Use Numba for CPU preprocessing, PyTorch for GPU computation
from numba import jit
import torch

@jit(nopython=True)
def preprocess_cpu(data):
    """Fast CPU preprocessing."""
    return normalize_and_filter(data)

# Preprocess on CPU
processed = preprocess_cpu(raw_data)

# Compute on GPU
data_gpu = torch.tensor(processed, device='cuda')
result_gpu = gpu_heavy_computation(data_gpu)
```

---

## 5. Visualization Performance

**Impact: MEDIUM**

Plotting large datasets can be slow. These optimizations maintain interactivity.

### 5.1 Downsample Data for Plotting

**Impact: HIGH (100× faster rendering)**

Plotting millions of points is slow and unnecessary. Downsample to screen resolution.

**Incorrect: plots 10 million points**

```python
import matplotlib.pyplot as plt

# 10 million data points - very slow to render
plt.plot(q_values, intensities)  # Hangs for minutes
```

**Correct: downsample to screen resolution**

```python
def downsample(x, y, num_points=2000):
    """Downsample to num_points while preserving shape."""
    if len(x) <= num_points:
        return x, y
    
    indices = np.linspace(0, len(x) - 1, num_points, dtype=int)
    return x[indices], y[indices]

# Plot ~2000 points (matches typical screen resolution)
x_plot, y_plot = downsample(q_values, intensities)
plt.plot(x_plot, y_plot)  # Renders instantly
```

**Alternative: use datashader for very large datasets**

```python
import datashader as ds
import datashader.transfer_functions as tf

cvs = ds.Canvas(plot_width=800, plot_height=400)
agg = cvs.line(df, 'q', 'intensity')
img = tf.shade(agg)
```

### 5.2 Use Rasterization for Dense Plots

**Impact: HIGH (smaller file sizes, faster rendering)**

Rasterize plots with many data points to avoid huge vector files.

**Incorrect: creates 50 MB PDF**

```python
fig, ax = plt.subplots()
ax.scatter(x, y, s=1)  # 1 million points
plt.savefig('plot.pdf')  # 50 MB vector file, slow to open
```

**Correct: rasterize the plot**

```python
fig, ax = plt.subplots()
ax.scatter(x, y, s=1, rasterized=True)
plt.savefig('plot.pdf', dpi=300)  # 1 MB file, fast to open
```

**Rasterize only data, keep labels as vectors:**

```python
fig, ax = plt.subplots()
ax.plot(x, y, rasterized=True)  # Data rasterized
ax.set_xlabel('Q (Å⁻¹)')  # Labels stay vector
ax.set_ylabel('Intensity')
plt.savefig('plot.pdf', dpi=300)
```

### 5.3 Reuse Figure Objects

**Impact: MEDIUM (avoids recreation overhead)**

Reuse figure and axes objects instead of creating new ones.

**Incorrect: creates new figure every iteration**

```python
for i in range(100):
    fig, ax = plt.subplots()  # Slow
    ax.plot(data[i])
    plt.close()
```

**Correct: reuse figure**

```python
fig, ax = plt.subplots()
for i in range(100):
    ax.clear()
    ax.plot(data[i])
    fig.canvas.draw()
```

### 5.4 Use Appropriate Plot Types

**Impact: MEDIUM (better performance and clarity)**

Choose the right plot type for your data density.

**Incorrect: scatter plot for dense data**

```python
# Slow, overlapping points
plt.scatter(x, y, s=1)  # 1 million points
```

**Correct: use hexbin or 2D histogram**

```python
# Fast, shows density
plt.hexbin(x, y, gridsize=50, cmap='viridis')
plt.colorbar(label='Count')

# Or 2D histogram
plt.hist2d(x, y, bins=100, cmap='viridis')
```

### 5.5 Batch Plot Updates

**Impact: MEDIUM (reduces redraws)**

Update multiple plot elements before redrawing.

**Incorrect: redraws after each update**

```python
line, = ax.plot(x, y)
for i in range(100):
    line.set_ydata(new_data[i])  # Redraws each time
    fig.canvas.draw()
```

**Correct: batch updates**

```python
line, = ax.plot(x, y)
with plt.ion():  # Interactive mode
    for i in range(100):
        line.set_ydata(new_data[i])
    fig.canvas.draw()  # Single redraw at end
```

---

## 6. Notebook Organization

**Impact: MEDIUM**

Well-organized notebooks are easier to maintain, debug, and reproduce.

### 6.1 Separate Data Loading from Analysis

**Impact: MEDIUM (faster iteration)**

Load data once in early cells, analyze in later cells to avoid reloading on every run.

**Incorrect: loads data in analysis cells**

```python
# Cell 5: Analysis A
data = load_data()  # Loads data
result_a = analyze_a(data)

# Cell 6: Analysis B  
data = load_data()  # Reloads same data!
result_b = analyze_b(data)
```

**Correct: load once**

```python
# Cell 1: Data Loading
data = load_data()

# Cell 5: Analysis A
result_a = analyze_a(data)

# Cell 6: Analysis B
result_b = analyze_b(data)
```

### 6.2 Use Cell Magic for Profiling

**Impact: HIGH (identify bottlenecks)**

Use IPython magic commands to profile and time code.

**Time a cell:**

```python
%%time
# Times execution of entire cell
data = load_large_dataset()
processed = process_data(data)
```

**Profile line by line:**

```python
%load_ext line_profiler
%lprun -f process_data process_data(data)
```

**Memory profiling:**

```python
%load_ext memory_profiler
%memit process_data(data)
```

**Interactive profiling:**

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = expensive_function(data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 6.3 Clear Outputs Before Saving

**Impact: LOW (reduces repository size)**

Clear cell outputs before committing notebooks to version control.

**Automated clearing:**

```bash
# In terminal
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

**Pre-commit hook:**

```bash
# .git/hooks/pre-commit
#!/bin/bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
git add notebooks/*.ipynb
```

### 6.4 Use Notebook Parameters

**Impact: MEDIUM (enables automation)**

Use papermill to parameterize notebooks for batch processing.

**Parameterized notebook:**

```python
# Cell 1: Parameters (tagged with 'parameters')
scan_id = 1
threshold = 100
output_dir = './results'
```

**Run with different parameters:**

```bash
papermill notebook.ipynb output.ipynb -p scan_id 42 -p threshold 150
```

**Python API:**

```python
import papermill as pm

for scan_id in range(1, 100):
    pm.execute_notebook(
        'template.ipynb',
        f'output_{scan_id}.ipynb',
        parameters={'scan_id': scan_id}
    )
```

### 6.5 Extract Reusable Code to Modules

**Impact: HIGH (maintainability, testing)**

Extract reusable functions to Python modules instead of duplicating in notebooks.

**Incorrect: duplicate code across notebooks**

```python
# notebook_1.ipynb
def load_xrr_data(filepath):
    # Complex loading logic...
    pass

# notebook_2.ipynb  
def load_xrr_data(filepath):
    # Same code duplicated
    pass
```

**Correct: extract to module**

```python
# xrr_utils.py
def load_xrr_data(filepath):
    """Load and parse XRR data."""
    # Logic here
    return data

# notebook_1.ipynb
from xrr_utils import load_xrr_data
data = load_xrr_data('scan.txt')

# notebook_2.ipynb
from xrr_utils import load_xrr_data
data = load_xrr_data('scan.txt')
```

**Auto-reload modules during development:**

```python
# At top of notebook
%load_ext autoreload
%autoreload 2

# Now edits to xrr_utils.py are automatically reloaded
```

---

## 7. Memory Management

**Impact: MEDIUM-HIGH**

Managing memory prevents crashes and improves performance.

### 7.1 Delete Unused Variables

**Impact: MEDIUM (frees memory)**

Delete large variables when no longer needed.

**Incorrect: keeps unused variables**

```python
raw_data = load_large_file()  # 10 GB
processed = process(raw_data)

# raw_data still in memory (20 GB total)
final_result = finalize(processed)
```

**Correct: delete when done**

```python
raw_data = load_large_file()  # 10 GB
processed = process(raw_data)
del raw_data  # Frees 10 GB

final_result = finalize(processed)
```

**Force garbage collection:**

```python
import gc

del raw_data
gc.collect()  # Force immediate cleanup
```

### 7.2 Use Context Managers for File Operations

**Impact: MEDIUM (prevents resource leaks)**

Always use `with` statements for file operations.

**Incorrect: manual file handling**

```python
f = open('data.txt')
data = f.read()
f.close()  # May not execute if error occurs
```

**Correct: context manager**

```python
with open('data.txt') as f:
    data = f.read()
# File automatically closed, even if exception occurs
```

**Multiple files:**

```python
with open('input.txt') as f_in, open('output.txt', 'w') as f_out:
    data = f_in.read()
    f_out.write(process(data))
```

### 7.3 Monitor Memory Usage

**Impact: HIGH (prevents OOM crashes)**

Monitor memory usage to prevent out-of-memory errors.

**Check current memory:**

```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Current memory: {memory_mb:.1f} MB")
```

**Track memory during execution:**

```python
def memory_usage_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print(f"Start: {memory_usage_mb():.1f} MB")
data = load_data()
print(f"After load: {memory_usage_mb():.1f} MB")
processed = process(data)
print(f"After process: {memory_usage_mb():.1f} MB")
```

**Use memory_profiler:**

```python
%load_ext memory_profiler

%memit data = np.random.random((10000, 10000))
# Output: peak memory: 762.5 MiB, increment: 762.5 MiB
```

### 7.4 Use Generators for Large Data Pipelines

**Impact: HIGH (constant memory usage)**

Process data in streams using generators instead of loading everything.

**Incorrect: loads all data**

```python
def process_all_files(filepaths):
    all_data = []
    for path in filepaths:
        all_data.append(load_file(path))  # 100 GB in memory
    return [process(d) for d in all_data]
```

**Correct: generator pipeline**

```python
def process_all_files(filepaths):
    for path in filepaths:
        data = load_file(path)  # Only one file in memory
        yield process(data)
        del data

# Process one at a time
for result in process_all_files(filepaths):
    save_result(result)
```

### 7.5 Avoid Copying DataFrames

**Impact: MEDIUM (reduces memory usage)**

Use `inplace=True` or avoid unnecessary copies.

**Incorrect: creates copies**

```python
df2 = df.drop(columns=['temp_col'])  # Copy
df3 = df2.fillna(0)  # Another copy
df4 = df3.reset_index()  # Another copy
```

**Correct: modify in place**

```python
df.drop(columns=['temp_col'], inplace=True)
df.fillna(0, inplace=True)
df.reset_index(drop=True, inplace=True)
```

**Or use method chaining (still creates copies but clearer):**

```python
df = (df
    .drop(columns=['temp_col'])
    .fillna(0)
    .reset_index(drop=True)
)
```

---

## 8. Reproducibility & Code Quality

**Impact: HIGH**

Reproducible, well-documented code is essential for scientific work.

### 8.1 Set Random Seeds

**Impact: CRITICAL (ensures reproducibility)**

Set random seeds for all random number generators.

**Incorrect: non-reproducible**

```python
import numpy as np

data = np.random.random(100)  # Different every run
```

**Correct: set seeds**

```python
import numpy as np
import random

# Set all seeds
np.random.seed(42)
random.seed(42)

# For newer NumPy
rng = np.random.default_rng(42)
data = rng.random(100)  # Same every run
```

**For PyTorch/TensorFlow:**

```python
import torch
torch.manual_seed(42)

import tensorflow as tf
tf.random.set_seed(42)
```

### 8.2 Version Pin Dependencies

**Impact: CRITICAL (reproducibility)**

Pin exact versions of all dependencies.

**Incorrect: unpinned versions**

```txt
# requirements.txt
numpy
pandas
matplotlib
```

**Correct: pinned versions**

```txt
# requirements.txt
numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.2
scipy==1.12.0
```

**Export current environment:**

```bash
pip freeze > requirements.txt
```

**Or use conda:**

```bash
conda env export > environment.yml
```

### 8.3 Document Data Provenance

**Impact: HIGH (reproducibility, debugging)**

Document where data came from, when, and how it was processed.

**Incorrect: no documentation**

```python
data = pd.read_csv('data.csv')
```

**Correct: document provenance**

```python
"""
Data Provenance:
- Source: Beamline 7-ID-C, APS
- Date: 2024-01-15
- Experiment: XRR_2024_001
- Sample: Si wafer with 50nm SiO2
- Raw data: /data/raw/scan_001.dat
- Processing: background subtraction, normalization to incident beam
"""

data = pd.read_csv('data.csv')

# Add metadata to results
metadata = {
    'source': 'APS 7-ID-C',
    'date': '2024-01-15',
    'processing_date': datetime.now().isoformat(),
    'processing_steps': ['background_subtraction', 'normalization']
}
```

### 8.4 Use Type Hints

**Impact: MEDIUM (code quality, IDE support)**

Add type hints to function signatures for clarity and error checking.

**Incorrect: no type hints**

```python
def calculate_reflectivity(q, sld, thickness):
    # What types are these?
    return result
```

**Correct: type hints**

```python
import numpy as np
from numpy.typing import NDArray

def calculate_reflectivity(
    q: NDArray[np.float64],
    sld: float,
    thickness: float
) -> NDArray[np.float64]:
    """
    Calculate X-ray reflectivity.
    
    Parameters
    ----------
    q : array of float
        Wavevector transfer in Å⁻¹
    sld : float
        Scattering length density in Å⁻²
    thickness : float
        Layer thickness in Å
        
    Returns
    -------
    reflectivity : array of float
        Calculated reflectivity
    """
    return result
```

### 8.5 Validate Input Data

**Impact: HIGH (prevents silent errors)**

Validate inputs to catch errors early.

**Incorrect: no validation**

```python
def process_scan(q, intensity):
    normalized = intensity / intensity[0]
    return normalized
```

**Correct: validate inputs**

```python
def process_scan(q: NDArray, intensity: NDArray) -> NDArray:
    """Process XRR scan data."""
    
    # Validate inputs
    assert len(q) == len(intensity), "q and intensity must have same length"
    assert len(q) > 0, "Empty arrays not allowed"
    assert np.all(q > 0), "q values must be positive"
    assert np.all(intensity >= 0), "Intensity cannot be negative"
    assert intensity[0] != 0, "Cannot normalize by zero"
    
    normalized = intensity / intensity[0]
    return normalized
```

**Alternative: raise informative errors**

```python
def process_scan(q: NDArray, intensity: NDArray) -> NDArray:
    if len(q) != len(intensity):
        raise ValueError(f"Length mismatch: q has {len(q)} points, "
                        f"intensity has {len(intensity)} points")
    
    if len(q) == 0:
        raise ValueError("Cannot process empty arrays")
    
    if not np.all(q > 0):
        raise ValueError(f"q values must be positive, got min={q.min()}")
    
    if not np.all(intensity >= 0):
        raise ValueError(f"Intensity cannot be negative, got min={intensity.min()}")
    
    if intensity[0] == 0:
        raise ValueError("Cannot normalize: first intensity value is zero")
    
    return intensity / intensity[0]
```

### 8.6 Use Logging Instead of Print

**Impact: MEDIUM (production readiness)**

Use logging module for better control and filtering.

**Incorrect: print statements**

```python
def process_data(data):
    print("Starting processing...")
    result = expensive_operation(data)
    print(f"Processing complete. Result shape: {result.shape}")
    return result
```

**Correct: use logging**

```python
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Starting processing...")
    result = expensive_operation(data)
    logger.info(f"Processing complete. Result shape: {result.shape}")
    logger.debug(f"Result stats: mean={result.mean():.3f}, std={result.std():.3f}")
    return result
```

**Different log levels:**

```python
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning: potential issue")
logger.error("Error occurred")
logger.critical("Critical error")
```

---

## 9. X-ray Specialized Libraries

**Impact: HIGH**

Specialized X-ray analysis libraries (periodictable, PyFAI, PyPXR, pyref) provide domain-specific functionality. Understanding their proper use and common pitfalls is essential for reliable analysis.

### 9.1 Periodic Table Package (periodictable)

**Impact: MEDIUM (elemental data access)**

The `periodictable` package provides access to elemental properties, but should be used carefully to avoid common pitfalls.

**Correct usage:**

```python
import periodictable as pt

# Access element properties
si = pt.Si
atomic_number = si.number
atomic_mass = si.mass
density = si.density

# X-ray scattering factors
si_f1 = pt.xray.scattering_factors(si, energy=8048)  # Cu K-alpha in eV
si_f2 = pt.xray.scattering_factors(si, energy=8048)[1]

# Form factor
form_factor = pt.xray.ff(si, q=0.1)  # q in Angstrom^-1
```

**Common pitfalls:**

```python
# Incorrect: accessing properties without checking units
si_density = pt.Si.density  # Default may not be in expected units

# Correct: verify units and use appropriate properties
si_density_g_cm3 = pt.Si.density  # Density in g/cm^3
si_atomic_volume = pt.Si.volume  # Atomic volume in cm^3/mol

# Incorrect: not specifying energy for scattering factors
f = pt.xray.scattering_factors(pt.Si)  # May use default energy

# Correct: explicitly specify energy
energy_kev = 8.048  # Cu K-alpha
f = pt.xray.scattering_factors(pt.Si, energy=energy_kev*1000)  # Convert to eV if needed

# Incorrect: assuming isotropic scattering for anisotropic materials
# periodictable assumes free atoms, not crystal structure effects

# Correct: understand limitations for anisotropic/crystalline systems
# For oriented samples, manual calculations may be needed
```

**Best practices:**

```python
# Always specify units explicitly
from periodictable import elements

element = elements['Si']
atomic_weight = element.mass  # Unified atomic mass units (u)
density = element.density  # g/cm^3 at standard conditions

# Use X-ray scattering factors with explicit energy
energy_eV = 8048  # Cu K-alpha
f1, f2 = pt.xray.scattering_factors(element, energy=energy_eV)

# Understand temperature-dependent properties
# Some properties may need correction for temperature
```

### 9.2 PyFAI for Diffraction Analysis

**Impact: HIGH (powder diffraction integration)**

PyFAI (Python Fast Azimuthal Integration) is powerful for 2D detector integration, but has specific requirements and common pitfalls.

**Correct usage:**

```python
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

# Initialize detector geometry
detector = pyFAI.detectors.Detector(pixel1=172e-6, pixel2=172e-6)
detector.max_shape = (2048, 2048)

# Define geometry (distance, center, rotation angles)
distance = 0.5  # meters
poni1 = 0.1  # meters
poni2 = 0.1  # meters
rot1 = rot2 = rot3 = 0.0
wavelength = 1.54e-10  # meters (Cu K-alpha)

# Create integrator
ai = AzimuthalIntegrator(
    dist=distance,
    poni1=poni1,
    poni2=poni2,
    rot1=rot1,
    rot2=rot2,
    rot3=rot3,
    detector=detector,
    wavelength=wavelength
)

# Integrate 2D image to 1D powder pattern
q, intensity = ai.integrate1d(
    image,
    npt=1000,
    unit='q_A^-1',
    method='csr'
)
```

**Common pitfalls:**

```python
# Incorrect: incorrect units for geometry parameters
ai = AzimuthalIntegrator(
    dist=0.5,  # Must be consistent units
    poni1=0.1,
    poni2=0.1,
    wavelength=1.54  # Wrong: should be in meters, not Angstroms
)

# Correct: consistent unit system
ai = AzimuthalIntegrator(
    dist=0.5,  # meters
    poni1=0.1,  # meters
    poni2=0.1,  # meters
    wavelength=1.54e-10  # meters
)

# Incorrect: not accounting for detector tilt/rotation
# Missing rot1, rot2, rot3 can cause systematic errors

# Correct: measure and include all geometry parameters
ai = AzimuthalIntegrator(
    dist=distance,
    poni1=poni1,
    poni2=poni2,
    rot1=measured_rot1,  # Include tilt
    rot2=measured_rot2,
    rot3=measured_rot3,
    detector=detector,
    wavelength=wavelength
)

# Incorrect: using wrong integration method
# 'numpy' method is slow for large arrays

# Correct: use optimized methods
q, intensity = ai.integrate1d(
    image,
    npt=1000,
    method='csr'  # Fast sparse matrix method
)
```

**Understanding the analysis:**

The agent should understand that PyFAI performs:
- Solid angle correction for 2D detectors
- Polarization corrections
- Geometric transformations (q-space or 2-theta space)
- Intensity normalization

The underlying physics involves:
- Transformation from detector coordinates to reciprocal space
- Solid angle corrections: `dOmega = sin(2theta) d(2theta) d(phi)`
- Polarization factors for synchrotron radiation
- Geometric distortion corrections

### 9.3 PyPXR and pyref for Reflectivity Analysis

**Impact: HIGH (reflectivity fitting and analysis)**

PyPXR and pyref provide reflectivity analysis capabilities, but understanding the underlying physics is more important than just using the libraries.

**Understanding reflectivity physics:**

The agent must understand how reflectivity analysis works, not just library usage:

1. **Fresnel reflection and transmission:**
   - Interface reflectivity: `R = |r|^2` where `r = (q1 - q2)/(q1 + q2)`
   - `q = sqrt(k^2 - 4*pi*SLD)` where `k = 2*pi/lambda` and `SLD` is scattering length density

2. **Parratt recursive formalism:**
   - Recursive calculation through multiple layers
   - Each layer has thickness `d`, SLD `rho`, roughness `sigma`
   - `R(j+1) = (r(j,j+1) + R(j) * exp(2*i*q(j)*d(j))) / (1 + r(j,j+1)*R(j)*exp(2*i*q(j)*d(j)))`

3. **Born approximation for thin films:**
   - Valid when `q*d << 1` where `d` is film thickness
   - `R(q) ~ |Fourier_transform(SLD_profile)|^2 / q^4`

4. **Resolution and instrument smearing:**
   - Angular divergence smearing
   - Wavelength spread smearing
   - Both must be convolved with theoretical reflectivity

**Using PyPXR (understanding what it does):**

```python
# PyPXR provides reflectivity calculation tools
# But understand what's happening internally

# The agent should know how to implement reflectivity calculations
# even if not using the library directly

# Example: manual Parratt calculation (what PyPXR does internally)
def calculate_reflectivity_parratt(q_values, layers):
    """
    Calculate X-ray reflectivity using Parratt recursion.
    
    Parameters
    ----------
    q_values : array
        Wavevector transfer values in A^-1
    layers : list of dict
        Each layer: {'thickness': float, 'sld': float, 'roughness': float}
        
    Returns
    -------
    reflectivity : array
        Reflectivity R(q)
    """
    # Implementation would involve:
    # 1. Calculate q for each layer (accounting for SLD)
    # 2. Calculate Fresnel coefficients at each interface
    # 3. Apply Parratt recursion from substrate to air
    # 4. Account for roughness via Nevot-Croce factor
    # 5. Apply instrument resolution smearing
    pass
```

**Common pitfalls:**

```python
# Incorrect: using library without understanding physics
from pyref import calculate_reflectivity
R = calculate_reflectivity(q, layers)  # What is it doing?

# Correct: understand what the calculation represents
# Know the Parratt formalism, Fresnel coefficients, roughness effects

# Incorrect: not accounting for instrument resolution
# Raw reflectivity curves need smearing convolution

# Correct: include resolution effects
R_theory = calculate_parratt_reflectivity(q, layers)
R_smeared = convolve_resolution(R_theory, q, angular_divergence, wavelength_spread)

# Incorrect: fitting without proper constraints
# Physical parameters must respect physical bounds

# Correct: use physical constraints
# Thickness > 0, roughness >= 0, SLD within reasonable ranges
# Roughness < min(neighboring layer thicknesses) / 2
```

**Best practices:**

```python
# Always validate physical reasonableness
def validate_layer_parameters(layers):
    """Validate physical constraints on layer parameters."""
    for i, layer in enumerate(layers):
        assert layer['thickness'] > 0, f"Layer {i} thickness must be positive"
        assert layer['roughness'] >= 0, f"Layer {i} roughness must be non-negative"
        if i > 0:
            # Roughness should be less than layer thickness
            assert layer['roughness'] < layer['thickness'] / 2, \
                f"Layer {i} roughness too large relative to thickness"
    
    # Check SLD values are reasonable (typically -1e-6 to 1e-5 A^-2 for X-rays)
    for layer in layers:
        assert abs(layer['sld']) < 1e-4, "SLD value unreasonably large"

# Include instrument parameters
instrument_params = {
    'angular_divergence': 0.001,  # radians
    'wavelength_spread': 0.001,   # relative spread
    'beam_intensity': 1e12,       # photons/sec
}
```

---

## 10. Spectroscopic Analysis

**Impact: HIGH**

Spectroscopic analysis (NEXAFS, XAS) requires understanding of the underlying physics and analysis methods, particularly for oriented samples and polarization-dependent measurements.

### 10.1 NEXAFS/XAS Analysis Principles

**Impact: CRITICAL (correct interpretation of spectroscopic data)**

Understanding NEXAFS/XAS analysis principles is essential for interpreting near-edge X-ray absorption fine structure data, especially for oriented samples. The agent should be familiar with the analysis framework described in Stohr, J. et al. (2016).

**Key physical principles:**

1. **Electric dipole approximation:**
   - X-ray absorption cross-section: `sigma ~ |<i|epsilon·r|f>|^2`
   - Transition matrix element depends on orientation of transition dipole moment
   - For p-orbitals: `sigma(theta) ~ cos^2(theta)` where `theta` is angle between E-field and orbital axis

2. **Polarization-dependent absorption:**
   - For linear polarization: `mu(alpha) = mu_0 * [1 + beta * P2(cos(alpha))]`
   - Where `alpha` is angle between E-field and molecular axis
   - `beta` is the linear dichroism parameter
   - `P2(x) = (3x^2 - 1)/2` is the second Legendre polynomial

3. **Transition moments for oriented molecules:**
   - For C 1s -> pi*: `sigma(alpha) ~ sin^2(alpha)` (perpendicular to molecular plane)
   - For C 1s -> sigma*: `sigma(alpha) ~ cos^2(alpha)` (parallel to molecular axis)
   - These relationships allow determination of molecular orientation

**Analysis workflow:**

```python
import numpy as np
from scipy.optimize import curve_fit

def analyze_nexafs_dichroism(energy, mu_parallel, mu_perpendicular):
    """
    Analyze NEXAFS linear dichroism to determine molecular orientation.
    
    Parameters
    ----------
    energy : array
        Photon energy in eV
    mu_parallel : array
        Absorption for E parallel to reference direction
    mu_perpendicular : array
        Absorption for E perpendicular to reference direction
        
    Returns
    -------
    dichroism_ratio : array
        (mu_parallel - mu_perpendicular) / (mu_parallel + 2*mu_perpendicular)
    tilt_angle : float
        Estimated tilt angle of transition dipole moment
    """
    # Calculate dichroism ratio
    # For pi* transitions: positive dichroism indicates molecules oriented normal to surface
    # For sigma* transitions: negative dichroism indicates molecules lying on surface
    dichroism = (mu_parallel - mu_perpendicular) / (mu_parallel + 2*mu_perpendicular)
    
    # Estimate tilt angle from dichroism
    # For pi* transitions: beta = P2(cos(tilt))
    # Dichroism = beta * (3*cos^2(alpha) - 1)/2
    # Solve for tilt angle
    beta = dichroism  # Simplified for perpendicular geometry
    tilt_angle = np.arccos(np.sqrt((2*beta + 1)/3)) * 180 / np.pi
    
    return dichroism, tilt_angle

def fit_orientation_angle(energy, mu_alpha, alpha_values):
    """
    Fit molecular orientation from polarization-dependent NEXAFS.
    
    Based on: mu(alpha) = mu_0 * [1 + beta * P2(cos(alpha - alpha_0))]
    """
    def model(alpha, mu_0, beta, alpha_0):
        """NEXAFS orientation model."""
        P2 = 0.5 * (3 * np.cos(np.deg2rad(alpha - alpha_0))**2 - 1)
        return mu_0 * (1 + beta * P2)
    
    # Fit for each energy point
    # For pi* transitions: beta should be positive
    # tilt_angle = arccos(sqrt((2*beta + 1)/3))
    pass
```

**Common pitfalls:**

```python
# Incorrect: not accounting for experimental geometry
# NEXAFS analysis requires knowing incident angle and polarization

# Correct: account for experimental setup
incident_angle = 20  # degrees from surface normal
polarization_angle = 0  # degrees, 0 = s-polarization, 90 = p-polarization

# Effective angle between E-field and surface
# For s-polarization: E-field is parallel to surface
# For p-polarization: E-field has component normal to surface
effective_angle = calculate_effective_angle(incident_angle, polarization_angle)

# Incorrect: not normalizing by total intensity
# Raw absorption needs proper normalization

# Correct: normalize properly
# Background subtract pre-edge, normalize to post-edge
pre_edge_region = (energy < 280)  # For C K-edge
post_edge_region = (energy > 310)
background = np.mean(mu[pre_edge_region])
mu_normalized = (mu - background) / (np.mean(mu[post_edge_region]) - background)

# Incorrect: assuming perfect polarization
# Real beamlines have partial polarization

# Correct: account for polarization factor
polarization_factor = 0.95  # 95% polarized
mu_corrected = (mu - mu_isotropic) / polarization_factor + mu_isotropic
```

**Best practices:**

```python
# Always validate physical constraints
def validate_nexafs_parameters(beta, tilt_angle):
    """Validate NEXAFS fit parameters."""
    # Beta must be between -1 and 1 for linear dichroism
    assert -1 <= beta <= 1, f"Beta ({beta}) must be between -1 and 1"
    
    # Tilt angle must be between 0 and 90 degrees
    assert 0 <= tilt_angle <= 90, f"Tilt angle ({tilt_angle}) must be between 0 and 90 degrees"
    
    # For pi* transitions, beta should be positive if molecules are upright
    # This is a physics-based check, not a mathematical constraint

# Use proper error propagation
from uncertainties import ufloat

beta = ufloat(0.6, 0.1)  # Value with uncertainty
tilt_angle = np.arccos(np.sqrt((2*beta + 1)/3)) * 180 / np.pi
# tilt_angle automatically has propagated uncertainty
```

**Understanding the physics:**

The agent should understand that:
- NEXAFS measures unoccupied density of states projected onto specific symmetry (pi*, sigma*)
- Orientation dependence comes from transition dipole moment direction
- Quantitative orientation analysis requires fitting polarization dependence
- Multiple angles are needed to determine 3D orientation distribution
- Anisotropy can indicate molecular alignment, crystalline orientation, or both

---

## 11. Units and Uncertainty Propagation

**Impact: MEDIUM-HIGH**

Proper unit management and uncertainty propagation are essential for reliable scientific analysis and reporting.

### 11.1 Using Pint for Unit Management

**Impact: MEDIUM-HIGH (prevents unit errors)**

Use Pint for explicit unit management to prevent unit conversion errors and improve code clarity.

**Correct usage:**

```python
import pint
ureg = pint.UnitRegistry()

# Define quantities with units
wavelength = 1.54 * ureg.angstrom
energy = 8.048 * ureg.keV
distance = 0.5 * ureg.meter

# Automatic unit conversion
wavelength_nm = wavelength.to('nanometer')
energy_eV = energy.to('electron_volt')

# Dimensional analysis
# Pint will catch unit errors at runtime
q = 2 * np.pi / wavelength  # Automatically in 1/angstrom

# Combining units
momentum = (2 * np.pi * ureg.hbar / wavelength).to('kg*m/s')

# Unit checking in calculations
thickness = 50 * ureg.nanometer
density = 2.3 * ureg.g / ureg.cm**3
mass = thickness * density  # Automatically correct units
```

**Common pitfalls:**

```python
# Incorrect: mixing unit systems without conversion
wavelength_angstrom = 1.54
wavelength_meter = 1.54e-10
q = 2 * np.pi / wavelength_angstrom  # Units unclear

# Correct: use Pint for explicit units
wavelength = 1.54 * ureg.angstrom
q = (2 * np.pi / wavelength).to('1/angstrom')

# Incorrect: not checking units match
energy1 = 8.048 * ureg.keV
energy2 = 8048 * ureg.eV
# These are the same, but Pint makes it explicit
total = energy1 + energy2  # Pint handles conversion automatically

# Incorrect: unit mismatches in formulas
# This will raise an error with Pint:
# wavelength = 1.54 * ureg.angstrom
# time = 1.0 * ureg.second
# invalid = wavelength + time  # DimensionalAnalysisError
```

**Best practices:**

```python
# Define custom units for domain-specific quantities
ureg.define('XU = 1.00202 * angstrom')  # X-ray unit
ureg.define('SLD_unit = 1 / angstrom**2')  # Scattering length density

# Use contexts for dimensionless calculations
with ureg.context('spectroscopy'):
    # Energy can be dimensionless in reduced units
    energy_reduced = energy / (1 * ureg.Ry)

# Store units with data
def calculate_reflectivity(q, layers):
    """
    Calculate reflectivity with units.
    
    Parameters
    ----------
    q : pint.Quantity
        Wavevector in 1/angstrom
    layers : list
        Each with 'thickness' (angstrom), 'sld' (1/angstrom^2)
    """
    # All calculations maintain units
    pass

# Convert to dimensionless for NumPy operations
q_values = q.magnitude  # Extract numeric values for NumPy
result = np.sqrt(q_values**2 - 4*np.pi*sld_values)
result_with_units = result * q.units  # Reattach units
```

### 11.2 Using Uncertainties Package for Error Propagation

**Impact: MEDIUM-HIGH (proper uncertainty quantification)**

Use the `uncertainties` package for automatic uncertainty propagation through calculations.

**Correct usage:**

```python
from uncertainties import ufloat
from uncertainties.umath import sqrt, exp, log
import uncertainties.unumpy as unp

# Define values with uncertainties
thickness = ufloat(50.0, 2.0)  # 50 nm ± 2 nm
density = ufloat(2.3, 0.1)  # 2.3 g/cm³ ± 0.1

# Automatic uncertainty propagation
mass = thickness * density
# mass = 115.0 ± 11.3 (uncertainty automatically calculated)

# Complex calculations
sld = ufloat(6.35e-6, 0.5e-6)  # Angstrom^-2
q = ufloat(0.1, 0.001)  # Angstrom^-1
q_eff = sqrt(q**2 - 4*3.14159*sld)
# Uncertainty in q_eff automatically calculated

# Working with arrays
thicknesses = unp.uarray([50.0, 100.0, 150.0], [2.0, 3.0, 4.0])
densities = unp.uarray([2.3, 2.4, 2.5], [0.1, 0.1, 0.15])
masses = thicknesses * densities
# Each element has its own uncertainty
```

**Common pitfalls:**

```python
# Incorrect: manual uncertainty propagation (error-prone)
# For f(x, y) = x * y, uncertainty is:
# df = sqrt((y*dx)^2 + (x*dy)^2)
# Easy to make mistakes in complex formulas

# Correct: let uncertainties package handle it
result = x * y  # Uncertainty automatically calculated correctly

# Incorrect: not propagating uncertainties through fits
# Fitted parameters should include uncertainties

# Correct: use uncertainties with fitting
from scipy.optimize import curve_fit

# Fit and get covariance matrix
params, cov = curve_fit(model_function, x, y, sigma=y_errors)

# Convert to ufloat with uncertainties
param_with_uncertainties = [
    ufloat(params[i], np.sqrt(cov[i, i]))
    for i in range(len(params))
]

# Incorrect: ignoring correlation in uncertainties
# When parameters are correlated, uncertainties are larger

# Correct: account for full covariance
# Use correlation matrix from fit
correlation_matrix = cov / np.outer(
    np.sqrt(np.diag(cov)),
    np.sqrt(np.diag(cov))
)
```

**Best practices:**

```python
# Combine Pint and uncertainties for complete error handling
from pint import UnitRegistry
from uncertainties import ufloat

ureg = UnitRegistry()

# Define quantities with both units and uncertainties
wavelength = ufloat(1.54, 0.01) * ureg.angstrom
energy = ufloat(8.048, 0.005) * ureg.keV

# Uncertainty propagates through unit conversions
wavelength_nm = wavelength.to('nanometer')
# wavelength_nm = (0.154 ± 0.001) nanometer

# Extract numerical values with uncertainties
wavelength_value = wavelength.magnitude  # ufloat(1.54, 0.01)
wavelength_units = wavelength.units  # angstrom

# Use in calculations
q = (2 * np.pi / wavelength).to('1/angstrom')
# q = (4.08 ± 0.03) / angstrom

# Report results with proper formatting
print(f"Thickness: {thickness:.2f}")  # "Thickness: 50.00±2.00"
print(f"Thickness: {thickness:.1u}")  # "Thickness: 50±2" (compact)

# Export for plotting
q_nominal = unp.nominal_values(q_array)
q_std = unp.std_devs(q_array)
error_bars = q_std
```

**Advanced usage:**

```python
# Correlated uncertainties
from uncertainties import correlated_values

# Parameters from fit with correlation
params, cov = curve_fit(model, x, y, sigma=y_err)

# Create correlated variables
a, b = correlated_values(params, cov)

# Calculation preserves correlation
result = a * x + b  # Uncertainty accounts for correlation

# Monte Carlo uncertainty propagation (for non-linear cases)
from uncertainties import unumpy

# Generate Monte Carlo samples
n_samples = 10000
q_samples = np.random.normal(
    q.nominal_value,
    q.std_dev,
    n_samples
)

# Propagate through non-linear function
result_samples = complex_nonlinear_function(q_samples)

# Calculate statistics
result_nominal = np.mean(result_samples)
result_std = np.std(result_samples)
result = ufloat(result_nominal, result_std)
```

---

## References

1. [NumPy Documentation](https://numpy.org/doc/)
2. [Pandas Documentation](https://pandas.pydata.org/docs/)
3. [Polars Documentation](https://pola-rs.github.io/polars/)
4. [DuckDB Documentation](https://duckdb.org/docs/)
5. [SciPy Documentation](https://docs.scipy.org/doc/)
6. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
7. [Numba Documentation](https://numba.pydata.org/)
8. [PyTorch Documentation](https://pytorch.org/docs/)
9. [Dask Documentation](https://docs.dask.org/)
10. [Joblib Documentation](https://joblib.readthedocs.io/)
11. [Periodic Table Python Package](https://periodictable.readthedocs.io/en/latest/)
12. [PyFAI Documentation](https://pyfai.readthedocs.io/en/stable/)
13. [PyPXR Documentation](https://p-rsoxr.readthedocs.io/en/latest/)
14. [pyref GitHub Repository](https://github.com/WSU-Carbon-Lab/pyref/tree/main)
15. [Pint Documentation](https://pint.readthedocs.io/)
16. [Uncertainties Package Documentation](https://uncertainties-python-package.readthedocs.io/)
17. Stohr, J. et al. (2016). "NEXAFS Spectroscopy of Oriented Molecules." Surface Science, 646, 26-32. DOI: 10.1016/j.susc.2015.10.007
18. [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed)

---

This guide provides a comprehensive framework for writing performant, reproducible Python code for X-ray data analysis. The patterns here should be applied based on your specific bottlenecks—use profiling tools to identify where optimization efforts will have the most impact.

## Learned User Preferences

- For seaborn lineplot under Pyright or Pylance, pass a DataFrame or other supported tabular `data` argument; a bare pandas Series commonly fails `DataSource` typing and produces assignability errors.
- In agent mode, change notebooks and project files directly instead of only giving manual edit instructions for the user to apply.
# SeqCoBench


## Installing dependencies

Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

Create a Virtual Environment

```bash
conda create --name seqcobench_env python=3.9
```

## Benchmark construction

Run the following commands to apply different perturbations to the MBPP dataset:

```bash
#!/bin/bash

NUM_OUTPUTS=5  
DATASET="mbpp"

# Iterate over augmentation methods 0 to 6
for AUG_METHOD in {0..6}; do
    python source/data/transformations/run_robust.py perturb natgen \
        --aug_method $AUG_METHOD --datasets $DATASET --overwrite \
        --n_outputs $NUM_OUTPUTS
done
```


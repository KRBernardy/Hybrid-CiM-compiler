
# PUMA Compiler

## Overview

This repository includes the compiler for the Programmable Ultra-efficient Memristor-based Accelerator.

## Organization

The compiler is implemented in the `src` directory.

The programming interface is provided in the `include` directory.

Example programs are provided in the `test` directory.

## How to run

First make the library:

```
cd src
make
```

Add the library to path:

```
export LD_LIBRARY_PATH=`pwd`/../src:$LD_LIBRARY_PATH
```

Go to examples folder and generate files(mlp_l4_mnist as a example)

```
make mlp_l4_mnist.test
./mlp_l4_mnist.test
```

You should see a folder created with ops.json in it(if contains weights, also weights.json).
After that you can convert weights.json to weights.npz:

```
python json_weight_to_npz.py json_file mlp_l4_mnist/weights.json
```

## Citation

Please cite the following paper if you find this work useful:

* A. Ankit, I. El Hajj, S. Chalamalasetti, G. Ndu, M. Foltin, R. S. Williams, P. Faraboschi, W.-M. Hwu, J. P. Strachan, K. Roy, D. Milojicic.
  **PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference**.
  In *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, 2019.


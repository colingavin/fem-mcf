# Curvature Flows using the Finite Element Method

This repository contains an implementation of [Walkington's FEM](http://epubs.siam.org/doi/pdf/10.1137/S0036142994262068) for curve shortening flow using `deal.II`. It also includes a modified version of this method for solving mean curvature flow of surfaces of revolution in 3D.

## Building

- Prerequisites:
  - deal.II version 8.4: Download from http://dealii.org/download.html and install according to the instructions for your platform.
  - nlopt: Download from http://ab-initio.mit.edu/wiki/index.php/NLopt or install on OS X using homebrew. Assumed to be installed at `/usr/local/opt/nlopt`.
- Use the deal.II setup script to configure your build environment.
- Run `./setup.sh` to create output directories.
- Run `cmake .`
- Run `make all` to build all of the test programs.
- Test programs will be built to bin/

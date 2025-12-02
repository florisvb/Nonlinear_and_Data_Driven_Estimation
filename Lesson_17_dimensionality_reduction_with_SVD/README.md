# Overview

This repository demonstrates terrain generation, optic flow calculation, and dimensionality reduction techniques for visual navigation of a planar drone. The notebooks provide tools for understanding how high-dimensional visual measurements can be compressed and used for state estimation.

## Key Concepts

* Generate synthetic terrain and compute analytic optic flow for a camera-equipped drone using ray tracing
* Apply Singular Value Decomposition (SVD) to compress high-dimensional optic flow into a low-dimensional representation
* Evaluate compression quality through reconstruction error and generalization tests
* Understand the relationship between drone motion, terrain complexity, and optic flow patterns

## Notebook A: Terrain and Optic Flow for Planar Drone

Introduction to terrain generation, ray tracing, and analytic optic flow mathematics. Demonstrates how optic flow patterns emerge from different motion profiles and terrain features.

**Key learning objectives:**
- Understand the full equations for optic flow
- Gain intuition for optic flow patterns corresponding to different types of motion
- Consider implications for high-dimensional measurements in state estimation

## Notebook B: SVD Compression of Optic Flow

Application of SVD to reduce dimensionality of optic flow measurements while preserving information content. Includes validation on test datasets and exercises for exploring compression limits.

**Key learning objectives:**
- Understand how SVD reduces dimensionality while preserving information
- Learn to choose appropriate truncation ranks for compression
- Evaluate compression quality and generalization to out-of-distribution data

# References

For Singular Value Decomposition:
  * Chapter 1 from Book: Brunton, S. L., & Kutz, J. N. (2019, 2022). Data-driven science and engineering: Machine learning, dynamical systems, and control. Cambridge University Press. https://www.databookuw.com/
  * [Video lecture series by Steve Brunton on the SVD](https://www.youtube.com/watch?v=gXbThCXjZFM&list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv)
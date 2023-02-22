# ML_NEGF

A machine learning approach to optimizing [NEGF](https://arxiv.org/abs/2008.01275) convergence.


NEGF produces possesses different derivations. In the simplest, 1D case, NEGF generates a series of charge distributions across 2D slices of a nano device.

These 2D charge distributions are stitched together to make the solution of a real-world 3D device.

## Goal
The goal of this approach is to train a generative network on 2D charge density slices. However, unlike in traditional generative methods, the goal of the network will be to minimize the difference betweens between the charge distribution at the first and last iterations. In this way the ML model can eliminate a large number of the iteration procedure, thus saving time and resources.

## Benefits

Optimizing the convergence of NEGF will allow it to be used by industry as the cost vs performance of the method currently is too large to warrant adoption. If the model improves convergence, we can assume that NEGF will always converge in 2 steps. The first to initialize charge, and the last to guarantee physical accuracy.

## Process

![ML_NEGF](diagrams/mode_of_operation-Page-1.svg "Mode of operation")

## Input

![Charge_NEGF](diagrams/charge_distribution.png "Charge distro" )


## Network

This is based on autoencoders, we avoid denoising autoencoders because there is no issue with achieving the identity operation.

## Resizing?
How to make sure that any cross section of a device will work?

- The device is always rectangular and if not, we can pad it out to be with 0s
- Then we need to make it into the same size square
- The idea is that we can condense any shape into a square and we can train the network that padding is always padding. This means that we will reduce the accuracy
- Maybe we can exclude the padding from the loss function
- However, it again all boils down to resolution, but ideally we don't need much resolution.


So the plan:

```
1 1 1    1 1 1     4/3 4/3 4/3 
1 1 1 vs 1 1 1 ->  4/3 4/3 4/3
1 1 1    1 1 1     4/3 4/3 4/3
         1 1 1 
```

Integral of the charge over are divided by unit area;
In the case above the field has a value of 1 same as with the original, but the area we are looking at is larger, (1 x 4/3), so the integral is 1x1x4/3 = 4/3. This is then divided by unit area 1 = 4/3. Ok this is cool.


Convergence is a bit of an issue. We train the models, however, the accuracy for small numbers is bad. In essence, we have a portion of the dataset in the range 1e5 and another in the range 1e-5, so the model trains well on the large data and poorly on the small data because the loss is much smaller.

Potential solution: Custom loss function. Something like relative loss. (Tar/Pred) 
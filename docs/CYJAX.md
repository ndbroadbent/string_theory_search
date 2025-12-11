https://iopscience.iop.org/article/10.1088/2632-2153/acdc84

Abstract
We present the first version of CYJAX, a package for machine learning Calabi–Yau metrics using JAX. It is meant to be accessible both as a top-level tool and as a library of modular functions. CYJAX is currently centered around the algebraic ansatz for the Kähler potential which automatically satisfies Kählerity and compatibility on patch overlaps. As of now, this implementation is limited to varieties defined by a single defining equation on one complex projective space. We comment on some planned generalizations.

More documentation can be found at: https://cyjax.readthedocs.io.

The code is available at: https://github.com/ml4physics/cyjax.



Calabi-Yau metrics with JAX
CYJAX is a python library for numerically approximating Calabi-Yau metrics using machine learning implemented with the JAX library. It is meant to be accessible both as a top-level library as well as a toolkit of modular functions. As of now, this implementation is limited to varieties given by a single defining equation on one complex projective space. A generalization to a wider class of cases is planned.

Good places to start are the introduction and the tutorial notebooks listed below or this introductory paper. More background can also be found in this paper. Some background knowledge on JAX and Flax may be helpful.

The introduction gives a summary of the mathematical context and aim of the library, which serves to give a broad overview to the structure and code of the library. The tutorials show how to use the library on a code level and give several examples.

If you find this work useful, please cite:

@article{gerdes2022cyjax,
    title = "{CYJAX: A package for Calabi-Yau metrics with JAX}",
    author = "Gerdes, Mathis and Krippendorf, Sven",
    eprint = "2211.12520",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1088/2632-2153/acdc84",
    journal = "Mach. Learn. Sci. Tech.",
    volume = "4",
    number = "2",
    pages = "025031",
    year = "2023"
}
Conventions
Generally, when the patch is None the coordinates are assumed to be homogeneous coordinates. Otherwise, the patch gives the index for the affine patch.

The index of the affine patch is given in terms of the homogeneous coordinates.

The dependent coordinate index (for which we can solve using the defining equation) is given in terms of the affine coordinate vector. Specifically, this means that, numerically, the dependent and the patch index may have the same value but do not refer to the same coordinate index.

Input variables of polynomials should have integer subscripts as in
. Parameters can use any valid sympy symbol expression including non-numerical subscripts.

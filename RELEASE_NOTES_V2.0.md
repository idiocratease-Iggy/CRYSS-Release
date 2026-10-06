Here are three stand alone Windows applications:

i) CRYSS DOE - a guidance app for planning arrays of experiments suitable for analysis and ranking by the CRYSS application

ii) NODES - a self-consistent thermodynamic engine giving in silico results from conceptual experiment such as envisioned in CRYSS DOE

iii) CRYSS - CRYSS V2.0 preserves the underlying mechanistic model of V1.0 but introduces four major advancements:

1.	Enhanced Input Parser Updated logic supporting the CRYSS V2.0 workbook format, including improved slice handling, replicate absorption, salt former tracking, and tolerance based filtering.
2.	Interactive Parser Validation GUI A new front end interface enabling users to tune analytical and noise tolerances, inspect parser diagnostics, and ensure high quality data is passed to the optimizer.
3.	CRYSS-P Core (C++ Engine) A full C++ implementation of the Monte Carlo + optimizer pipeline, leveraging multi core, multi thread processors to deliver ~40× faster execution, reducing runtime from minutes to seconds.
4.	Interactive Results Viewer A back end GUI providing heatmap based Rmax visualization, best system plots, and interactive stacked composition/solubility analysis 

If this helps, please consider supporting the project:

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-FFD700?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/idiocratease)

============================================================================================

CRYSS V2.0 uses the C++ CRYSS‑P Core by default, but users can also run the pure‑Python CRYSS Core for comparison or integration purposes.  
To run the Python engine, launch the application with the --py flag (for example, a .bat file containing cryss.exe --py). When the Python core is active, the MC+Optimizer splash screen displays "CRYSS"; the C++ engine displays "CRYSS‑P".

Downstream of the input parser, both engines execute the same algorithmic workflow.  The Python implementation in the public repository: 

<https://github.com/idiocratease-Iggy/CRYSS-Release> 

contains the full optimization logic, thermodynamic model, and multicomponent eutectic calculations used in the V2.0 app. The only differences are the C++ bindings and GUI hooks required for the desktop application, plus the removal of a small amount of legacy looping scaffolding inherited from NODES that was never called in CRYSS. These minor clean‑ups do not affect functionality: the Python core remains a faithful, fully working reference implementation.

For organisations wishing to integrate or transcode CRYSS into an existing software platform, the Python core provides an excellent starting point. It is readable, self‑contained, and mirrors the behaviour of the C++ engine, making it ideal for technical evaluation, internal validation, or custom deployment.

============================================================================================

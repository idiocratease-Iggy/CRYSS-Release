> **Note:** Windows Defender may block install as an unknown app.  
> We have logged this with Microsoft for a Defender update. Current status is below.

---

### Submission details

- **File:** `cryss_instalv2.0.zip`  
- **Submission ID:** `1c897fb7-221d-425d-8e65-d3a955885809`  
- **Status:** In progress  
- **Submitted by:** idiocratease@####  
- **Submitted:** Oct 6, 2026 14:13:45  
- **User Opinion:** `PuaFalsePositive`  
- **Analyst comments:**  
  The submitted files do not meet our criteria for malware or potentially unwanted applications.  
  No detection will be added for these files.

In the meantime, until Defender recognizes the app from the submission above, it can be installed by selecting **"Run anyway"**.

---

## Included standalone Windows applications

1. **CRYSS DOE**  
   A guidance app for planning arrays of experiments suitable for analysis and ranking by the CRYSS application.

2. **NODES**  
   A self-consistent thermodynamic engine giving *in silico* results from conceptual experiments such as envisioned in CRYSS DOE.

3. **CRYSS**  
   CRYSS V2.0 preserves the underlying mechanistic model of V1.0 but introduces four major advancements:

   - **Enhanced Input Parser**  
     Updated logic supporting the CRYSS V2.0 workbook format, including improved slice handling, replicate absorption, salt former tracking, and tolerance-based filtering.
   - **Interactive Parser Validation GUI**  
     A new front-end interface enabling users to tune analytical and noise tolerances, inspect parser diagnostics, and ensure high-quality data is passed to the optimizer.
   - **CRYSS-P Core (C++ Engine)**  
     A full C++ implementation of the Monte Carlo + optimizer pipeline, leveraging multi-core, multi-thread processors to deliver ~40× faster execution, reducing runtime from minutes to seconds.
   - **Interactive Results Viewer**  
     A back-end GUI providing heatmap-based Rmax visualization, best-system plots, and interactive stacked composition/solubility analysis.

---

## Support this project

If this helps, please consider supporting my work:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/idiocratease)

---

CRYSS V2.0 uses the **C++ CRYSS‑P Core** by default, but users can also run the pure-Python CRYSS Core for comparison or integration purposes.

To run the Python engine, launch the application with the `--py` flag (for example, a `.bat` file containing `cryss.exe --py`).  
When the Python core is active, the MC+Optimizer splash screen displays **"CRYSS"**; the C++ engine displays **"CRYSS‑P"**.

Downstream of the input parser, both engines execute the same algorithmic workflow. The Python implementation in the public repository:

https://github.com/idiocratease-Iggy/CRYSS-Release

contains the full optimization logic, thermodynamic model, and multicomponent eutectic calculations used in the V2.0 app.

The only differences are the C++ bindings and GUI hooks required for the desktop application, plus the removal of a small amount of legacy looping scaffolding inherited from NODES that was never called in CRYSS. These minor clean-ups do not affect functionality: the Python core remains a faithful, fully working reference implementation.

For organizations wishing to integrate or transcode CRYSS into an existing software platform, the Python core provides an excellent starting point. It is readable, self-contained, and mirrors the behaviour of the C++ engine, making it ideal for technical evaluation, internal validation, or custom deployment.

## Experiment Log: Chunk Size Optimization

### 1. Chunk Size Effect
* **Initial Strategy (Size: 1000, Overlap: 200):** Tested with a larger text-splitting configuration. The retrieved documents felt slightly vague and off-target regarding the specific query intent, likely due to chunks crossing multiple conceptual boundaries in the Markdown source.
* **Optimized Strategy (Size: 500, Overlap: 100):** Reduced chunk sizes to create tighter, more granular segments. Although both strategies ultimately pulled from the same sections of the source Markdown file, the smaller chunks provided cleaner boundaries, better contextual alignment, and slightly improved readability for subsequent retrieval and generation steps.

## Experiment Log: Local vLLM Deployment Blocked by CUDA/Driver Version Mismatch

* **What went wrong:** 
  Attempted to run vLLM with `Qwen/Qwen2.5-3B-Instruct` locally inside WSL. The process repeatedly crashed with a `RuntimeError` stating that *The NVIDIA driver on your system is too old (found version 12070)*. This occurred because modern PyTorch and vLLM CUDA wheels (CUDA 12.1/13.0) require a newer GPU driver version than what the Windows host currently passes through to the WSL environment.

* **What I tried:**
  * Checked existing PyTorch and CUDA compatibility (`torch.version.cuda`).
  * Attempted to reinstall/downgrade PyTorch versions to match CUDA 12.1 and CUDA 11.8 packages.
  * Encountered Python environment package limits (Python 3.14 / 3.13 wheel availability mismatches with older CUDA builds).
  * Explored potential host-side driver updates, but decided against it to avoid stability risks on the host machine.

* **Next approach:**
  * Rolling back the repository to discard local virtual environment clutter.
  * Pivoting away from vLLM's strict GPU-only engine requirements for now.
  * Shifting to a standard **Hugging Face Transformers** pipeline (CPU or fallback mode) to fulfill the local model serving objective and keep the project moving forward.
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

## Experimental Log: Local RAG Pipeline Execution & Performance Bottlenecks
The goal of this experiment was to integrate a local RAG retrieval system (`app.rag`) with a local causal language model (`app.local_llm`) to handle agricultural supply chain and logistics queries offline.

### **What I Tried**
* **Model Initialization:** Loaded the `Qwen/Qwen2.5-3B-Instruct` model via Hugging Face Transformers using `torch.float32` precision for stable CPU execution, coupled with `device_map="auto"`.
* **Pipeline Configuration:** Configured a Hugging Face `text-generation` pipeline wrapped around a custom chat-template prompt builder.
* **RAG Workflow Execution:** Executed a test query looking up inventory stock quantities and transit times, retrieving matching enterprise chunks from the local vector database, and passing the combined context into the model for inference.

### **What Went Wrong**
* **Disk/CPU Offloading Bottleneck:** Due to hardware memory limitations, the system RAM was insufficient to hold the full 3-billion-parameter model weights. Hugging Face automatically triggered a fallback, offloading model components to the local disk/SSD. Because transformer text generation requires sequential, token-by-token processing, constantly reading weights from disk created a severe I/O bottleneck, freezing the execution indefinitely after reaching the generation step.
* **Pipeline Configuration Conflict:** The model's internal base configuration (`config.json`) hardcoded a default `max_length=20` parameter. This conflicted with the explicit `max_new_tokens` passed during pipeline calls, generating persistent deprecation and parameter-precedence warnings.
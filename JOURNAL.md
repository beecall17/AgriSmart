## Experiment Log: Chunk Size Optimization

### 1. Chunk Size Effect
* **Initial Strategy (Size: 1000, Overlap: 200):** Tested with a larger text-splitting configuration. The retrieved documents felt slightly vague and off-target regarding the specific query intent, likely due to chunks crossing multiple conceptual boundaries in the Markdown source.
* **Optimized Strategy (Size: 500, Overlap: 100):** Reduced chunk sizes to create tighter, more granular segments. Although both strategies ultimately pulled from the same sections of the source Markdown file, the smaller chunks provided cleaner boundaries, better contextual alignment, and slightly improved readability for subsequent retrieval and generation steps.
# Fix: Improve retrieval accuracy for ingested source code files

Fixes #1871

## Problem

When using PrivateGPT with Ollama and llama3, ingested source code files (e.g. `.cs` scripts) are effectively invisible to the LLM unless the user provides explicit hints about file names or locations. The model behaves as if the ingested files aren't there.

## Root Causes Identified

1. **`file_name` hidden from LLM** — `ingest_helper.py` added `file_name` to `excluded_llm_metadata_keys`, so the LLM never saw which file a chunk came from and couldn't reference files by name.

2. **No source code file readers** — Extensions like `.cs`, `.py`, `.js`, `.go`, `.java`, etc. had no registered reader and fell through to `StringIterableReader`, which read entire files as a single blob with no structural metadata.

3. **Sentence-based chunking for code** — `SentenceWindowNodeParser` splits on sentence boundaries, which breaks function definitions, class hierarchies, and other code structures. Code needs a structure-aware splitter.

4. **Only 2 chunks retrieved** — `similarity_top_k: 2` meant only 2 chunks from the entire corpus were ever passed to the LLM. For codebases with many files, this was far too few.

5. **Reranking disabled** — Without a reranker, raw embedding similarity results included many false positives.

6. **Underutilized context window** — `context_window: 3900` wasted llama3's 8192-token capacity.

## Changes

### `private_gpt/components/ingest/ingest_helper.py`

- **Added `CodeFileReader`** — A new reader class that reads source code as plain text and attaches `language` metadata (e.g. `"python"`, `"csharp"`).
- **Added `CODE_EXTENSIONS` mapping** — Maps 50+ source code extensions (`.py`, `.js`, `.ts`, `.cs`, `.java`, `.go`, `.rb`, `.rs`, `.cpp`, `.c`, `.h`, `.php`, `.swift`, `.kt`, `.scala`, `.sh`, `.sql`, `.lua`, `.dart`, `.vue`, `.html`, `.css`, `.yaml`, `.toml`, etc.) to their language names.
- **Registered code readers in `FILE_READER_CLS`** — All code extensions now get a dedicated reader instead of falling through to the generic `StringIterableReader`.
- **Fixed `excluded_llm_metadata_keys`** — Removed `file_name` from the exclusion list so the LLM can see which file each chunk originates from. `doc_id` and `page_label` remain excluded.

### `private_gpt/server/ingest/ingest_service.py`

- **Added `CodeAwareNodeParser`** — Routes documents to the appropriate splitter based on file extension:
  - Source code files → `CodeSplitter` (tree-sitter based, respects function/class boundaries)
  - Other documents → `SentenceWindowNodeParser` (existing behavior)
  - Graceful fallback: if `CodeSplitter` fails for a language (missing grammar), falls back to sentence parser with a warning.
- Replaced `SentenceWindowNodeParser.from_defaults()` with `CodeAwareNodeParser()` in `IngestService`.

### `settings.yaml` (default config)

- `similarity_top_k`: 2 → **10** (retrieve more candidate chunks)
- `rerank.enabled`: false → **true** (filter irrelevant chunks via cross-encoder)
- `rerank.top_n`: 1 → **5** (keep top 5 after reranking)

### `settings-ollama.yaml`

- `context_window`: 3900 → **8192** (use llama3's full capacity)
- Added `prompt_style: "llama3"` (correct prompt template for llama3 models)
- Added `rag` section with tuned retrieval settings (matching defaults)

### `tests/test_code_ingestion.py` (new)

21 new unit tests covering:
- `CodeFileReader` — reads code, attaches language metadata, preserves extra info
- `CODE_EXTENSIONS` mapping — all extensions registered in `FILE_READER_CLS`
- Metadata exclusion fix — `file_name` NOT in `excluded_llm_metadata_keys`
- `CodeAwareNodeParser` — routes code vs. text, handles empty input, handles missing metadata

## How to Test

### Unit tests

```bash
PGPT_PROFILES=mock pytest tests/test_code_ingestion.py -v
```

### Full test suite

```bash
PGPT_PROFILES=mock pytest tests/ -v
```

### Manual end-to-end test

1. Start the server:
   ```bash
   PGPT_PROFILES=ollama make run
   ```

2. Ingest some `.cs` files (or any source code):
   ```bash
   curl -X POST http://localhost:8001/v1/ingest/file \
     -F "file=@MyScript.cs"
   ```

3. Ask the LLM about the ingested code without mentioning file names:
   ```
   "What functions are available for handling user input?"
   ```

4. Expected: The LLM now references the correct source files by name and retrieves relevant code chunks.

## Before / After

**Before:**
- "I don't have any information about user input handling" (even though the code was ingested)
- Only worked if user said "look in InputHandler.cs"

**After:**
- "Based on the ingested files, `InputHandler.cs` contains `HandleKeyPress()` and `ProcessMouseInput()` functions for handling user input."
- LLM sees file names in context and retrieves 10x more candidate chunks with reranking

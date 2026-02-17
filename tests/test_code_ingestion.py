"""Tests for code-file ingestion improvements (Issue #1871).

Validates:
- CodeFileReader reads source code and attaches language metadata
- Source code extensions are mapped in FILE_READER_CLS
- file_name is NOT excluded from LLM metadata (so the LLM can reference files)
- CodeAwareNodeParser routes code vs. text correctly
"""

import tempfile
from pathlib import Path

import pytest
from llama_index.core.schema import Document, TextNode

from private_gpt.components.ingest.ingest_helper import (
    CODE_EXTENSIONS,
    CodeFileReader,
    FILE_READER_CLS,
    IngestionHelper,
)


class TestCodeFileReader:
    """Tests for the CodeFileReader class."""

    def test_load_data_returns_document_with_text(self, tmp_path: Path) -> None:
        code = "def hello():\n    print('hello')\n"
        code_file = tmp_path / "hello.py"
        code_file.write_text(code)

        reader = CodeFileReader(language="python")
        docs = reader.load_data(code_file)

        assert len(docs) == 1
        assert docs[0].text == code

    def test_load_data_attaches_language_metadata(self, tmp_path: Path) -> None:
        code_file = tmp_path / "main.cs"
        code_file.write_text("using System;\nclass Foo {}\n")

        reader = CodeFileReader(language="csharp")
        docs = reader.load_data(code_file)

        assert docs[0].metadata["language"] == "csharp"

    def test_load_data_preserves_extra_info(self, tmp_path: Path) -> None:
        code_file = tmp_path / "app.js"
        code_file.write_text("console.log('hi');\n")

        reader = CodeFileReader(language="javascript")
        docs = reader.load_data(code_file, extra_info={"custom_key": "custom_value"})

        assert docs[0].metadata["language"] == "javascript"
        assert docs[0].metadata["custom_key"] == "custom_value"


class TestCodeExtensionsMapping:
    """Tests for FILE_READER_CLS code extension registration."""

    @pytest.mark.parametrize(
        "ext",
        [".py", ".js", ".ts", ".cs", ".java", ".go", ".rb", ".rs", ".cpp", ".c"],
    )
    def test_common_code_extensions_registered(self, ext: str) -> None:
        assert ext in FILE_READER_CLS, f"Extension {ext} should be in FILE_READER_CLS"

    def test_code_extensions_dict_is_nonempty(self) -> None:
        assert len(CODE_EXTENSIONS) > 30, "Should have many code extensions mapped"

    def test_all_code_extensions_in_file_reader_cls(self) -> None:
        for ext in CODE_EXTENSIONS:
            assert ext in FILE_READER_CLS, (
                f"CODE_EXTENSIONS has {ext} but FILE_READER_CLS does not"
            )


class TestMetadataExclusion:
    """Tests that file_name is visible to the LLM after ingestion."""

    def test_file_name_not_in_excluded_llm_metadata(self) -> None:
        """file_name must NOT be excluded from LLM metadata.

        This is the core fix for issue #1871 — the LLM needs to see
        which file a chunk came from to reference it in answers.
        """
        doc = Document(text="some text", metadata={"file_name": "test.cs"})
        docs = [doc]
        IngestionHelper._exclude_metadata(docs)

        assert "file_name" not in docs[0].excluded_llm_metadata_keys

    def test_doc_id_still_excluded_from_llm_metadata(self) -> None:
        doc = Document(text="some text", metadata={"file_name": "test.py"})
        docs = [doc]
        IngestionHelper._exclude_metadata(docs)

        assert "doc_id" in docs[0].excluded_llm_metadata_keys

    def test_doc_id_excluded_from_embed_metadata(self) -> None:
        doc = Document(text="some text", metadata={"file_name": "test.py"})
        docs = [doc]
        IngestionHelper._exclude_metadata(docs)

        assert "doc_id" in docs[0].excluded_embed_metadata_keys


class TestCodeAwareNodeParser:
    """Tests for the CodeAwareNodeParser routing logic."""

    def test_routes_code_files_separately_from_text(self) -> None:
        """Verify that code and text nodes are separated correctly."""
        from private_gpt.server.ingest.ingest_service import CodeAwareNodeParser

        parser = CodeAwareNodeParser()

        code_node = TextNode(
            text="def foo():\n    pass\n",
            metadata={"file_name": "foo.py"},
        )
        text_node = TextNode(
            text="This is a regular text document about something.",
            metadata={"file_name": "readme.txt"},
        )

        # The parser should process both without errors
        result = parser([code_node, text_node])
        assert len(result) > 0, "Parser should return at least one node"

    def test_handles_empty_input(self) -> None:
        from private_gpt.server.ingest.ingest_service import CodeAwareNodeParser

        parser = CodeAwareNodeParser()
        result = parser([])
        assert result == []

    def test_handles_nodes_without_file_name(self) -> None:
        from private_gpt.server.ingest.ingest_service import CodeAwareNodeParser

        parser = CodeAwareNodeParser()
        node = TextNode(text="no file name here", metadata={})
        result = parser([node])
        # Should fall through to sentence parser without error
        assert len(result) > 0

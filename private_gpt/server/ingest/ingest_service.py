import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, AnyStr, BinaryIO

from injector import inject, singleton
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.storage import StorageContext

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.ingest.ingest_helper import CODE_EXTENSIONS
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)


class CodeAwareNodeParser:
    """A node parser that routes code files to a code-aware splitter.

    Documents whose ``file_name`` metadata ends with a recognised source-code
    extension are split using ``CodeSplitter`` (tree-sitter based), which
    respects function / class boundaries.  All other documents fall through to
    the default ``SentenceWindowNodeParser``.
    """

    def __init__(self, chunk_lines: int = 40, chunk_lines_overlap: int = 15) -> None:
        self._chunk_lines = chunk_lines
        self._chunk_lines_overlap = chunk_lines_overlap
        self._sentence_parser = SentenceWindowNodeParser.from_defaults()

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        code_nodes: list[BaseNode] = []
        text_nodes: list[BaseNode] = []

        for node in nodes:
            file_name = node.metadata.get("file_name", "")
            ext = Path(file_name).suffix.lower() if file_name else ""
            if ext in CODE_EXTENSIONS:
                code_nodes.append(node)
            else:
                text_nodes.append(node)

        result: list[BaseNode] = []

        # Process code documents with CodeSplitter
        if code_nodes:
            try:
                from llama_index.core.node_parser import CodeSplitter

                code_splitter = CodeSplitter(
                    language=CODE_EXTENSIONS.get(
                        Path(code_nodes[0].metadata.get("file_name", "")).suffix.lower(),
                        "python",
                    ),
                    chunk_lines=self._chunk_lines,
                    chunk_lines_overlap=self._chunk_lines_overlap,
                )
                # Group code nodes by language to use the right splitter per group
                by_language: dict[str, list[BaseNode]] = {}
                for n in code_nodes:
                    fn = n.metadata.get("file_name", "")
                    ext = Path(fn).suffix.lower() if fn else ""
                    lang = CODE_EXTENSIONS.get(ext, "python")
                    by_language.setdefault(lang, []).append(n)

                for lang, lang_nodes in by_language.items():
                    try:
                        splitter = CodeSplitter(
                            language=lang,
                            chunk_lines=self._chunk_lines,
                            chunk_lines_overlap=self._chunk_lines_overlap,
                        )
                        result.extend(splitter(lang_nodes))
                    except Exception:
                        # If CodeSplitter fails for a language (e.g. missing
                        # tree-sitter grammar), fall back to sentence parser.
                        logger.warning(
                            "CodeSplitter failed for language=%s, falling back to sentence parser",
                            lang,
                        )
                        result.extend(self._sentence_parser(lang_nodes))
            except ImportError:
                logger.warning(
                    "CodeSplitter not available, falling back to sentence parser for code files"
                )
                result.extend(self._sentence_parser(code_nodes))

        # Process non-code documents with the sentence-window parser
        if text_nodes:
            result.extend(self._sentence_parser(text_nodes))

        return result


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        node_parser = CodeAwareNodeParser()

        self.ingest_component = get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component.embedding_model,
            transformations=[node_parser, embedding_component.embedding_model],
            settings=settings(),
        )

    def _ingest_data(self, file_name: str, file_data: AnyStr) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return self.ingest_file(file_name, path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def ingest_file(self, file_name: str, file_data: Path) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = self.ingest_component.ingest(file_name, file_data)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(document) for document in documents]

    def ingest_text(self, file_name: str, text: str) -> list[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return self._ingest_data(file_name, text)

    def ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return self._ingest_data(file_name, file_data)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[IngestedDoc]:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        documents = self.ingest_component.bulk_ingest(files)
        logger.info("Finished ingestion file_name=%s", [f[0] for f in files])
        return [IngestedDoc.from_document(document) for document in documents]

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs: list[IngestedDoc] = []
        try:
            docstore = self.storage_context.docstore
            ref_docs: dict[str, RefDocInfo] | None = docstore.get_all_ref_doc_info()

            if not ref_docs:
                return ingested_docs

            for doc_id, ref_doc_info in ref_docs.items():
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_id
        )
        self.ingest_component.delete(doc_id)

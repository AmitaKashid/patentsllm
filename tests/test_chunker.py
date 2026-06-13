from patentllm.parsing.chunker import PatentChunker
from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import ParagraphRecord, PatentMetadata


def test_chunker_creates_metadata_and_description_chunks() -> None:
    metadata = PatentMetadata(
        document_id="WO_TEST_A1",
        source_file="WO_TEST_A1.pdf",
        page_count=1,
        publication_number="WO-TEST-A1",
        title="HOT MELT ADHESIVE",
        abstract="A hot melt adhesive composition.",
    )
    paragraphs = [
        ParagraphRecord(
            paragraph_id="p1",
            document_id="WO_TEST_A1",
            source_file="WO_TEST_A1.pdf",
            page_number=1,
            sequence_index=0,
            section="technical_field",
            text="The present invention relates to hot melt adhesive compositions for absorbent articles.",
            token_estimate=20,
            marker="0001",
        )
    ]

    chunks = PatentChunker(PatentParserConfig()).build_chunks(metadata, paragraphs)
    chunk_types = {chunk.chunk_type for chunk in chunks}

    assert "metadata" in chunk_types
    assert "title_abstract_claims" in chunk_types
    assert "description_paragraph_window" in chunk_types

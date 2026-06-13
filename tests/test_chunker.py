from patentllm.parsing.chunker import PatentChunker
from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import ParagraphRecord, PatentMetadata


def test_chunker_creates_non_embeddable_metadata_and_description_chunks() -> None:
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
    metadata_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "metadata")

    assert "metadata" in chunk_types
    assert "title_abstract_claims" in chunk_types
    assert "description_paragraph_window" in chunk_types
    assert metadata_chunk.embeddable is False


def test_chunker_claims_use_defensive_extraction() -> None:
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
            paragraph_id="p_claims",
            document_id="WO_TEST_A1",
            source_file="WO_TEST_A1.pdf",
            page_number=5,
            sequence_index=0,
            section="claims",
            text=(
                "20. Search report noise.\n"
                "1. A hot melt adhesive composition comprising a polymer and a tackifier.\n"
                "2. The composition according to claim 1, wherein the polymer is polyolefinic."
            ),
            token_estimate=80,
        )
    ]

    chunks = PatentChunker(PatentParserConfig()).build_chunks(metadata, paragraphs)
    claim_chunks = [chunk for chunk in chunks if chunk.chunk_type == "claim"]
    assert [chunk.metadata["claim_number"] for chunk in claim_chunks] == [1, 2]

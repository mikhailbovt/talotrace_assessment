"""The user-approved extraction policy is a prerequisite for paid media generation."""

EXTRACTION_POLICY = "pdf-inspector+gpt-5.6-luna-vision-v1"


class CorpusNotReady(ValueError):
    """A completed corrected corpus has not been activated yet."""


def require_corrected_corpus(summary: dict | None) -> None:
    if (
        not summary
        or summary.get("extraction_policy") != EXTRACTION_POLICY
        or summary.get("extraction_verified") is not True
    ):
        raise CorpusNotReady("The verified PDF-inspector plus Luna-vision corpus is not active")
    required = summary.get("vision_required_pages")
    completed = summary.get("vision_completed_pages")
    if type(required) is not int or required < 0 or completed != required:
        raise CorpusNotReady("Required vision page transcription has not been verified complete")


def validate_extracted_sources(context: dict) -> None:
    require_corrected_corpus(context.get("corpus_summary"))
    for hit in context["results"]:
        if hit.get("extractor") not in {"pdf-inspector", "gpt-5.6-luna-vision"}:
            raise CorpusNotReady("Answer source does not use the verified extraction policy")

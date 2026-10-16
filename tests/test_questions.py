from docbenchtr.questions import verify_answer_spans


def test_span_verifier_accepts_extracted_answer_with_wrapped_whitespace():
    question = {
        "document_id": "law",
        "source_pages": [4],
        "reference_answer": "Kişisel veriler, ilgili kişinin açık rızası olmaksızın aktarılamaz.",
    }
    pages = {("law", 4): "Kişisel veriler, ilgili kişinin açık rızası\n olmaksızın aktarılamaz."}
    assert verify_answer_spans(question, pages)


def test_span_verifier_rejects_wrong_page_or_missing_answer():
    question = {
        "document_id": "law",
        "source_pages": [4],
        "reference_answer": "Bu ifade kaynakta yoktur.",
    }
    assert not verify_answer_spans(question, {("law", 4): "Başka bir metin."})
    assert not verify_answer_spans(question, {("law", 5): "Bu ifade kaynakta yoktur."})

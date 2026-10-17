import json
from pathlib import Path

from docbenchtr.questions import (
    _articles_for_document,
    _pick_span,
    normalize_text,
    verify_answer_spans,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "data" / "pilot_questions.json").read_text(encoding="utf-8"))
QUESTION_BY_ID = {question["question_id"]: question for question in QUESTIONS}


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


def test_article_parser_reads_real_2547_article_22_case():
    question = QUESTION_BY_ID["yuksekogretim_2547-cross_section-2"]
    first_answer = question["reference_answer"].split(" / ")[0]
    parsed = _articles_for_document([{
        "document_id": "yuksekogretim_2547",
        "page_number": 20,
        "text": f"MADDE 22 - {first_answer}",
    }], "yuksekogretim_2547")

    assert parsed[0]["number"] == "22"
    assert "Yükseköğretim kurumlarında" in parsed[0]["page_segments"][0][1]


def test_span_picker_prefers_2547_planned_term_over_footnote_fragment():
    invalid = QUESTION_BY_ID["yuksekogretim_2547-paraphrase-1"]["reference_answer"]
    source_clause = QUESTION_BY_ID["yuksekogretim_2547-distractor-2"]["reference_answer"]
    chosen = _pick_span({"answer_spans": [
        {"page": 20, "answer": invalid},
        {"page": 24, "answer": source_clause},
    ]}, ["eğitim - öğretim", "özel bilgi ve uzmanlık"])

    assert "eğitim - öğretim" in normalize_text(chosen["answer"])
    assert " 23 22/2/2018" not in normalize_text(chosen["answer"])


def test_validator_keeps_and_reports_the_four_real_planned_term_failures():
    expected = {
        "kvkk_6698-paraphrase-2",
        "yuksekogretim_2547-paraphrase-1",
        "yuksekogretim_2547-cross_section-1",
        "yuksekogretim_2547-cross_section-2",
    }
    failed = set()
    for question in QUESTIONS:
        segments = question["reference_answer"].split(" / ")
        page_lookup = {
            (question["document_id"], page): segments[index]
            for index, page in enumerate(question["source_pages"])
        }
        if not verify_answer_spans(question, page_lookup):
            failed.add(question["question_id"])

    assert failed == expected


def test_validator_strips_2547_footnote_marker_but_requires_terms():
    answer = (
        "Yükseköğretim kurumlarında eğitim - öğretim, bilimsel araştırmalar ve yayımlar yapmak, "
        "23 22/2/2018 tarihli ve 7100 sayılı Kanunun 11 inci maddesiyle değiştirilmiştir."
    )
    question = {
        "document_id": "yuksekogretim_2547",
        "source_pages": [20],
        "reference_answer": answer,
        "planned_terms": [["eğitim - öğretim", "bilimsel araştırmalar"]],
    }
    page_lookup = {("yuksekogretim_2547", 20): f"{answer}\n20"}

    assert verify_answer_spans(question, page_lookup)
    question["planned_terms"] = [["üniversiteyi temsil"]]
    assert not verify_answer_spans(question, page_lookup)

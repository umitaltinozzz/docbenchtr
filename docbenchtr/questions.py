from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .config import DOCUMENTS, QUESTIONS_PATH, ROOT

ARTICLE_RE = re.compile(r"(?im)^\s*(?:MADDE|Madde)\s+([0-9]+(?:\s*/\s*[A-Z])?)\s*[-–—:]\s*")
WORD_RE = re.compile(r"[a-zçğıöşü]{3,}", re.IGNORECASE)
EQUIVALENT_GOLD_PAGES = {
    "tuketicinin_korunmasi_6502-direct_fact-3": {"21": [139]},
    "tuketicinin_korunmasi_6502-distractor-3": {"21": [139]},
    "tuketicinin_korunmasi_6502-cross_section-3": {"21": [139]},
    "tuketicinin_korunmasi_6502-paraphrase-1": {"12": [54]},
    "tuketicinin_korunmasi_6502-paraphrase-2": {"26": [188]},
}
EQUIVALENT_PROVISIONS = {
    139: "Mesafeli Sözleşmeler Yönetmeliği tanımlarındaki mesafeli sözleşme tanımı; 6502 m. 48 ile aynı hüküm.",
    54: "Taksitli satış sözleşmelerinde cayma hakkına ilişkin hüküm; 6502 m. 18 ile eşdeğer hüküm.",
    188: "Fiyat Etiketi Yönetmeliği m. 5; 6502 m. 54 ile aynı fiyat etiketi gerekliliği.",
}

# Question stems are tied to article topics in the source PDFs. Answer spans
# and page numbers are selected from extracted article text at run time.
QUESTION_PLAN: dict[str, dict[str, list[dict[str, Any]]]] = {
    "kvkk_6698": {
        "direct_fact": [
            {"article": "5", "question": "Kişisel veriler, ilgili kişinin açık rızası olmadan hangi şartlarda işlenebilir?", "terms": ["açık rızası aranmaksızın", "şartlardan birinin varlığı"]},
            {"article": "8", "question": "Kişisel veriler hangi koşullarda ilgili kişinin açık rızası olmadan aktarılabilir?", "terms": ["aktarılabilir", "yeterli önlemler"]},
            {"article": "12", "question": "Veri sorumlusu, kişisel verileri korumak için hangi tedbirleri almak zorundadır?", "terms": ["teknik ve idari tedbirleri"]},
        ],
        "paraphrase": [
            {"article": "10", "question": "Kişisel veriler toplanırken kişi hangi konularda bilgilendirilmelidir?", "terms": ["hangi amaçla işleneceği", "hukuki sebebi"]},
            {"article": "11", "question": "Bir kişi, kendisiyle ilgili veriler bakımından veri sorumlusundan neleri öğrenebilir veya talep edebilir?", "terms": ["haklarına sahiptir", "bilgi talep etme"]},
            {"article": "9", "question": "Kişisel veriler yurt dışına aktarılırken hangi ek güvenceler aranır?", "terms": ["yeterli korumanın bulunmaması", "Kurulun izninin bulunması"]},
        ],
        "distractor": [
            {"article": "8", "distractor_article": "5", "question": "Kişisel veri işleme istisnalarından farklı olarak, verilerin aktarımı hangi hâllerde açık rıza olmadan mümkündür?", "terms": ["aktarılabilir", "yeterli önlemler"]},
            {"article": "9", "distractor_article": "8", "question": "Yurt dışına veri aktarımında yeterli koruma yoksa hangi yazılı güvence ve izin gerekir?", "terms": ["yeterli korumanın bulunmaması", "yazılı olarak taahhüt"]},
            {"article": "12", "distractor_article": "4", "question": "Veri sorumlusu, hukuka aykırı erişimi önlemek ve verileri korumak için ne yapmalıdır?", "terms": ["hukuka aykırı olarak erişilmesini önlemek", "tedbirleri almak"]},
        ],
        "cross_section": [
            {"articles": ["5", "8"], "question": "Kişisel verilerin işlenmesi ve aktarılması için açık rıza bakımından hangi kurallar geçerlidir?", "terms": [["açık rızası aranmaksızın", "şartlardan birinin varlığı"], ["aktarılabilir", "yeterli önlemler"]]},
            {"articles": ["8", "12"], "question": "Veri aktarımı koşulları ile veri güvenliğini sağlama yükümlülüğü nelerdir?", "terms": [["aktarılabilir", "yeterli önlemler"], ["teknik ve idari tedbirleri"]]},
            {"articles": ["10", "12"], "question": "Veri sorumlusunun kişiyi bilgilendirme ve verileri koruma yükümlülükleri nelerdir?", "terms": [["hangi amaçla işleneceği", "hukuki sebebi"], ["teknik ve idari tedbirleri"]]},
        ],
    },
    "tuketicinin_korunmasi_6502": {
        "direct_fact": [
            {"article": "8", "question": "Kanuna göre bir mal hangi durumda ayıplı sayılır?", "terms": ["sözleşmeye aykırı olan maldır"]},
            {"article": "11", "question": "Ayıplı bir mal karşısında tüketici hangi seçimlik hakları kullanabilir?", "terms": ["sözleşmeden dönme", "satış bedelinden indirim"]},
            {"article": "48", "question": "Kanun, mesafeli sözleşmeyi nasıl tanımlar?", "terms": ["eş zamanlı fiziksel varlığı olmaksızın"]},
        ],
        "paraphrase": [
            {"article": "18", "question": "Taksitle alışveriş yapan tüketici sözleşmeden hangi koşullarda cayabilir?", "terms": ["herhangi bir gerekçe göstermeksizin", "cayma hakkına sahiptir"]},
            {"article": "54", "question": "Tüketici bir ürünün fiyatını satış yerinde nasıl görebilmelidir?", "terms": ["kolaylıkla görülebilir ve okunabilir", "birim fiyatını"]},
            {"article": "58", "question": "Bir mal satıldıktan sonra bakım ve onarım hizmetlerini kim sağlamakla yükümlüdür?", "terms": ["satış sonrası bakım ve onarım hizmetlerini", "sağlamak zorundadır"]},
        ],
        "distractor": [
            {"article": "8", "distractor_article": "13", "question": "Ayıplı bir malın sözleşmeye aykırı sayılmasına yol açan durumlar nelerdir?", "terms": ["taraflarca kararlaştırılmış olan örnek", "özellikleri taşımaması"]},
            {"article": "11", "distractor_article": "15", "question": "Mal ayıplı çıktığında tüketici hangi seçimlik haklardan yararlanabilir?", "terms": ["satılanı geri vermeye hazır", "sözleşmeden dönme"]},
            {"article": "48", "distractor_article": "49", "question": "Genel bir mesafeli mal veya hizmet sözleşmesi nasıl kurulur?", "terms": ["eş zamanlı fiziksel varlığı olmaksızın", "uzaktan pazarlanmasına"]},
        ],
        "cross_section": [
            {"articles": ["8", "13"], "question": "Ayıplı mal ve ayıplı hizmet hangi sözleşmeye aykırılık durumlarını kapsar?", "terms": [["sözleşmeye aykırı olan maldır"], ["sözleşmeye aykırı olan hizmettir"]]},
            {"articles": ["11", "15"], "question": "Ayıplı mal ve ayıplı hizmet için tüketici hangi seçimlik haklara sahiptir?", "terms": [["satış bedelinden indirim", "sözleşmeden dönme"], ["bedelden indirim", "sözleşmeden dönme"]]},
            {"articles": ["48", "49"], "question": "Mesafeli sözleşme ile finansal hizmetlere ilişkin mesafeli sözleşme hangi konuları kapsar?", "terms": [["eş zamanlı fiziksel varlığı olmaksızın"], ["banka hizmeti, kredi, sigorta"]]},
        ],
    },
    "yuksekogretim_2547": {
        "direct_fact": [
            {"article": "6", "question": "Yükseköğretim Kurulu nasıl bir kurul olarak tanımlanır?", "terms": ["özerkliğe ve kamu tüzel kişiliğine sahip"]},
            {"article": "13", "question": "Devlet ve vakıf üniversitelerine rektör nasıl atanır?", "terms": ["rektör, Cumhurbaşkanınca atanır"]},
            {"article": "54", "question": "Kanunda öğrenciler için sayılan disiplin cezalarından biri nedir?", "terms": ["disiplin cezaları", "kınama"]},
        ],
        "paraphrase": [
            {"article": "22", "question": "Öğretim üyelerinden eğitim ve araştırma yönünden hangi çalışmalar beklenir?", "terms": ["eğitim - öğretim", "araştırma"]},
            {"article": "23", "question": "Doktor öğretim üyesi kadrosu için adaylık süreci nasıl başlatılır?", "terms": ["kadroları rektörlükçe ilan edilir"]},
            {"article": "33", "question": "Araştırma görevlileri araştırma ve deneylerde nasıl bir görev üstlenir?", "terms": ["araştırma, inceleme ve deneylerde yardımcı olan"]},
        ],
        "distractor": [
            {"article": "23", "distractor_article": "24", "question": "Doktor öğretim üyesi kadrosuna atanma için hangi ilan ve başvuru koşulu uygulanır?", "terms": ["kadroları rektörlükçe ilan edilir"]},
            {"article": "31", "distractor_article": "33", "question": "Öğretim görevlileri hangi ders veya uzmanlık ihtiyacını karşılamak üzere atanır?", "terms": ["öğretim üyesi bulunmayan dersler", "özel bilgi ve uzmanlık"]},
            {"article": "53/A", "distractor_article": "53/D", "question": "Disiplin soruşturması başlatılırken amir hangi usulü izlemelidir?", "terms": ["yazılı olarak disiplin soruşturması başlatır"]},
        ],
        "cross_section": [
            {"articles": ["13", "14"], "question": "Üniversite yönetiminde rektörün görevi ile senatonun kuruluş ve işleyişi nasıl düzenlenmiştir?", "terms": [["rektör", "üniversiteyi temsil"], ["senato, rektörün başkanlığında"]]},
            {"articles": ["22", "33"], "question": "Öğretim üyelerinin eğitim ve araştırma görevleri ile araştırma görevlilerinin katkısı nelerdir?", "terms": [["eğitim - öğretim", "araştırma"] , ["araştırma, inceleme ve deneylerde yardımcı"]]},
            {"articles": ["53/A", "53/D"], "question": "Disiplin soruşturmasının başlatılması ve disiplin cezasının belirlenmesinde hangi ilkeler uygulanır?", "terms": [["yazılı olarak disiplin soruşturması başlatır"], ["en ağır cezayı gerektiren"]]},
        ],
    },
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    value = re.sub(r"(?m)^\s*(?:\d{1,4}|[-–—]\s*\d+\s*[-–—])\s*$", " ", value)
    # Xpdf places amendment-footnote numbers directly before their note text.
    value = re.sub(
        r"\s+\d{1,3}\s+(?=\d{1,2}/\d{1,2}/\d{4}\s+tarihli ve\s+\d+\s+sayılı Kanunun\b)",
        " ", value, flags=re.IGNORECASE,
    )
    return " ".join(value.split()).casefold()


def answer_span_issues(
    question: dict[str, Any], page_lookup: dict[tuple[str, int], str]
) -> list[dict[str, Any]]:
    """Return provenance and planned-term failures for every selected answer span."""
    document_id = question["document_id"]
    cited_pages = question["source_pages"]
    if not cited_pages:
        return [{"type": "missing_source_pages"}]
    source_texts = [page_lookup.get((document_id, page)) for page in cited_pages]
    if any(text is None for text in source_texts):
        return [{"type": "missing_source_page_text", "pages": cited_pages}]
    answer_segments = [segment.strip() for segment in question["reference_answer"].split(" / ")]
    if not all(answer_segments):
        return [{"type": "empty_answer_segment"}]
    normalized_pages = [normalize_text(text or "") for text in source_texts]
    planned_terms = question.get("planned_terms", [])
    issues: list[dict[str, Any]] = []
    for index, segment in enumerate(answer_segments):
        if not any(normalize_text(segment) in page_text for page_text in normalized_pages):
            issues.append({"type": "answer_not_found", "segment_index": index})
        terms = planned_terms[index] if index < len(planned_terms) else []
        missing = [term for term in terms if normalize_text(term) not in normalize_text(segment)]
        if missing:
            issues.append({
                "type": "planned_terms_missing",
                "segment_index": index,
                "missing_terms": missing,
            })
    return issues


def verify_answer_spans(question: dict[str, Any], page_lookup: dict[tuple[str, int], str]) -> bool:
    """Require source provenance and every planned term in each selected span."""
    return not answer_span_issues(question, page_lookup)


def verify_question_set(questions: list[dict[str, Any]], pages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    page_lookup = {(p["document_id"], p["page_number"]): p["text"] for p in pages}
    valid, dropped = [], []
    for question in questions:
        (valid if verify_answer_spans(question, page_lookup) else dropped).append(question)
    return valid, dropped


def _clean_heading(text: str) -> str:
    lines = [" ".join(line.split()).strip(" -–—:;.") for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        if len(line) < 70 and not line.endswith((".", ";", ":")) and not re.fullmatch(r"[A-ZÇĞİÖŞÜ\s\d.,()/-]+", line):
            return line
    return ""


def _articles_for_document(pages: list[dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
    doc_pages = [page for page in pages if page["document_id"] == document_id]
    articles: dict[str, dict[str, Any]] = {}
    last_article: dict[str, Any] | None = None
    for page in doc_pages:
        text = page["text"]
        matches = list(ARTICLE_RE.finditer(text))
        if not matches:
            if last_article and text.strip():
                last_article["page_segments"].append((page["page_number"], text.strip()))
            continue
        if last_article and text[:matches[0].start()].strip():
            last_article["page_segments"].append((page["page_number"], text[:matches[0].start()].strip()))
        for index, match in enumerate(matches):
            article_number = re.sub(r"\s+", "", match.group(1))
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            segment = text[match.start():end].strip()
            if article_number not in articles:
                articles[article_number] = {
                    "number": article_number,
                    "heading": _clean_heading(text[:match.start()]),
                    "page_segments": [],
                }
            articles[article_number]["page_segments"].append((page["page_number"], segment))
            last_article = articles[article_number]
    return list(articles.values())


def _answer_candidates(segment: str) -> list[str]:
    marker = ARTICLE_RE.search(segment)
    body = segment[marker.end():] if marker else segment
    body = " ".join(body.split())
    sentences = [part.strip(" -–—; \t") for part in re.split(r"(?<=[.!?])\s+", body)]
    sentences = [re.sub(r"^\d{5,6}\s+(?=[A-Za-zÇĞİÖŞÜçğıöşü]\))", "", part) for part in sentences]
    usable = [part for part in sentences if 45 <= len(part) <= 700 and len(WORD_RE.findall(part)) >= 6]
    if usable:
        return usable
    lines = [" ".join(line.split()) for line in body.splitlines() if line.strip()]
    return [line for line in lines if 45 <= len(line) <= 500 and len(WORD_RE.findall(line)) >= 6]


def _article_candidates(pages: list[dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for article in _articles_for_document(pages, document_id):
        spans = []
        for page_number, text in article["page_segments"]:
            spans.extend({"page": page_number, "answer": answer} for answer in _answer_candidates(text))
        if spans:
            article["answer_spans"] = spans
            article["first_page"] = spans[0]["page"]
            candidates.append(article)
    return candidates


def _pick_span(article: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    normalized_terms = [normalize_text(term) for term in terms]
    ranked = sorted(
        article["answer_spans"],
        key=lambda span: (
            sum(term in normalize_text(span["answer"]) for term in normalized_terms),
            len(span["answer"]),
        ),
        reverse=True,
    )
    # Choose the best available source span; the verifier reports any missing terms.
    return ranked[0]


def _make_question(
    doc_id: str, qid: str, prompt: str, spans: list[dict[str, Any]],
    question_type: str, planned_terms: list[list[str]],
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question": prompt,
        "reference_answer": " / ".join(span["answer"] for span in spans),
        "document_id": doc_id,
        "source_pages": sorted({span["page"] for span in spans}),
        "equivalent_pages": EQUIVALENT_GOLD_PAGES.get(qid, {}),
        "planned_terms": planned_terms,
        "question_type": question_type,
    }


def draft_questions(pages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drafts: list[dict[str, Any]] = []
    draft_log: list[dict[str, Any]] = []
    for document in DOCUMENTS:
        doc_id = document["document_id"]
        articles = {article["number"]: article for article in _article_candidates(pages, doc_id)}
        for question_type, specs in QUESTION_PLAN[doc_id].items():
            for index, spec in enumerate(specs, start=1):
                if question_type == "cross_section":
                    spans = [
                        _pick_span(articles[number], terms)
                        for number, terms in zip(spec["articles"], spec["terms"])
                    ]
                    question = spec["question"]
                    source_articles = spec["articles"]
                else:
                    article_number = spec["article"]
                    spans = [_pick_span(articles[article_number], spec["terms"])]
                    question = spec["question"]
                    source_articles = [article_number]
                question_id = f"{doc_id}-{question_type}-{index}"
                planned_terms = spec["terms"] if question_type == "cross_section" else [spec["terms"]]
                drafts.append(_make_question(
                    doc_id, question_id, question, spans, question_type, planned_terms
                ))
                draft_log.append({
                    "question_id": question_id,
                    "article_numbers": source_articles,
                    "distractor_article_number": spec.get("distractor_article"),
                })

    valid, failed = verify_question_set(drafts, pages)
    # Keep failed validator cases in the published question set for transparency and
    # scoring. They are identified separately for the sensitivity analysis.
    retained = valid + failed
    retained.sort(key=lambda question: next(
        index for index, draft in enumerate(drafts)
        if draft["question_id"] == question["question_id"]
    ))
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(json.dumps(retained, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    valid_ids = {question["question_id"] for question in valid}
    page_lookup = {(page["document_id"], page["page_number"]): page["text"] for page in pages}
    verification = {
        "label": "pilot set; hand-written stems, source spans and planned terms checked; not yet human-reviewed",
        "drafted_count": len(drafts),
        "verified_count": len(valid),
        "failed_count": len(failed),
        "retained_count": len(retained),
        "dropped_count": 0,
        "full_document_search": {
            "text_source": "data/processed/pages.jsonl from pinned PDFs; pdftotext -enc UTF-8 -layout (actual extractor version in data/sources.json)",
            "page_scope": "all extracted physical pages in each question's document",
            "normalization": "NFKC, soft-hyphen and line-wrap normalization, blank page-number and Xpdf amendment-footnote noise removal, whitespace collapse, casefold",
            "purpose": "Search each selected answer segment and every planned term across the complete extracted text for equivalent-page review.",
        },
        "checks": [
            {
                **item,
                "verified": item["question_id"] in valid_ids,
                "source_pages": next(question["source_pages"] for question in drafts if question["question_id"] == item["question_id"]),
                "equivalent_pages": next(question["equivalent_pages"] for question in drafts if question["question_id"] == item["question_id"]),
                "planned_terms": next(question["planned_terms"] for question in drafts if question["question_id"] == item["question_id"]),
                "issues": answer_span_issues(
                    next(question for question in drafts if question["question_id"] == item["question_id"]),
                    page_lookup,
                ),
            }
            for item in draft_log
        ],
    }
    verification["full_document_search"]["question_searches"] = _search_question_evidence(
        retained, pages
    )
    verification["equivalent_page_evidence"] = _equivalent_page_evidence(retained, pages)
    (ROOT / "data" / "question_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return retained, failed


def _search_question_evidence(
    questions: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pages_by_document: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        pages_by_document.setdefault(page["document_id"], []).append(page)
    evidence = []
    for question in questions:
        full_text = pages_by_document.get(question["document_id"], [])
        normalized = [(page["page_number"], normalize_text(page["text"])) for page in full_text]
        segments = question["reference_answer"].split(" / ")
        evidence.append({
            "question_id": question["question_id"],
            "document_page_count": len(full_text),
            "answer_segment_pages": [
                {
                    "segment_index": index,
                    "pages": [page for page, text in normalized if normalize_text(segment) in text],
                }
                for index, segment in enumerate(segments)
            ],
            "planned_term_pages": [
                {
                    "segment_index": index,
                    "term_pages": {
                        term: [page for page, text in normalized if normalize_text(term) in text]
                        for term in terms
                    },
                }
                for index, terms in enumerate(question.get("planned_terms", []))
            ],
        })
    return evidence


def _equivalent_page_evidence(
    questions: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    page_lookup = {(page["document_id"], page["page_number"]): page["text"] for page in pages}
    evidence = []
    for question in questions:
        for original_page, equivalent_pages in question.get("equivalent_pages", {}).items():
            for page_number in equivalent_pages:
                page_text = normalize_text(page_lookup[(question["document_id"], page_number)])
                phrase = next(
                    (normalize_text(term) for group in question.get("planned_terms", [])
                     for term in group if normalize_text(term) in page_text),
                    "",
                )
                start = page_text.find(phrase) if phrase else -1
                evidence.append({
                    "question_id": question["question_id"],
                    "original_gold_page": int(original_page),
                    "equivalent_page": page_number,
                    "provision": EQUIVALENT_PROVISIONS[page_number],
                    "matched_planned_phrase": phrase,
                    "normalized_extracted_text_excerpt": (
                        page_text[max(0, start - 130):start + len(phrase) + 260]
                        if start >= 0 else page_text[:500]
                    ),
                    "text_origin": (
                        "data/processed/pages.jsonl; pinned PDF processed with pdftotext "
                        "version 4.06 [www.xpdfreader.com], -enc UTF-8 -layout"
                    ),
                })
    return evidence

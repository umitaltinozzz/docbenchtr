from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .config import DOCUMENTS, QUESTIONS_PATH, ROOT

ARTICLE_RE = re.compile(r"(?im)^\s*(?:MADDE|Madde)\s+([0-9]+(?:\s*/\s*[A-Z])?)\s*[-–—:]\s*")
WORD_RE = re.compile(r"[a-zçğıöşü]{3,}", re.IGNORECASE)

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
    return " ".join(value.split()).casefold()


def verify_answer_spans(question: dict[str, Any], page_lookup: dict[tuple[str, int], str]) -> bool:
    """Require every exact answer segment to occur on at least one cited page."""
    document_id = question["document_id"]
    cited_pages = question["source_pages"]
    if not cited_pages:
        return False
    source_texts = [page_lookup.get((document_id, page)) for page in cited_pages]
    if any(text is None for text in source_texts):
        return False
    answer_segments = [segment.strip() for segment in question["reference_answer"].split(" / ")]
    if not all(answer_segments):
        return False
    normalized_pages = [normalize_text(text or "") for text in source_texts]
    return all(
        any(normalize_text(segment) in page_text for page_text in normalized_pages)
        for segment in answer_segments
    )


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
    if normalized_terms and not any(term in normalize_text(ranked[0]["answer"]) for term in normalized_terms):
        # A mismatch is surfaced for the verifier to drop; do not invent a fallback answer.
        return ranked[0]
    return ranked[0]


def _make_question(doc_id: str, qid: str, prompt: str, spans: list[dict[str, Any]], question_type: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "question": prompt,
        "reference_answer": " / ".join(span["answer"] for span in spans),
        "document_id": doc_id,
        "source_pages": sorted({span["page"] for span in spans}),
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
                drafts.append(_make_question(doc_id, question_id, question, spans, question_type))
                draft_log.append({
                    "question_id": question_id,
                    "article_numbers": source_articles,
                    "distractor_article_number": spec.get("distractor_article"),
                })

    valid, dropped = verify_question_set(drafts, pages)
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(json.dumps(valid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    valid_ids = {question["question_id"] for question in valid}
    verification = {
        "label": "pilot set, auto-drafted from source text, span-verified; not yet human-reviewed",
        "drafted_count": len(drafts),
        "verified_count": len(valid),
        "dropped_count": len(dropped),
        "checks": [
            {
                **item,
                "verified": item["question_id"] in valid_ids,
                "source_pages": next(question["source_pages"] for question in drafts if question["question_id"] == item["question_id"]),
            }
            for item in draft_log
        ],
    }
    (ROOT / "data" / "question_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return valid, dropped

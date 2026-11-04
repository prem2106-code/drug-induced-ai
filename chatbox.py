import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


def build_local_chat_response(
    prompt,
    prediction=None,
    drug_catalogue=None,
    ollama_unavailable=False,
):
    """Answer common safety-chat questions without relying on Ollama."""
    question = prompt.casefold()
    prediction = prediction or {}
    drug_catalogue = drug_catalogue or {}

    if re.search(r"\b(?:hi+|hello+|hey+|vanakkam)\b", question):
        return (
            "👋🩺 Welcome to this website! I'm your Drug Safety AI Assistant. "
            "Ask me about a medicine, its general use, or a risk result."
        )

    safe_question = any(
        phrase in question
        for phrase in (
            "everything fine",
            "all fine",
            "am i fine",
            "is it fine",
            "is everything ok",
            "is everything okay",
            "anything wrong",
            "is it safe",
            "safe to take",
            "nothing problem",
        )
    )
    if safe_question:
        if prediction:
            risk = float(prediction["risk_probability"])
            label = prediction["risk_label"]
            drug = prediction["drug_name"]
            return (
                f"The dashboard's preliminary estimate for {drug} is "
                f"{risk:.1%} ({label}). This result cannot confirm that "
                "everything is fine or that a medicine is safe for a person. "
                "The model uses synthetic data and is not a diagnosis; ask a "
                "qualified healthcare professional before starting, stopping, "
                "or changing a medicine."
            )
        return (
            "I can't confirm that everything is fine or that a medicine is "
            "safe without a clinical assessment. This website is an educational "
            "prototype, not a diagnosis. Please ask a qualified healthcare "
            "professional before starting, stopping, or changing a medicine."
        )

    named_drug = next(
        (
            name
            for name in drug_catalogue
            if name.casefold() in question
            or (
                name == "Statins (Atorvastatin)"
                and "atorvastatin" in question
            )
        ),
        None,
    )
    drug_name = named_drug or prediction.get("drug_name")
    disease_question = any(
        term in question
        for term in ("disease", "diseases", "condition", "conditions")
    ) and any(term in question for term in ("risk", "high", "cause", "develop"))

    if disease_question:
        conditions_by_drug = {
            "Hydralazine": "drug-induced lupus-like syndrome and, rarely, vasculitis",
            "Procainamide": "drug-induced lupus-like syndrome",
            "Isoniazid": "a rare lupus-like syndrome",
            "Minocycline": (
                "lupus-like syndrome, autoimmune hepatitis, or vasculitis"
            ),
            "Methyldopa": "autoimmune hemolytic anemia",
            "Infliximab": "a lupus-like syndrome",
            "Etanercept": "a lupus-like syndrome",
            "Adalimumab": "a lupus-like syndrome",
        }
        specific_conditions = conditions_by_drug.get(drug_name)
        if specific_conditions:
            condition_text = (
                f"For {drug_name}, reported immune-related examples include "
                f"{specific_conditions}."
            )
        else:
            condition_text = (
                "Depending on the specific medicine, examples of drug-related "
                "immune conditions can include lupus-like illness, autoimmune "
                "hepatitis, vasculitis, or autoimmune blood-cell disorders."
            )

        current_risk = ""
        if prediction:
            current_risk = (
                f"The current dashboard result is {prediction['risk_label']} "
                f"({float(prediction['risk_probability']):.1%}). "
            )
        return (
            f"{current_risk}{condition_text} The model does not predict which "
            "disease, if any, a person will develop, and a high score does not "
            "prove that a drug caused one. Discuss symptoms or treatment concerns "
            "with a healthcare professional."
        )

    if prediction and any(
        term in question for term in ("risk", "prediction", "score")
    ):
        return (
            f"The dashboard's preliminary estimate for "
            f"{prediction['drug_name']} is "
            f"{float(prediction['risk_probability']):.1%} "
            f"({prediction['risk_label']}). This is a research signal from "
            "synthetic data, not a diagnosis or proof that the medicine will "
            "cause an autoimmune condition."
        )

    medicine_question = any(
        term in question
        for term in (
            "medicine",
            "medication",
            "drug",
            "used for",
            "what is",
            "what does",
            "purpose",
            "treat",
        )
    )
    if medicine_question:
        if not drug_name or drug_name not in drug_catalogue:
            return (
                "I can give basic information about medicines in this demo. "
                "Please name the medicine you mean. I can't recommend a dose "
                "or decide whether it is suitable for you."
            )

        drug_class = drug_catalogue[drug_name]["class"]
        if any(term in question for term in ("side effect", "adverse effect")):
            return (
                f"This demo does not contain a complete side-effect guide for "
                f"{drug_name}. Possible side effects depend on the person and "
                "prescription; check the official patient leaflet or ask a "
                "pharmacist or healthcare professional."
            )

        class_uses = {
            "Antihypertensive": "high blood pressure",
            "Antiarrhythmic": "abnormal heart rhythms",
            "Antitubercular": "tuberculosis",
            "Antibiotic": "some bacterial infections",
            "TNF-inhibitor": "some inflammatory or autoimmune conditions",
            "Lipid-lowering": "high cholesterol",
            "Antidiabetic": "type 2 diabetes",
            "Anti-inflammatory": "inflammation and related symptoms",
            "Anticonvulsant": "seizures and, depending on the medicine, other conditions",
            "Antipsychotic": "some mental health conditions",
            "NSAID": "pain, fever, or inflammation",
            "Analgesic": "pain or fever",
            "Hormone": "thyroid hormone replacement",
            "PPI": "acid-related stomach conditions",
        }
        use = class_uses.get(drug_class)
        if use:
            return (
                f"{drug_name} is listed in this demo as {drug_class}; medicines "
                f"in this class are generally used for {use}. This is general "
                "information, not personal medical advice. The app's risk score "
                "is a preliminary estimate from synthetic data, not a measure of "
                "all side effects."
            )
        return (
            f"{drug_name} is listed in this demo as a {drug_class}. Its use and "
            "risks depend on the person and prescription; please confirm details "
            "with a pharmacist or healthcare professional."
        )

    if ollama_unavailable:
        return (
            "Ollama is unavailable, but the built-in offline guide is ready. "
            "I can answer greetings, basic questions about medicines in this "
            "demo, and questions about the dashboard's preliminary risk result. "
            "For other medical questions, please consult a qualified healthcare "
            "professional."
        )

    return None


def _duckduckgo_search(query, max_results=5):
    """Perform a simple DuckDuckGo HTML search and return top results.

    Returns a list of dicts: {'title', 'link', 'snippet'}
    """
    q = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={q}"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Bot/0.1; +https://example.com)"
    }

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    results = []

    containers = soup.find_all("div", class_=lambda v: v and "result" in v)

    for c in containers:
        a = c.find("a")

        if not a or not a.get("href"):
            continue

        title = a.get_text(strip=True)
        link = a.get("href")

        snippet_tag = c.find("a")
        snippet = ""

        # try to find a nearby snippet
        p = c.find("p")
        if p:
            snippet = p.get_text(strip=True)
        else:
            # fallback: use title if no snippet
            snippet = title

        results.append({
            "title": title,
            "link": link,
            "snippet": snippet,
        })

        if len(results) >= max_results:
            break

    return results


def web_search(query, max_results=5):
    """Public function to search the web and return results.

    If SERPAPI_API_KEY is provided, try to use serpapi (if installed).
    Otherwise, fall back to DuckDuckGo scraping.
    """
    # Try SerpAPI if key present and library available
    key = os.getenv("SERPAPI_API_KEY", "").strip()

    if key:
        try:
            from serpapi import GoogleSearch

            params = {
                "q": query,
                "engine": "google",
                "hl": "en",
                "num": max_results,
                "api_key": key,
            }

            search = GoogleSearch(params)
            data = search.get_dict()

            results = []

            for r in data.get("organic_results", [])[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("snippet", ""),
                })

            if results:
                return results

        except Exception:
            # fall through to DuckDuckGo
            pass

    # fallback
    return _duckduckgo_search(query, max_results=max_results)


def summarize_results(results, max_chars=400):
    """Create a short, conservative summary from search snippets."""
    if not results:
        return "No web results found to answer this question."

    texts = [r.get("snippet", "") for r in results if r.get("snippet")]
    joined = " \n\n ".join(texts)

    # simple truncation-based summary
    summary = joined.strip()

    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."

    return summary


def answer_query(query, max_results=5):
    """Run a web search and return a concise answer + sources."""
    results = web_search(query, max_results=max_results)

    summary = summarize_results(results)

    sources = [
        {"title": r.get("title", ""), "link": r.get("link", "")}
        for r in results
    ]

    answer = f"Based on top web results:\n\n{summary}\n\nSources:\n"

    for s in sources:
        if s["link"]:
            answer += f"- {s['title']}: {s['link']}\n"

    return {"answer": answer, "sources": sources}

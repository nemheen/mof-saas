from google import genai
from google.genai import types



# ADC (gcloud auth application-default login)
client = genai.Client(
    vertexai=True,
    project="gen-lang-client-0135398585",
    location="us-central1",
)

MODEL_NAME = "gemini-2.5-flash-lite"
RAG_CORPUS_PATH = (
    "projects/gen-lang-client-0135398585/locations/us-central1/ragCorpora/6917529027641081856"
)

def _rag_tools():
    return [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(rag_corpus=RAG_CORPUS_PATH)
                    ]
                )
            )
        )
    ]

def ask_gemini_rag(user_query: str) -> str:
    # contents = [types.Content(role="user", parts=[types.Part.from_text(user_query)])]
    contents = [types.Content(role="user", parts=[types.Part(text=user_query)])]
    config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=1024,
        tools=_rag_tools(),
    )
    resp = client.models.generate_content(
        model=MODEL_NAME, contents=contents, config=config
    )
    return resp.text

def ask_gemini_recommendation(user_query: str) -> str:
    """
    LLM-based material similarity/synthesis recommendation.
    Reuses RAG so answers are grounded in your corpus.
    """
    prompt = (
        "Recommend 3–6 MOF materials based on the requirement below. "
        "For each item, include: short name/ID, 1–2 key properties (e.g., LCD/PLD/ASA/Has_OMS if known), "
        "and a one-line reason. If uncertain, explicitly say so.\n\n"
        f"Requirement: {user_query}"
    )
    return ask_gemini_rag(prompt)

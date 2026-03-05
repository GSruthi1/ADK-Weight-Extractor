from google.adk.agents import LlmAgent

MODEL = "gemini-2.5-flash"

root_agent = LlmAgent(
    name="net_weight_agent",
    model=MODEL,
    description="Extracts net weight from a label image and returns JSON.",
    instruction="""
You are an extraction agent.

Task:
- Read the label image and extract ONLY the net weight value.
- Return STRICT JSON ONLY (no extra text).

Output JSON format:
{
  "net_weight": "31.60 lb",
  "confidence": 0.94,
  "reason": "short reason"
}

Rules:
- net_weight must be in the format NN.NN lb
- confidence must be a number between 0 and 1
- If you cannot find net weight, return:
  { "net_weight": null, "confidence": 0.0, "reason": "not found" }
"""
)
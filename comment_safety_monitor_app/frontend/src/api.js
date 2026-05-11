const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function readApiError(response) {
  try {
    const data = await response.json();
    return data?.detail || "The request could not be completed.";
  } catch {
    return "The request could not be completed.";
  }
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    throw new Error("Backend is not available. Start the API server and try again.");
  }

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
  return response.json();
}

export function analyzeText(text) {
  return requestJson("/api/analyze-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
}

export function inspectFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson("/api/inspect-file", {
    method: "POST",
    body: formData
  });
}

export function analyzeFile(file, textColumn) {
  const formData = new FormData();
  formData.append("file", file);
  if (textColumn) {
    formData.append("text_column", textColumn);
  }
  return requestJson("/api/analyze-file", {
    method: "POST",
    body: formData
  });
}

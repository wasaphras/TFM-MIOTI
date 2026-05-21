/** Strip Gemini/LangChain stringified content-block payloads from SSE tokens. */
export function sseTokenToPlainText(data: string): string {
  const t = data.trim();
  if (!t.startsWith("[{") || !t.includes("text")) {
    return data;
  }
  const parts: string[] = [];
  const re = /['"]text['"]\s*:\s*['"]((?:[^'"\\]|\\.)*)['"]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(t)) !== null) {
    if (m[1]) parts.push(m[1]);
  }
  return parts.length > 0 ? parts.join("") : "";
}

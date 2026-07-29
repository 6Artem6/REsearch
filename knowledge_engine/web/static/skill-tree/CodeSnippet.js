import React, { useEffect, useRef } from "react";
import { highlightCodeInRoot } from "./codeHighlight.js";

export function CodeSnippet({ code, language }) {
  const ref = useRef(null);
  const text = (code || "").trim();
  if (!text) return null;

  useEffect(() => {
    if (ref.current) highlightCodeInRoot(ref.current);
  }, [text, language]);

  const langClass =
    language && language !== "auto"
      ? `language-${language}`
      : "";

  return React.createElement(
    "pre",
    { className: "code-snippet hljs-snippet", ref },
    React.createElement("code", { className: langClass }, text),
  );
}

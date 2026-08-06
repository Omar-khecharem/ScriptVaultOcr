import { useEffect, useRef, useState } from "react";
import { BoldIcon, CheckIcon, CopyIcon, EraserIcon, ItalicIcon, UnderlineIcon } from "./icons.jsx";

const FONT_SIZES = [10, 11, 12, 14, 16, 18, 20, 24, 28];

export default function EditorPanel({ fileKey, text, placeholder, onChange }) {
  const editorRef = useRef(null);
  const lastSyncedRef = useRef("");
  const [fontSize, setFontSize] = useState("12");
  const [count, setCount] = useState(0);
  const [copied, setCopied] = useState(false);
  const [formats, setFormats] = useState({ bold: false, italic: false, underline: false });

  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    if (lastSyncedRef.current !== text) {
      el.innerHTML = escapeHtml(text);
      lastSyncedRef.current = text;
    }
    setCount(text.length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileKey, text]);

  function handleInput() {
    const value = editorRef.current.innerText;
    lastSyncedRef.current = value;
    setCount(value.length);
    onChange?.(value);
  }

  function exec(command, value = null) {
    document.execCommand(command, false, value);
    editorRef.current?.focus();
    handleInput();
    syncFormats();
  }

  function applyFontSize(sizePt) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;
    const range = selection.getRangeAt(0);
    if (range.collapsed) {
      const marker = document.createElement("span");
      marker.style.fontSize = `${sizePt}pt`;
      range.insertNode(marker);
      range.setStartAfter(marker);
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);
    } else {
      const span = document.createElement("span");
      span.style.fontSize = `${sizePt}pt`;
      try {
        range.surroundContents(span);
      } catch {
        const contents = range.extractContents();
        span.appendChild(contents);
        range.insertNode(span);
      }
    }
    editorRef.current?.focus();
    handleInput();
  }

  function syncFormats() {
    setFormats({
      bold: document.queryCommandState("bold"),
      italic: document.queryCommandState("italic"),
      underline: document.queryCommandState("underline"),
    });
  }

  function clear() {
    const el = editorRef.current;
    if (!el) return;
    el.innerHTML = "";
    handleInput();
  }

  function copy() {
    const text = editorRef.current?.innerText || "";
    if (!text) return;
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  }

  return (
    <section className="card card-right">
      <div className="card-head">
        <h2 className="card-title">Texte reconnu</h2>
      </div>
      <div className="card-body">
        <div className="ed-toolbar">
          <button
            type="button"
            className={`ed-btn${formats.bold ? " active" : ""}`}
            title="Gras (Ctrl+B)"
            onClick={() => exec("bold")}
          >
            <BoldIcon />
          </button>
          <button
            type="button"
            className={`ed-btn${formats.italic ? " active" : ""}`}
            title="Italique (Ctrl+I)"
            onClick={() => exec("italic")}
          >
            <ItalicIcon />
          </button>
          <button
            type="button"
            className={`ed-btn${formats.underline ? " active" : ""}`}
            title="Souligné (Ctrl+U)"
            onClick={() => exec("underline")}
          >
            <UnderlineIcon />
          </button>
          <select
            className="ed-select"
            aria-label="Taille de police"
            value={fontSize}
            onChange={(event) => {
              const size = event.target.value;
              setFontSize(size);
              applyFontSize(size);
            }}
          >
            {FONT_SIZES.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
          <span className="ed-divider" />
          <span className="ed-count">
            {count} caractère{count === 1 ? "" : "s"}
          </span>
          <span className="spacer" />
          <button
            type="button"
            className="ed-btn"
            onClick={copy}
            disabled={count === 0}
            title="Copier le texte"
          >
            {copied ? <CheckIcon /> : <CopyIcon />}
          </button>
          <button
            type="button"
            className="ed-btn"
            onClick={clear}
            disabled={count === 0}
            title="Effacer le contenu"
          >
            <EraserIcon />
          </button>
        </div>
        <div
          ref={editorRef}
          className="editor"
          contentEditable
          data-placeholder={placeholder || "Le texte reconnu s'affichera ici. Vous pouvez le corriger avant l'exportation."}
          suppressContentEditableWarning
          onInput={handleInput}
          onMouseUp={syncFormats}
          onKeyUp={syncFormats}
        />
      </div>
    </section>
  );
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

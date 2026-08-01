import { useEffect, useRef, useState } from "react";

const FONT_SIZES = [10, 11, 12, 14, 16, 18, 20, 24, 28];

/**
 * Éditeur de texte corrigeable (contentEditable) avec mise en forme
 * B/I/U + taille de police + compteur de caractères.
 *
 * Le contenu n'est synchronisé que lors d'un changement de fichier
 * sélectionné (``fileKey``) : pendant l'édition, React ne touche pas au DOM.
 */
export default function EditorPanel({ fileKey, text, placeholder, onChange }) {
  const editorRef = useRef(null);
  const [fontSize, setFontSize] = useState("12");
  const [count, setCount] = useState(0);
  const [formats, setFormats] = useState({ bold: false, italic: false, underline: false });

  useEffect(() => {
    const el = editorRef.current;
    if (!el) return;
    el.innerHTML = escapeHtml(text);
    setCount(text.length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileKey]);

  function handleInput() {
    setCount(editorRef.current.innerText.length);
    onChange?.(editorRef.current.innerText);
  }

  function exec(command, value = null) {
    document.execCommand(command, false, value);
    editorRef.current?.focus();
    handleInput();
    syncFormats();
  }

  /** Applique une taille de police à la sélection (span dédié). */
  function applyFontSize(sizePt) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;
    const range = selection.getRangeAt(0);
    if (range.collapsed) {
      // Pas de sélection : on définit la taille par défaut du prochain contenu
      // en insérant un span vide ancré à la position du curseur.
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

  return (
    <div className="card right">
      <h3>Éditeur de texte — corrigez le résultat de l'OCR</h3>
      <div className="editor-toolbar">
        <button
          type="button"
          className={`toolbtn${formats.bold ? " active" : ""}`}
          title="Gras (Ctrl+B)"
          onClick={() => exec("bold")}
        >
          B
        </button>
        <button
          type="button"
          className={`toolbtn${formats.italic ? " active" : ""}`}
          title="Italique (Ctrl+I)"
          onClick={() => exec("italic")}
        >
          I
        </button>
        <button
          type="button"
          className={`toolbtn${formats.underline ? " active" : ""}`}
          title="Souligné (Ctrl+U)"
          onClick={() => exec("underline")}
        >
          U
        </button>
        <select
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
        <span className="count">
          {count} caractère{count === 1 ? "" : "s"}
        </span>
        <span style={{ flex: 1 }} />
        <button type="button" className="toolbtn" onClick={clear} disabled={count === 0}>
          Effacer
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
  );
}

/** Échappe le HTML (le contenu provient de l'OCR, jamais de l'utilisateur). */
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

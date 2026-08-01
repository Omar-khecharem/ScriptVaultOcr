import { useRef, useState } from "react";

/**
 * Zone de glisser-déposer : accepte images + PDF, émet la liste des fichiers.
 */
export default function DropZone({ onFiles }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length) onFiles(files);
  }

  return (
    <div
      className={`dropzone${dragging ? " dragover" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        handleFiles(event.dataTransfer.files);
      }}
    >
      <div className="icon">＋</div>
      <div className="title">Glissez-déposez vos fichiers ici</div>
      <div className="hint">
        PNG · JPG · TIFF · WebP · PDF — ou cliquez pour parcourir
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp,.pdf"
        style={{ display: "none" }}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
    </div>
  );
}

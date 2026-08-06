import { useRef, useState } from "react";
import { UploadIcon } from "./icons.jsx";

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
      <span className="dz-ring">
        <UploadIcon size={26} />
      </span>
      <span className="dz-title">Glissez-déposez vos documents ici</span>
      <span className="dz-hint">PNG · JPG · TIFF · WebP · PDF — traitement 100% local</span>
      <button
        type="button"
        className="btn btn-primary dz-btn"
        onClick={(event) => {
          event.stopPropagation();
          inputRef.current?.click();
        }}
      >
        <UploadIcon />
        <span>Parcourir</span>
      </button>
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

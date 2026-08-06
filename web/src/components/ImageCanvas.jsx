import { useEffect, useRef, useState } from "react";
import {
  EyeIcon,
  EyeOffIcon,
  ImageIcon,
  MaximizeIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from "./icons.jsx";

export function boxColor(confidence) {
  if (confidence >= 0.85) return "#34d399";
  if (confidence >= 0.6) return "#fbbf24";
  return "#f87171";
}

function roundRect(ctx, x, y, w, h, radius) {
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, radius);
  } else {
    ctx.rect(x, y, w, h);
  }
}

export default function ImageCanvas({ previewUrl, boxes = [], message }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const [showBoxes, setShowBoxes] = useState(true);
  const dragRef = useRef(null);

  const visibleBoxes = showBoxes ? boxes : [];
  const zoomPct = Math.round(view.scale * 100);

  useEffect(() => {
    imageRef.current = null;
    setView({ scale: 1, x: 0, y: 0 });
    if (!previewUrl) return undefined;
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      draw();
    };
    img.src = previewUrl;
    return () => {
      img.onload = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewUrl]);

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, visibleBoxes, previewUrl]);

  function draw() {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const margin = 8;
    const availW = Math.max(1, rect.width - 2 * margin);
    const availH = Math.max(1, rect.height - 2 * margin);
    const fit = Math.min(availW / img.width, availH / img.height);
    const scale = fit * view.scale;
    const drawW = img.width * scale;
    const drawH = img.height * scale;
    const x0 = (rect.width - drawW) / 2 + view.x;
    const y0 = (rect.height - drawH) / 2 + view.y;

    ctx.drawImage(img, x0, y0, drawW, drawH);

    for (const item of visibleBoxes) {
      const { box, text, confidence } = item;
      if (!box || box.length < 3) continue;
      ctx.beginPath();
      box.forEach(([px, py], index) => {
        const x = x0 + px * scale;
        const y = y0 + py * scale;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = boxColor(confidence);
      ctx.lineWidth = 2;
      ctx.stroke();

      if (text && view.scale >= 0.5) {
        const [bx, by] = box[0];
        const labelX = x0 + bx * scale;
        const labelY = y0 + by * scale;
        ctx.font = "11px 'Segoe UI', sans-serif";
        const textW = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
        ctx.beginPath();
        roundRect(ctx, labelX, labelY - 19, textW + 10, 16, 4);
        ctx.fill();
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(text, labelX + 5, labelY - 7);
      }
    }
  }

  function onWheel(event) {
    if (!imageRef.current) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const factor = 1.12 ** (-event.deltaY / 100);
    setView((prev) => {
      const scale = Math.min(10, Math.max(0.25, prev.scale * factor));
      const ratio = scale / (prev.scale || 1);
      const x = event.clientX - rect.left - (event.clientX - rect.left - prev.x) * ratio;
      const y = event.clientY - rect.top - (event.clientY - rect.top - prev.y) * ratio;
      return { scale, x, y };
    });
  }

  function onMouseDown(event) {
    if (!imageRef.current || event.button !== 0) return;
    dragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      origin: view,
    };
    event.currentTarget.style.cursor = "grabbing";
  }

  function onMouseMove(event) {
    if (!dragRef.current) return;
    const { startX, startY, origin } = dragRef.current;
    setView({
      ...origin,
      x: origin.x + (event.clientX - startX),
      y: origin.y + (event.clientY - startY),
    });
  }

  function onMouseUp(event) {
    dragRef.current = null;
    event.currentTarget.style.cursor = "";
  }

  function zoomBy(factor) {
    setView((prev) => ({
      ...prev,
      scale: Math.min(10, Math.max(0.25, prev.scale * factor)),
    }));
  }

  function resetView() {
    setView({ scale: 1, x: 0, y: 0 });
  }

  if (!previewUrl) {
    return (
      <div className="canvas-wrap">
        <div className="canvas-placeholder">
          <span className="ph-ring">
            <ImageIcon size={22} />
          </span>
          <span>{message || "Sélectionnez un fichier pour afficher le document"}</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="canvas-wrap"
      onWheel={onWheel}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      <canvas ref={canvasRef} />
      <div
        className="canvas-tools"
        role="toolbar"
        aria-label="Zoom et affichage"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="canvas-tool"
          onClick={() => zoomBy(1.25)}
          title="Zoom avant"
        >
          <ZoomInIcon />
        </button>
        <span className="canvas-zoom">{zoomPct}%</span>
        <button
          type="button"
          className="canvas-tool"
          onClick={() => zoomBy(0.8)}
          title="Zoom arrière"
        >
          <ZoomOutIcon />
        </button>
        <button
          type="button"
          className="canvas-tool"
          onClick={resetView}
          title="Ajuster la vue"
        >
          <MaximizeIcon />
        </button>
        <span className="canvas-sep" />
        <button
          type="button"
          className={`canvas-tool${showBoxes ? " active" : ""}`}
          onClick={() => setShowBoxes((value) => !value)}
          title={showBoxes ? "Masquer les boîtes OCR" : "Afficher les boîtes OCR"}
        >
          {showBoxes ? <EyeIcon /> : <EyeOffIcon />}
        </button>
      </div>
      <span className="canvas-hint">Molette : zoom · Glisser : déplacer</span>
    </div>
  );
}

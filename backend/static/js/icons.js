// Inline SVG icons. Kept in one file so every icon shares the same stroke
// weight and optical size — large and unambiguous, never decorative.

const SVG_NS = "http://www.w3.org/2000/svg";

const PATHS = {
  mic: [
    "M12 2.8a3.2 3.2 0 0 0-3.2 3.2v6a3.2 3.2 0 0 0 6.4 0V6A3.2 3.2 0 0 0 12 2.8z",
    "M5 11v1a7 7 0 0 0 14 0v-1",
    "M12 19v3",
  ],
  stop: ["M7.5 7.5h9v9h-9z"],
  book: [
    "M4 4.5A1.5 1.5 0 0 1 5.5 3H19v15H5.5A1.5 1.5 0 0 0 4 19.5z",
    "M4 19.5A1.5 1.5 0 0 1 5.5 21H19",
  ],
  home: ["M3.5 10.5 12 3.5l8.5 7", "M5.5 9.4V20h13V9.4"],
  calendar: [
    "M4 6.5h16v14H4z",
    "M4 10.5h16",
    "M8.5 3.5v4",
    "M15.5 3.5v4",
  ],
  sparkle: [
    "M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z",
    "M18.5 16.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z",
  ],
  photo: [
    "M3.5 6.5h17v11h-17z",
    "M3.5 14.5 8 10.5l3.5 3 3-2.5 6 5",
    "M15.5 9.2h.01",
  ],
  plus: ["M12 5v14", "M5 12h14"],
  pen: ["M4 20h4L19.5 8.5a2.1 2.1 0 0 0-3-3L5 17v3z"],
  trash: ["M4.5 6.5h15", "M9.5 6.5V4.5h5v2", "M6.5 6.5 7.5 20h9l1-13.5"],
  check: ["M5 12.5 10 17.5 19 7"],
};

export function icon(name, { size = 24, className = "" } = {}) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.9");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  if (className) svg.setAttribute("class", className);
  for (const d of PATHS[name] || []) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.appendChild(path);
  }
  return svg;
}

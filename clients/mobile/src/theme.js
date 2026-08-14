// Same warm paper palette as the web client, same reasoning: no dark mode,
// no neon, large type, and colour pairs that clear WCAG AA for body text.

export const colors = {
  paper: "#f4efe4",
  paper2: "#faf6ee",
  surface: "#fffdf8",
  surface2: "#f3ece0",
  surface3: "#e9e0d0",

  text: "#2c2820",
  muted: "#5b5449",
  faint: "#7d7466",

  line: "#e2d9c8",
  lineStrong: "#c9bca4",

  accent: "#36675b",
  accentDeep: "#28503f",
  accentSoft: "#dde9e3",
  accentInk: "#fffdf8",

  clay: "#a54e37",
  claySoft: "#f7e5df",
  gold: "#7f6224",
  goldSoft: "#f4ebd7",

  // Legacy aliases, so the dormant analysis screens keep rendering.
  bg: "#f4efe4",
  panel: "#fffdf8",
  panel2: "#f3ece0",
  border: "#e2d9c8",
  accent2: "#28503f",
  danger: "#a54e37",
  warn: "#7f6224",
  ok: "#36675b",
};

// Deliberately larger than a typical mobile scale.
export const type = {
  xs: 15,
  sm: 17,
  base: 19,
  md: 21,
  lg: 24,
  xl: 29,
  xxl: 34,
};

export const spacing = {
  xs: 6,
  sm: 10,
  md: 16,
  lg: 22,
  xl: 30,
};

export const radius = {
  sm: 13,
  md: 18,
  pill: 999,
};

export const TAP = 56;

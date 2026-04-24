import { api } from "../api.js";
import { el, MENTION_LABELS } from "./utils.js";

const KIND_FILTERS = [
  { key: "", label: "Tümü" },
  { key: "PERSON", label: "Kişi" },
  { key: "LOCATION", label: "Yer" },
  { key: "TIME", label: "Zaman" },
  { key: "EVENT", label: "Olay" },
  { key: "ORG", label: "Kurum" },
];

export async function render(root) {
  root.innerHTML = "";
  const state = { kind: "" };

  const filters = el("div", { class: "filter-row" });
  KIND_FILTERS.forEach((f) => {
    const b = el("button", { onclick: () => pick(f.key) }, [f.label]);
    if (f.key === state.kind) b.classList.add("active");
    filters.appendChild(b);
  });
  root.appendChild(filters);

  const search = el("input", {
    type: "text",
    placeholder: "Düğümlerde ara...",
    style: "width:100%; padding:10px 12px; margin-bottom:10px; background:var(--bg-elev-2); color:var(--fg); border:1px solid var(--border); border-radius:10px;",
  });
  root.appendChild(search);

  const listRoot = el("div", {});
  root.appendChild(listRoot);

  async function load() {
    const params = {};
    if (state.kind) params.kind = state.kind;
    if (search.value.trim()) params.q = search.value.trim();
    const nodes = await api.listNodes(params);
    paint(nodes);
  }

  function paint(nodes) {
    listRoot.innerHTML = "";
    if (!nodes || nodes.length === 0) {
      listRoot.appendChild(el("div", { class: "empty-state" }, [
        el("div", { class: "big-icon" }, ["◇"]),
        el("p", {}, ["Henüz bu türde düğüm yok. Kayıt yapıp bağlantı kur."]),
      ]));
      return;
    }
    for (const n of nodes) {
      listRoot.appendChild(card(n));
    }
  }

  function pick(key) {
    state.kind = key;
    Array.from(filters.querySelectorAll("button")).forEach((b, i) =>
      b.classList.toggle("active", KIND_FILTERS[i].key === key)
    );
    load();
  }

  let handle;
  search.addEventListener("input", () => {
    clearTimeout(handle);
    handle = setTimeout(load, 200);
  });

  await load();
}

function card(n) {
  const iconCls = n.kind.toLowerCase();
  return el("div", {
    class: "session-row",
    onclick: () => (location.hash = `#/nodes/${n.id}`),
  }, [
    el("div", { class: "top" }, [
      el("span", { class: `chip ${iconCls} ${n.is_unknown ? "unknown" : ""}` }, [n.label]),
      el("span", { class: "muted", style: "font-size:11px;" }, [`${n.mention_count || 0} bahsedilme`]),
    ]),
    n.time_value
      ? el("div", { class: "muted", style: "font-size:12px;" }, [`Tarih: ${n.time_value}`])
      : null,
    n.notes
      ? el("div", { class: "excerpt" }, [n.notes])
      : null,
  ]);
}

export async function renderDetail(root, ctx) {
  root.innerHTML = "";
  const id = ctx.params.id;
  const [node, edges, mentions] = await Promise.all([
    fetch(`/api/nodes/${id}/`).then((r) => r.json()),
    fetch(`/api/edges/`).then((r) => r.json()),
    fetch(`/api/mentions/`).then((r) => r.json()),
  ]);

  const iconCls = node.kind.toLowerCase();
  root.appendChild(
    el("div", { class: "card node-detail" }, [
      el("span", { class: `chip ${iconCls} ${node.is_unknown ? "unknown" : ""}`, style: "margin-bottom:6px;" }, [node.kind_display || node.kind]),
      el("h2", {}, [node.label]),
      node.time_value ? el("div", { class: "muted" }, [`Tarih: ${node.time_value}`]) : null,
      node.notes ? el("p", {}, [node.notes]) : null,
    ])
  );

  const edgesArray = Array.isArray(edges) ? edges : edges.results || [];
  const myEdges = edgesArray.filter((e) => e.source === node.id || e.target === node.id);
  if (myEdges.length > 0) {
    const list = el("ul", { class: "edges-list" });
    for (const e of myEdges) {
      const other = e.source === node.id
        ? { label: e.target_label, kind: e.target_kind }
        : { label: e.source_label, kind: e.source_kind };
      list.appendChild(
        el("li", {}, [
          el("span", { class: `chip ${(other.kind || "").toLowerCase()}` }, [other.label]),
          " — ",
          el("span", { class: "muted" }, [e.relation_display || e.relation]),
        ])
      );
    }
    root.appendChild(el("div", { class: "card" }, [
      el("h2", { style: "font-size:14px;" }, ["Bağlantılar"]),
      list,
    ]));
  }

  const mentionsArray = Array.isArray(mentions) ? mentions : mentions.results || [];
  const mine = mentionsArray.filter((m) => m.node && m.node.id === node.id);
  if (mine.length > 0) {
    const list = el("div", { class: "mentions-list" });
    for (const m of mine) {
      list.appendChild(
        el("div", { class: "mention-card", onclick: () => (location.hash = `#/review/${m.session}`) }, [
          el("span", { class: "muted", style: "font-size:12px;" }, [
            `${MENTION_LABELS[m.mention_type] || m.mention_type}`,
          ]),
          el("div", {}, [`"${m.surface}"`]),
        ])
      );
    }
    root.appendChild(el("div", { class: "card" }, [
      el("h2", { style: "font-size:14px;" }, ["Bahsedildiği yerler"]),
      list,
    ]));
  }
}

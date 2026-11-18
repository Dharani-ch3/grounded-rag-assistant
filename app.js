// Grounded RAG Assistant web console. Plain JS, no build step.
// All user, document and model text is inserted with textContent, never innerHTML.

const $ = (id) => document.getElementById(id);
const state = { users: [], user: "ana", view: "ask", busy: false, roles: [], docs: [] };
const COLORS = { ana: "#0f766e", ben: "#2f4bd8", carla: "#9333ea", dev: "#b45309", omar: "#be123c" };
const POLICY_FLAG = { draft: "unapproved draft", archived: "outdated version", no_access: "no access" };
const ROUTE_LABEL = { documents: "document search", graph: "knowledge graph", both: "search + graph" };
const OUTCOME_LABEL = {
  correct: "correct", correct_abstain: "correct: I don't know", leak: "leak", draft: "used draft",
  stale: "outdated", unsupported: "made up", wrong: "wrong", over_abstain: "needless I don't know",
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

// Static demo (GitHub Pages): requests go to the Python assistant running in the browser via Pyodide.
async function api(path, opts = {}) {
  if (window.GRA_STATIC) {
    await window.graReady;
    // Yield so the "Working…" state paints before Python runs on this thread.
    await new Promise((r) => setTimeout(r, 30));
    const { status, data } = window.graBridge(opts.method || "GET", path, opts.json);
    if (status >= 400) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    return data;
  }
  const headers = {};
  if (opts.json !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method: opts.method || "GET", headers, body: opts.json !== undefined ? JSON.stringify(opts.json) : undefined });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : res.statusText);
  return data;
}

function toast(msg) {
  const t = el("div", { class: "toast", role: "status" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 4000);
}

async function busy(btn, fn) {
  if (state.busy) return;
  state.busy = true;
  const label = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "Working…"; }
  try { await fn(); } catch (e) { toast(e.message); console.error(e); } finally {
    state.busy = false;
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

const me = () => state.users.find((u) => u.id === state.user) || { name: state.user, id: state.user };
const firstName = (u) => (u.name || "").split(" ")[0];

// ---------------- people ----------------

function renderUsers() {
  $("users").replaceChildren(...state.users.map((u) => el("button", {
    type: "button", role: "radio", "aria-checked": String(u.id === state.user),
    class: "user" + (u.id === state.user ? " active" : ""),
    onclick: () => selectUser(u.id),
  },
    el("span", { class: "avatar", style: `background:${COLORS[u.id] || "#444"}` }, u.name.split(" ").map((p) => p[0]).join("")),
    el("span", {},
      el("div", { class: "name" }, u.name, u.admin ? el("span", { class: "admin", title: "Knowledge admin: can approve drafts" }, "ADMIN") : null),
      el("div", { class: "meta" }, `${u.title} · ${u.org}`),
    ),
  )));
  const u = me();
  $("suggestions").replaceChildren(...(u.suggestions || []).map((s) => el("button", {
    type: "button", class: "suggestion", onclick: () => { $("question").value = s.q; ask(); },
  }, el("small", {}, s.label), s.q)));
  const welcome = $("welcome");
  if (welcome) welcome.querySelector("strong").textContent = `Ask a policy question as ${u.name}, ${u.title.toLowerCase()}.`;
}

function selectUser(id) {
  state.user = id;
  renderUsers();
  $("thread").append(el("div", { class: "hint", style: "text-align:center" }, `Now asking as ${me().name} (${me().title})`));
  if (state.view === "knowledge") loadKnowledge();
}

// ---------------- asking ----------------

function citeText(text, card) {
  const parts = text.split(/\s*\[(\d+)\]/);
  const out = [];
  parts.forEach((part, i) => {
    if (i % 2 === 0) { if (part) out.push(document.createTextNode(part)); return; }
    out.push(el("button", {
      type: "button", class: "cite", title: `Source ${part}`,
      onclick: () => {
        const src = card.querySelector(`[data-n="${part}"]`);
        if (!src) return;
        src.classList.add("hl");
        src.scrollIntoView({ block: "nearest", behavior: "smooth" });
        setTimeout(() => src.classList.remove("hl"), 1600);
      },
    }, part));
    out.push(document.createTextNode(" "));
  });
  return out;
}

function renderTrace(r) {
  const t = r.trace;
  const body = el("div", { class: "trace-body" });
  body.append(el("p", {}, el("b", {}, "Route: "), `${ROUTE_LABEL[r.route.kind]}. ${r.route.reason}.`));
  if (r.mode === "grounded" && Object.keys(t.excluded || {}).length) {
    const parts = Object.entries(t.excluded).map(([k, n]) => `${n} ${k === "draft" ? "draft" : "outdated"} passage${n > 1 ? "s" : ""}`);
    body.append(el("p", {}, el("b", {}, "Filtered out before search: "), parts.join(", "),
      el("span", { class: "muted" }, ". Passages this person can't read aren't counted, so their existence isn't revealed.")));
  }
  if (t.retrieved.length) {
    body.append(el("h4", {}, "Retrieved passages (hybrid search)"));
    body.append(el("table", { class: "mini" },
      el("thead", {}, el("tr", {}, ["#", "Document · section", "Fused", "BM25", "Vector", "Policy"].map((h) => el("th", {}, h)))),
      el("tbody", {}, t.retrieved.map((h) => el("tr", {},
        el("td", { class: "num" }, h.n),
        el("td", {}, `${h.title} v${h.version}`, el("span", { class: "muted" }, h.heading ? ` › ${h.heading}` : "")),
        el("td", { class: "num" }, h.score.toFixed(2)),
        el("td", { class: "num" }, h.bm25_rank ?? "–"),
        el("td", { class: "num" }, h.vector_rank ?? "–"),
        el("td", {}, h.policy === "ok" ? "ok" : el("span", { class: "flag" }, POLICY_FLAG[h.policy])),
      ))),
    ));
  }
  if (t.confidence !== null && t.confidence !== undefined) {
    body.append(el("p", {}, el("b", {}, "Question coverage: "), `${Math.round(t.confidence * 100)}%`,
      el("span", { class: "muted" }, r.mode === "grounded" ? " (below 60% the assistant abstains)" : " (naive mode answers regardless)")));
  }
  if (t.verification && t.verification.length) {
    body.append(el("h4", {}, r.mode === "grounded" ? "Grounding check, sentence by sentence" : "Grounding check (not enforced in naive mode)"));
    for (const v of t.verification) {
      body.append(el("div", { class: "verdict" },
        el("span", { class: "badge " + (v.kept ? "grounded" : "naive") }, v.kept ? "kept" : v.reason.replace("_", " ")),
        el("span", {}, v.sentence, el("span", { class: "muted" }, ` (${Math.round(v.support * 100)}% of words found in the cited passage)`)),
      ));
    }
  }
  if (t.abstain_reason) body.append(el("p", {}, el("b", {}, "Why it abstained: "), t.abstain_reason));
  if (r.graph && r.graph.lookup) {
    body.append(el("h4", {}, "Graph lookup (fixed, only the parameters change)"));
    body.append(el("p", {}, r.graph.lookup));
    body.append(el("pre", { class: "code" }, "params: " + JSON.stringify(r.graph.params)));
  }
  body.append(el("p", { class: "muted" }, `${r.latency_ms} ms · provider: ${r.provider}`));
  return el("details", { class: "trace" }, el("summary", {}, "How this answer was built"), body);
}

function renderAnswer(r, { compare = true } = {}) {
  const card = el("article", { class: "answer" + (r.mode === "naive" ? " naive" : "") + (r.abstained ? " abstained" : "") });
  card.append(el("div", { class: "answer-head" },
    el("span", { class: "badge " + r.mode }, r.mode === "grounded" ? "guardrails on" : "naive RAG"),
    el("span", { class: "badge" + (r.route.kind !== "documents" ? " graph" : "") }, ROUTE_LABEL[r.route.kind]),
    r.abstained ? el("span", { class: "badge abstain" }, "abstained") : null,
    el("span", {}, `as ${r.user.name}`),
  ));
  card.append(el("div", { class: "answer-text" }, citeText(r.answer, card)));

  if (r.graph && r.graph.path && r.graph.path.length && !r.abstained) {
    card.append(el("div", { class: "path" }, r.graph.path.map((e) =>
      el("span", { class: "edge" }, e.source, el("i", {}, `—${e.rel}→`), e.target))));
  }
  if (r.citations.length) {
    card.append(el("div", { class: "sources" }, r.citations.map((c) => {
      const flagged = c.policy !== "ok";
      return el("div", { class: "source" + (flagged ? " flagged" : ""), "data-n": c.n },
        el("span", { class: "n" }, `[${c.n}]`),
        el("span", {}, el("span", { class: "title" }, c.title), ` v${c.version}`,
          c.kind === "graph" ? el("span", { class: "muted" }, " · graph") : null,
          flagged ? el("span", { class: "flag" }, `⚠ ${POLICY_FLAG[c.policy]}${c.policy === "no_access" ? ` for ${firstName(r.user)}` : ""}`) : null),
        el("span", { class: "sub" }, [c.heading ? `${c.heading} · ` : "", `${c.status} · owner ${c.owner ? c.owner.name : "?"}`].join("")),
        c.text ? el("span", { class: "quote" }, c.text) : null,
      );
    })));
  }
  const actions = el("div", { class: "answer-actions" }, renderTrace(r));
  if (compare && r.mode === "grounded") {
    const btn = el("button", { type: "button", class: "btn outline tiny" }, "Compare with naive RAG");
    btn.addEventListener("click", () => busy(btn, async () => {
      const naive = await api("/api/ask", { method: "POST", json: { user: r.user.id, question: r.question, mode: "naive" } });
      const wrap = el("div", { class: "compare" },
        el("div", {}, el("p", { class: "compare-title" }, "Guardrails on"), renderAnswer(r, { compare: false })),
        el("div", {}, el("p", { class: "compare-title" }, "Naive RAG (no filters, always answers)"), renderAnswer(naive, { compare: false })),
      );
      card.replaceWith(wrap);
    }));
    actions.prepend(btn);
  }
  card.append(actions);
  return card;
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) return;
  const mode = $("guardrails").checked ? "grounded" : "naive";
  await busy($("askBtn"), async () => {
    $("welcome")?.remove();
    const r = await api("/api/ask", { method: "POST", json: { user: state.user, question, mode } });
    $("thread").append(el("div", { class: "q-bubble" }, el("small", {}, r.user.name), question), renderAnswer(r));
    $("question").value = "";
    $("thread").lastElementChild.scrollIntoView({ block: "start", behavior: "smooth" });
  });
}

// ---------------- knowledge ----------------

async function loadKnowledge() {
  const [docs, graph] = await Promise.all([
    api(`/api/documents?user=${encodeURIComponent(state.user)}`),
    api(`/api/graph?user=${encodeURIComponent(state.user)}`),
  ]);
  state.docs = docs;
  state.roles = graph.nodes.filter((n) => n.kind === "Role");
  $("kbUser").textContent = me().name;
  $("docCount").textContent = docs.length;
  $("docRows").replaceChildren(...docs.map((d) => {
    const canApprove = me().admin && d.status === "draft" && d.owner && d.owner.id !== state.user;
    let approve = null;
    if (canApprove) {
      approve = el("button", { type: "button", class: "btn tiny" }, "Approve");
      approve.addEventListener("click", () => busy(approve, async () => {
        await api(`/api/documents/${d.id}/approve`, { method: "POST", json: { user: state.user } });
        toast(`Approved. "${d.title}" can now be used in answers${d.supersedes ? "; the version it replaces is archived" : ""}.`);
        await loadKnowledge();
      }));
    } else if (me().admin && d.status === "draft") {
      approve = el("span", { class: "muted", style: "font-size:12px" }, "Author can't self-approve");
    }
    return el("tr", { class: d.status === "approved" && !d.superseded_by ? "" : "dim" },
      el("td", {}, el("span", { class: "doc-title", onclick: () => openDoc(d.id) }, d.title), el("span", { class: "muted" }, ` v${d.version}`)),
      el("td", {}, el("span", { class: "status " + d.status }, d.status)),
      el("td", {}, d.owner ? d.owner.name : "", el("div", { class: "muted", style: "font-size:12px" }, d.org_name)),
      el("td", { style: "font-size:12.5px" }, d.audience_names.join(", ")),
      el("td", {}, approve),
    );
  }));
  $("docSupersedes").replaceChildren(el("option", { value: "" }, "Nothing, it's a new document"),
    ...docs.filter((d) => d.status === "approved").map((d) => el("option", { value: d.id }, `${d.title} v${d.version}`)));
  $("docAudience").replaceChildren(...state.roles.map((r) => {
    const id = r.id.replace("role:", "");
    return el("label", {}, el("input", { type: "checkbox", value: id, checked: id === "employee" }), r.label);
  }));
  renderGraph(graph);
}

async function openDoc(id) {
  const d = await api(`/api/documents/${id}?user=${encodeURIComponent(state.user)}`);
  $("dlgTitle").textContent = `${d.title} · v${d.version} · ${d.status}`;
  $("dlgBody").replaceChildren(
    el("p", { class: "muted" }, `Owner ${d.owner?.name || "?"} · ${d.approved_by ? "approved by " + d.approved_by.name : "not approved"} · readable by ${d.audience_names.join(", ")} · effective ${d.effective}`),
    ...d.chunks.map((c) => el("div", { class: "chunk" }, el("b", {}, `${c.id}${c.heading ? " · " + c.heading : ""}`), c.text)),
  );
  $("docDialog").showModal();
}

function renderGraph(g) {
  const COLS = [["Role", "Roles"], ["User", "People"], ["Document", "Documents"], ["Org", "Departments"]];
  const W = 1064, pad = 12, colW = 200, rowH = 46, top = 44, boxH = 34;
  const gap = (W - 2 * pad - colW * 4) / 3;
  const pos = {};
  let maxRows = 0;
  COLS.forEach(([kind], ci) => {
    const nodes = g.nodes.filter((n) => n.kind === kind);
    maxRows = Math.max(maxRows, nodes.length);
    nodes.forEach((n, i) => { pos[n.id] = { x: pad + ci * (colW + gap), y: top + i * rowH, col: ci, node: n }; });
  });
  const H = top + maxRows * rowH + 10;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Knowledge graph" });
  COLS.forEach(([, title], ci) => {
    const t = svgEl("text", { x: pad + ci * (colW + gap), y: 24, class: "col-title" });
    t.textContent = title;
    svg.append(t);
  });
  const edgesG = svgEl("g"), nodesG = svgEl("g");
  svg.append(edgesG, nodesG);
  const lines = [];
  for (const e of g.edges) {
    const a = pos[e.source], b = pos[e.target];
    if (!a || !b) continue;
    let d;
    const ay = a.y + boxH / 2, by = b.y + boxH / 2;
    if (a.col === b.col) {
      const x = a.x + colW, bulge = 30 + Math.abs(by - ay) * 0.15;
      d = `M${x},${ay} C${x + bulge},${ay} ${x + bulge},${by} ${x},${by}`;
    } else {
      const [l, r] = a.col < b.col ? [a, b] : [b, a];
      const x1 = l.x + colW, y1 = l.y + boxH / 2, x2 = r.x, y2 = r.y + boxH / 2, mid = (x2 - x1) / 2;
      d = `M${x1},${y1} C${x1 + mid},${y1} ${x2 - mid},${y2} ${x2},${y2}`;
    }
    const p = svgEl("path", { d, class: "edge-line" });
    const title = svgEl("title");
    title.textContent = `${a.node.label} —${e.rel}→ ${b.node.label}`;
    p.append(title);
    edgesG.append(p);
    lines.push({ el: p, source: e.source, target: e.target });
  }
  const nodeEls = {};
  for (const [id, p] of Object.entries(pos)) {
    const n = svgEl("g", { class: "node", transform: `translate(${p.x},${p.y})`, tabindex: "0" });
    n.append(svgEl("rect", { width: colW, height: boxH, rx: 8 }));
    const label = svgEl("text", { x: 10, y: p.node.sub ? 15 : 21 });
    label.textContent = p.node.label.length > 28 ? p.node.label.slice(0, 27) + "…" : p.node.label;
    n.append(label);
    if (p.node.sub) {
      const sub = svgEl("text", { x: 10, y: 28, class: "sub" });
      sub.textContent = p.node.sub;
      n.append(sub);
    }
    const t = svgEl("title");
    t.textContent = p.node.label + (p.node.sub ? ` (${p.node.sub})` : "");
    n.append(t);
    const focus = () => {
      $("graph").classList.add("focus");
      n.classList.add("on");
      for (const l of lines) {
        if (l.source === id || l.target === id) {
          l.el.classList.add("on");
          nodeEls[l.source === id ? l.target : l.source]?.classList.add("on");
        }
      }
    };
    const blur = () => {
      $("graph").classList.remove("focus");
      svg.querySelectorAll(".on").forEach((x) => x.classList.remove("on"));
    };
    n.addEventListener("mouseenter", focus);
    n.addEventListener("mouseleave", blur);
    n.addEventListener("focus", focus);
    n.addEventListener("blur", blur);
    nodeEls[id] = n;
    nodesG.append(n);
  }
  $("graph").replaceChildren(svg);
}

async function saveDoc(e) {
  e.preventDefault();
  const audience = [...$("docAudience").querySelectorAll("input:checked")].map((i) => i.value);
  await busy(e.submitter, async () => {
    const d = await api("/api/documents", { method: "POST", json: {
      user: state.user, title: $("docTitle").value, text: $("docText").value, audience, supersedes: $("docSupersedes").value || null,
    } });
    toast(`Saved "${d.title}" as a draft. Another knowledge admin must approve it before it's used.`);
    $("docForm").reset();
    $("docForm").hidden = true;
    await loadKnowledge();
  });
}

// ---------------- evaluation ----------------

function pct(x) { return `${Math.round(x * 100)}%`; }

async function runEval(btn, holdout) {
  await busy(btn, async () => {
    const r = await api("/api/eval", { method: "POST", json: { holdout } });
    const n = r.naive, g = r.grounded;
    const out = $("evalOut");
    out.replaceChildren(
      el("p", { class: "muted" }, holdout
        ? "Held-out set: questions written after tuning and never used to tune the engine."
        : "Tuning set: written alongside the engine, so the guardrails score here is optimistic. Run the held-out set for a fairer number."),
      el("div", { class: "scores" },
        el("div", { class: "score bad" }, el("span", { class: "label" }, "Naive RAG accuracy"), el("span", { class: "big" }, pct(n.accuracy)), el("span", { class: "label" }, `${n.grounding_errors} of ${n.cases} answers not grounded in current, permitted knowledge`)),
        el("div", { class: "score good" }, el("span", { class: "label" }, "With guardrails"), el("span", { class: "big" }, pct(g.accuracy)), el("span", { class: "label" }, `${g.grounding_errors} grounding errors · ${g.counts.over_abstain} needless "I don't know"`)),
        el("div", { class: "score" }, el("span", { class: "label" }, "Grounding errors removed"), el("span", { class: "big" }, pct(r.grounding_error_reduction)), el("span", { class: "label" }, "leaks + drafts + outdated versions + made-up answers")),
      ),
      el("table", { class: "mini" },
        el("thead", {}, el("tr", {}, el("th", {}, "Outcome"), el("th", {}, "Naive"), el("th", {}, "Guardrails"))),
        el("tbody", {}, Object.keys(OUTCOME_LABEL).map((k) => el("tr", {},
          el("td", {}, el("span", { class: "outcome " + k }, OUTCOME_LABEL[k])),
          el("td", { class: "num" }, n.counts[k]), el("td", { class: "num" }, g.counts[k])))),
      ),
      el("h3", {}, "Per question ", el("span", { class: "muted", style: "font-weight:400;font-size:12.5px" }, "(click a row to see both answers)")),
      el("table", { class: "mini eval-table" },
        el("thead", {}, el("tr", {}, ["#", "Asked by", "Question", "Naive", "Guardrails"].map((h) => el("th", {}, h)))),
        el("tbody", {}, r.rows.flatMap((row) => {
          const detail = el("tr", { class: "eval-detail", hidden: true }, el("td", { colspan: 5 },
            el("p", {}, el("b", {}, "Naive: "), row.naive.answer, row.naive.cited.length ? el("span", { class: "muted" }, ` [${row.naive.cited.join(", ")}]`) : null),
            el("p", { style: "margin-top:6px" }, el("b", {}, "Guardrails: "), row.grounded.answer, row.grounded.cited.length ? el("span", { class: "muted" }, ` [${row.grounded.cited.join(", ")}]`) : null),
          ));
          const user = state.users.find((u) => u.id === row.user);
          return [el("tr", { onclick: () => { detail.hidden = !detail.hidden; } },
            el("td", { class: "num" }, row.id),
            el("td", {}, user ? firstName(user) : row.user),
            el("td", {}, row.question),
            el("td", {}, el("span", { class: "outcome " + row.naive.outcome }, OUTCOME_LABEL[row.naive.outcome])),
            el("td", {}, el("span", { class: "outcome " + row.grounded.outcome }, OUTCOME_LABEL[row.grounded.outcome])),
          ), detail];
        })),
      ),
    );
  });
}

// ---------------- shell ----------------

function show(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  for (const v of ["ask", "knowledge", "eval"]) $(`view-${v}`).hidden = v !== view;
  $("suggestBox").hidden = view !== "ask";
  // The evaluation asks as several fixed people, so the "Asking as" panel doesn't apply there.
  document.querySelector(".side").hidden = view === "eval";
  document.querySelector(".layout").classList.toggle("wide", view === "eval");
  if (view === "knowledge") loadKnowledge().catch((e) => toast(e.message));
}

async function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.view)));
  $("askForm").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  $("guardrails").addEventListener("change", () => {
    $("guardLabel").textContent = $("guardrails").checked ? "Guardrails on" : "Guardrails off (naive RAG)";
  });
  $("newDocBtn").addEventListener("click", () => { $("docForm").hidden = !$("docForm").hidden; });
  $("docCancel").addEventListener("click", () => { $("docForm").hidden = true; });
  $("docForm").addEventListener("submit", saveDoc);
  $("evalBtn").addEventListener("click", (e) => runEval(e.currentTarget, false));
  $("holdoutBtn").addEventListener("click", (e) => runEval(e.currentTarget, true));
  if (window.GRA_STATIC) $("docsLink").hidden = true;
  try {
    const health = await api("/api/health");
    $("health").textContent = health.runtime === "browser" ? "running in your browser" : `${health.provider} · ${health.vector_store}`;
    state.users = await api("/api/users");
    renderUsers();
  } catch (e) {
    $("health").textContent = "offline";
    toast(e.message);
  }
}

init();

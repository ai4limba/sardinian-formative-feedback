"use strict";

const el = (id) => document.getElementById(id);
const textarea = el("text");
const backdrop = el("backdrop");
const popover = el("popover");

// Categories shown in the summary, in this order
const CATEGORY_ORDER = ["accent", "graphematic", "typing", "contextual", "unverified"];

let config = null;

const state = {
  text: "",
  raw: [],          // issues as returned by the server
  issues: [],       // issues currently shown (not kept by the writer, not being typed)
  words: 0,
  threshold: null,
  lastEvent: "",
  requestId: 0,
  timer: null,
  openIssue: null,
  kept: new Set(),  // words the writer decided to keep
  session: null,
};

// ==========================================
// HELPERS
// ==========================================

function t(key, values) {
  let text = config.strings[key] ?? key;
  for (const [name, value] of Object.entries(values ?? {})) text = text.replaceAll(`{${name}}`, value);
  return text;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function newSession() {
  return { started_at: new Date().toISOString(), events: [], seen: new Map(), accepted: 0, kept: 0 };
}

function logEvent(type, issue, extra) {
  state.session.events.push({
    time: new Date().toISOString(),
    type,
    ...(issue ? { word: issue.word, category: issue.category, pattern: issue.pattern } : {}),
    ...extra,
  });
}

function setStatus(name) {
  el("status").textContent = t(`status_${name}`);
}

// ==========================================
// ISSUES
// ==========================================

// Keeps the highlights aligned while the writer types, until the next analysis arrives:
// issues before the edit stay, issues after it move, issues touched by it are dropped
function shiftIssues(oldText, newText) {
  const maxCommon = Math.min(oldText.length, newText.length);
  let prefix = 0;
  while (prefix < maxCommon && oldText[prefix] === newText[prefix]) prefix++;
  let suffix = 0;
  while (suffix < maxCommon - prefix && oldText[oldText.length - 1 - suffix] === newText[newText.length - 1 - suffix]) suffix++;

  const oldEnd = oldText.length - suffix;
  const delta = newText.length - oldText.length;
  state.raw = state.raw
    .filter((issue) => issue.end < prefix || issue.start > oldEnd)
    .map((issue) => (issue.start > oldEnd ? { ...issue, start: issue.start + delta, end: issue.end + delta } : issue));
}

// Decides which issues are shown, and records the new ones in the session
function deriveIssues() {
  const typing = state.lastEvent === "input" && document.activeElement === textarea;
  const caret = textarea.selectionStart;

  state.issues = state.raw.filter((issue) => {
    if (state.kept.has(issue.word.toLowerCase())) return false;
    // The word under the caret is still being typed: wait before commenting on it
    if (typing && issue.start < caret && caret <= issue.end) return false;
    return true;
  });

  for (const issue of state.issues) {
    const suggestion = issue.suggestions[0]?.word ?? "";
    const key = `${issue.word.toLowerCase()}→${suggestion.toLowerCase()}`;
    if (!state.session.seen.has(key)) {
      state.session.seen.set(key, { word: issue.word, suggestion, category: issue.category, pattern: issue.pattern });
      logEvent("flagged", issue, { suggestion });
    }
  }
}

function scheduleAnalysis(delay) {
  clearTimeout(state.timer);
  state.timer = setTimeout(analyze, delay);
}

async function analyze() {
  const text = state.text;
  const requestId = ++state.requestId;

  if (!text.trim()) {
    state.raw = [];
    state.words = 0;
    deriveIssues();
    render();
    setStatus("idle");
    return;
  }

  setStatus("working");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, threshold: state.threshold }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    // The text has changed in the meantime: this answer is out of date
    if (requestId !== state.requestId || text !== state.text) return;

    state.raw = data.issues;
    state.words = data.words;
    deriveIssues();
    render();
    setStatus("idle");
  } catch (error) {
    console.error(error);
    if (requestId === state.requestId) setStatus("error");
  }
}

// ==========================================
// RENDERING
// ==========================================

function render() {
  renderBackdrop();
  renderSummary();
  if (state.openIssue && !state.issues.includes(state.openIssue)) closePopover();
}

function renderBackdrop() {
  backdrop.replaceChildren();
  let position = 0;

  state.issues.forEach((issue, index) => {
    if (issue.start < position) return;
    backdrop.append(state.text.slice(position, issue.start));
    const mark = node("mark", `cat-${issue.category}`, state.text.slice(issue.start, issue.end));
    mark.dataset.index = index;
    backdrop.append(mark);
    position = issue.end;
  });

  // The final line break keeps the two layers the same height
  backdrop.append(state.text.slice(position) + "\n");
  backdrop.scrollTop = textarea.scrollTop;
}

function countBy(field) {
  const counts = new Map();
  for (const entry of state.session.seen.values()) counts.set(entry[field], (counts.get(entry[field]) ?? 0) + 1);
  return counts;
}

function recurrentPatterns(minimum) {
  return [...countBy("pattern").entries()]
    .filter(([, count]) => count >= minimum)
    .sort((a, b) => b[1] - a[1])
    .map(([pattern, count]) => ({
      pattern,
      label: config.patterns[pattern] ?? pattern,
      count,
      examples: [...state.session.seen.values()].filter((entry) => entry.pattern === pattern),
    }));
}

function patternSentence(entry) {
  const key = entry.count === 1 ? "summary_pattern_item_one" : "summary_pattern_item";
  return t(key, { count: entry.count, pattern: entry.label });
}

function renderCounters(values) {
  el("count-words").textContent = values.words;
  el("count-flagged").textContent = values.flagged;
  el("count-accepted").textContent = values.accepted;
  el("count-kept").textContent = values.kept;
}

function renderSummary() {
  renderCounters({ words: state.words, flagged: state.session.seen.size, accepted: state.session.accepted, kept: state.session.kept });

  // Flagged words by category
  const byCategory = countBy("category");
  const highest = Math.max(1, ...byCategory.values());
  const bars = el("by-category");
  bars.replaceChildren();
  for (const category of CATEGORY_ORDER) {
    const count = byCategory.get(category) ?? 0;
    const item = node("li", `cat-${category}`);
    item.append(node("span", "chip", config.categories[category].label), node("span", "count", count));
    const track = node("span", "track");
    const fill = node("span", "fill");
    fill.style.width = `${(count / highest) * 100}%`;
    track.append(fill);
    item.append(track);
    bars.append(item);
  }

  // Recurrent difficulties
  const patterns = el("patterns");
  patterns.replaceChildren();
  const recurrent = recurrentPatterns(2).slice(0, 4);
  if (recurrent.length === 0) patterns.append(node("li", "empty", t("summary_patterns_none")));
  for (const entry of recurrent) patterns.append(node("li", "", patternSentence(entry)));

  // Issues still open
  const open = el("open-issues");
  open.replaceChildren();
  if (state.issues.length === 0) open.append(node("li", "empty", t("summary_open_none")));
  for (const issue of state.issues) {
    const button = node("button", `cat-${issue.category}`);
    button.type = "button";
    button.append(node("strong", "", issue.word));
    if (issue.suggestions.length > 0) {
      button.append(node("span", "arrow", " → "), issue.suggestions[0].word);
    } else {
      button.append(node("span", "arrow", ` · ${config.categories.unverified.label}`));
    }
    button.addEventListener("click", () => selectIssue(issue));
    const item = node("li");
    item.append(button);
    open.append(item);
  }
}

// ==========================================
// FEEDBACK CARD
// ==========================================

function openPopover(issue) {
  const mark = backdrop.querySelector(`mark[data-index="${state.issues.indexOf(issue)}"]`);
  if (!mark) return;
  state.openIssue = issue;

  popover.replaceChildren();
  popover.className = `popover cat-${issue.category}`;
  popover.append(node("span", "chip", config.categories[issue.category].label));

  const written = node("p", "written", `${t("popover_written")} `);
  written.append(node("strong", "", issue.word));
  popover.append(written, node("p", "explanation", issue.explanation));

  if (issue.suggestions.length > 0) {
    popover.append(node("h4", "", t("popover_suggestions")));
    issue.suggestions.forEach((suggestion, rank) => {
      const row = node("div", "suggestion");
      // The first suggestion is explained above: here, how well it fits the sentence
      const note = rank === 0 ? config.context_fit[suggestion.fit] : suggestion.explanation;
      const use = node("button", "button", t("popover_use"));
      use.type = "button";
      use.addEventListener("click", () => acceptSuggestion(issue, suggestion, rank + 1));
      row.append(node("span", "word", suggestion.word), use, node("span", "note", note));
      popover.append(row);
    });
  }

  const keep = node("div", "keep");
  const keepButton = node("button", "button", t("popover_keep"));
  keepButton.type = "button";
  keepButton.addEventListener("click", () => keepWord(issue));
  keep.append(keepButton, node("p", "hint", t("popover_keep_hint")));
  popover.append(keep);

  // Just below the word, without leaving the page
  popover.hidden = false;
  const markRect = mark.getBoundingClientRect();
  const editorRect = textarea.getBoundingClientRect();
  const maxLeft = window.scrollX + document.documentElement.clientWidth - popover.offsetWidth - 8;
  popover.style.top = `${window.scrollY + Math.min(markRect.bottom, editorRect.bottom) + 6}px`;
  popover.style.left = `${Math.max(8, Math.min(window.scrollX + markRect.left, maxLeft))}px`;
}

function closePopover() {
  state.openIssue = null;
  popover.hidden = true;
}

function selectIssue(issue) {
  textarea.focus();
  textarea.setSelectionRange(issue.end, issue.end);
  state.lastEvent = "click";
  openPopover(issue);
}

function applyTextChange(eventName) {
  const newText = textarea.value;
  shiftIssues(state.text, newText);
  state.text = newText;
  state.lastEvent = eventName;
  closePopover();
  deriveIssues();
  render();
}

function acceptSuggestion(issue, suggestion, rank) {
  state.session.accepted++;
  logEvent("accepted", issue, { suggestion: suggestion.word, suggestion_rank: rank, suggestion_category: suggestion.category });

  textarea.focus();
  textarea.setRangeText(suggestion.word, issue.start, issue.end, "end");
  applyTextChange("accept");
  analyze();
}

function keepWord(issue) {
  state.kept.add(issue.word.toLowerCase());
  state.session.kept++;
  logEvent("kept", issue);
  closePopover();
  deriveIssues();
  render();
  textarea.focus();
}

// ==========================================
// FINAL SUMMARY AND SESSION RECORD
// ==========================================

function buildSummary() {
  return {
    words: state.words,
    flagged: state.session.seen.size,
    accepted: state.session.accepted,
    kept: state.session.kept,
    open: state.issues.length,
    by_category: Object.fromEntries(countBy("category")),
    patterns: recurrentPatterns(1),
  };
}

async function finishRevision() {
  // Analyse once more, this time including the last word typed
  clearTimeout(state.timer);
  state.lastEvent = "finish";
  closePopover();
  await analyze();

  const summary = buildSummary();
  const body = el("final-body");
  body.replaceChildren();

  if (summary.flagged === 0) {
    body.append(node("p", "", t("final_nothing")));
  } else {
    body.append(node("p", "", t("final_intro")));

    const counters = node("dl", "counters");
    for (const [label, value] of [["summary_words", summary.words], ["summary_flagged", summary.flagged],
                                  ["summary_accepted", summary.accepted], ["summary_kept", summary.kept]]) {
      const cell = node("div");
      cell.append(node("dt", "", t(label)), node("dd", "", value));
      counters.append(cell);
    }
    body.append(counters);

    for (const entry of summary.patterns) {
      const block = node("div", "pattern");
      const examples = entry.examples.map((example) => (example.suggestion ? `${example.word} → ${example.suggestion}` : example.word));
      block.append(
        node("p", "", patternSentence(entry)),
        node("p", "examples", `${t("final_examples")}: ${examples.join(", ")}`),
      );
      body.append(block);
    }
  }

  const saved = el("final-saved");
  saved.textContent = "";
  el("final").showModal();

  try {
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        started_at: state.session.started_at,
        finished_at: new Date().toISOString(),
        detection_threshold: state.threshold,
        final_text: state.text,
        summary,
        events: state.session.events,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    saved.textContent = `${t("final_saved")} ${(await response.json()).saved_to}`;
  } catch (error) {
    console.error(error);
    saved.textContent = t("final_not_saved");
  }
}

function startNewSession() {
  textarea.value = "";
  state.text = "";
  state.raw = [];
  state.words = 0;
  state.kept = new Set();
  state.session = newSession();
  deriveIssues();
  render();
  setStatus("idle");
}

function replaceText(text) {
  textarea.value = text;
  applyTextChange("replace");
  analyze();
}

// ==========================================
// EVENTS AND START-UP
// ==========================================

textarea.addEventListener("input", (event) => {
  applyTextChange("input");
  // The end of a word triggers a quick analysis, anything else waits for a pause
  const endOfWord = event.data && /[\s.,;:!?]/.test(event.data);
  scheduleAnalysis(endOfWord ? 80 : 400);
});

textarea.addEventListener("click", () => {
  state.lastEvent = "click";
  deriveIssues();
  render();
  const caret = textarea.selectionStart;
  const issue = textarea.selectionStart === textarea.selectionEnd
    ? state.issues.find((candidate) => candidate.start <= caret && caret <= candidate.end)
    : undefined;
  if (issue) openPopover(issue); else closePopover();
});

textarea.addEventListener("keyup", (event) => {
  if (event.key.startsWith("Arrow") || event.key === "Home" || event.key === "End") {
    state.lastEvent = "move";
    deriveIssues();
    render();
  }
});

textarea.addEventListener("blur", () => {
  // Redraw only if something changed: redrawing the panel would swallow a click on it
  const shown = state.issues.length;
  deriveIssues();
  if (state.issues.length !== shown) render();
});
textarea.addEventListener("scroll", () => { backdrop.scrollTop = textarea.scrollTop; closePopover(); });

document.addEventListener("keydown", (event) => { if (event.key === "Escape") closePopover(); });
document.addEventListener("mousedown", (event) => {
  if (!popover.hidden && !popover.contains(event.target) && event.target !== textarea && !event.target.closest(".issue-list")) closePopover();
});
window.addEventListener("resize", closePopover);

el("example").addEventListener("click", () => replaceText(t("example_text")));
el("clear").addEventListener("click", () => replaceText(""));
el("finish").addEventListener("click", finishRevision);
el("final-continue").addEventListener("click", () => el("final").close());
el("final-new").addEventListener("click", () => { el("final").close(); startNewSession(); });

el("threshold").addEventListener("input", (event) => { el("threshold-value").textContent = Number(event.target.value).toFixed(2); });
el("threshold").addEventListener("change", (event) => {
  state.threshold = Number(event.target.value);
  logEvent("threshold_changed", null, { detection_threshold: state.threshold });
  analyze();
});

async function start() {
  config = await (await fetch("/api/config")).json();

  for (const element of document.querySelectorAll("[data-i18n]")) element.textContent = t(element.dataset.i18n);
  textarea.placeholder = t("editor_placeholder");
  document.title = t("title");

  const legend = el("legend");
  for (const category of CATEGORY_ORDER) {
    const item = node("li", `cat-${category}`);
    item.append(node("span", "chip", config.categories[category].label), node("span", "", config.categories[category].description));
    legend.append(item);
  }

  state.threshold = config.threshold;
  el("threshold").value = config.threshold;
  el("threshold-value").textContent = Number(config.threshold).toFixed(2);

  state.session = newSession();
  setStatus("idle");
  render();

  // A text can be passed in the address, e.g. /?text=su%20trenu...
  const initialText = new URLSearchParams(window.location.search).get("text");
  if (initialText) replaceText(initialText);
}

start();

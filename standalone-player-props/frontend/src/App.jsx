import React, { useEffect, useMemo, useState } from "react";
import { fetchProps, fetchResults } from "./api";

const LEAGUES = ["NBA", "NHL", "NFL", "MLB", "SOCCER", "NCAAB", "WNBA", "NCAAF", "NCAAW"];

/** [header, apiKey] — public names only; apiKey is internal engine field. */
const MODEL_COLUMNS = [
  ["Grinder2", "glicko2"],
  ["Takedown", "trueskill"],
  ["Edge", "xsharp"],
  ["XSharp", "xgboost"],
  ["Sharp Consensus", "sharp_consensus"],
];

function isoDateInEastern(d = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(d);
  const y = parts.find((p) => p.type === "year")?.value;
  const m = parts.find((p) => p.type === "month")?.value;
  const day = parts.find((p) => p.type === "day")?.value;
  return `${y}-${m}-${day}`;
}

/** Calendar math on YYYY-MM-DD (treated as an Eastern calendar anchor, not UTC midnight). */
function addCalendarDaysIso(isoYmd, deltaDays) {
  const [y, m, d] = isoYmd.split("-").map((x) => parseInt(x, 10));
  if (!y || !m || !d) return isoYmd;
  const dt = new Date(Date.UTC(y, m - 1, d + deltaDays));
  const yy = dt.getUTCFullYear();
  const mm = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(dt.getUTCDate()).padStart(2, "0");
  return `${yy}-${mm}-${dd}`;
}

function defaultResultsDateYesterdayEt() {
  return addCalendarDaysIso(isoDateInEastern(), -1);
}

function initialResultsDateFromUrl() {
  try {
    const raw = (new URLSearchParams(window.location.search).get("date") || "").trim().slice(0, 10);
    if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  } catch {
    /* ignore */
  }
  return "";
}

function initialRollupDaysFromUrl() {
  try {
    const raw = (new URLSearchParams(window.location.search).get("rollup_days") || "").trim();
    const n = parseInt(raw, 10);
    if (Number.isFinite(n)) return Math.min(14, Math.max(1, n));
  } catch {
    /* ignore */
  }
  return 1;
}

function initialViewFromUrl() {
  try {
    const v = new URLSearchParams(window.location.search).get("view");
    return v === "results" ? "results" : "props";
  } catch {
    return "props";
  }
}

function initialLeagueFromDomAndUrl() {
  try {
    const el = document.getElementById("root");
    const fromDom = (el?.dataset?.initialLeague || "").trim().toUpperCase();
    if (fromDom) return fromDom;
    const u = new URL(window.location.href);
    return (u.searchParams.get("league") || "").trim().toUpperCase();
  } catch {
    return "";
  }
}

function initialPlayerFilterFromDomAndUrl() {
  try {
    const el = document.getElementById("root");
    const fromDom = (el?.dataset?.playerFilter || "").trim();
    if (fromDom) return fromDom;
    const u = new URL(window.location.href);
    return (u.searchParams.get("player") || "").trim();
  } catch {
    return "";
  }
}

function sportSelectHiddenFromDom() {
  try {
    const el = document.getElementById("root");
    return (el?.dataset?.hideSportSelect || "") === "1";
  } catch {
    return false;
  }
}

const PROP_TYPE_LABELS = {
  points: "Points",
  rebounds: "Rebounds",
  assists: "Assists",
  threes: "3PT Made",
  shots_on_goal: "Shots on Goal",
  goals: "Goals",
  hits: "Hits",
  strikeouts: "Strikeouts",
  runs: "Runs",
  rbis: "RBI",
  home_runs: "HR",
  passing_yards: "Pass Yards",
  rushing_yards: "Rush Yards",
  receiving_yards: "Receiving Yards",
  receptions: "Receptions",
  shots: "Shots",
  shots_on_target: "Shots on Target",
};

function formatPropType(value) {
  if (!value) return "";
  return PROP_TYPE_LABELS[value] || value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function App() {
  const [propsRows, setPropsRows] = useState([]);
  const [playerFilter, setPlayerFilter] = useState(initialPlayerFilterFromDomAndUrl);
  const [hideSportSelect] = useState(sportSelectHiddenFromDom);
  const [selectedLeague, setSelectedLeague] = useState(initialLeagueFromDomAndUrl);
  const [propType, setPropType] = useState("");
  const [side, setSide] = useState("");
  const [propsSlateDate, setPropsSlateDate] = useState(() => isoDateInEastern());
  const [resultsDate, setResultsDate] = useState(
    () => initialResultsDateFromUrl() || defaultResultsDateYesterdayEt(),
  );
  const [resultsRollupDays, setResultsRollupDays] = useState(() => initialRollupDaysFromUrl());
  const [gradedDateLabel, setGradedDateLabel] = useState("");
  const [appliedFilters, setAppliedFilters] = useState({
    league: "",
    propType: "",
    side: "",
    slateDate: "",
    resultsDate: "",
    resultsRollupDays: 1,
  });
  const [resultsDaySummary, setResultsDaySummary] = useState(null);
  const [resultsRollup, setResultsRollup] = useState(null);
  const [resultsRollupTracking, setResultsRollupTracking] = useState(null);
  const [view, setView] = useState(initialViewFromUrl);
  const [resultsRows, setResultsRows] = useState([]);
  const [resultsSummary, setResultsSummary] = useState(null);
  const [resultsTracking, setResultsTracking] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [shareStatus, setShareStatus] = useState("");

  useEffect(() => {
    const lg = initialLeagueFromDomAndUrl();
    if (!lg) return;
    setSelectedLeague(lg);
    const rdInit = initialResultsDateFromUrl() || defaultResultsDateYesterdayEt();
    const rollInit = initialRollupDaysFromUrl();
    setResultsDate(rdInit);
    setResultsRollupDays(rollInit);
    setAppliedFilters({
      league: lg,
      propType: "",
      side: "",
      slateDate: isoDateInEastern(),
      resultsDate: rdInit,
      resultsRollupDays: rollInit,
    });
  }, []);

  useEffect(() => {
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("view", view);
      if (appliedFilters.league) u.searchParams.set("league", appliedFilters.league);
      else u.searchParams.delete("league");
      if (view === "results") {
        if (appliedFilters.resultsDate) u.searchParams.set("date", appliedFilters.resultsDate);
        else u.searchParams.delete("date");
        if (appliedFilters.resultsRollupDays > 1) {
          u.searchParams.set("rollup_days", String(appliedFilters.resultsRollupDays));
        } else u.searchParams.delete("rollup_days");
      } else {
        u.searchParams.delete("date");
        u.searchParams.delete("rollup_days");
      }
      window.history.replaceState({}, "", u);
    } catch {
      /* ignore */
    }
  }, [view, appliedFilters.league, appliedFilters.resultsDate, appliedFilters.resultsRollupDays]);

  useEffect(() => {
    if (!appliedFilters.league) {
      setPropsRows([]);
      setResultsRows([]);
      setResultsSummary(null);
      setResultsTracking(null);
      setResultsDaySummary(null);
      setResultsRollup(null);
      setResultsRollupTracking(null);
      setGradedDateLabel("");
      setLoading(false);
      return;
    }
    async function run() {
      setLoading(true);
      setApiError(null);
      try {
        if (view === "props") {
          const r = await fetchProps({
            league: appliedFilters.league,
            propType: appliedFilters.propType,
            side: appliedFilters.side,
            date: appliedFilters.slateDate || undefined,
          });
          setPropsRows(r.items || []);
        } else {
          const rr = await fetchResults(
            appliedFilters.league,
            appliedFilters.resultsDate || undefined,
            appliedFilters.resultsRollupDays,
          );
          setResultsRows(rr.items || []);
          setResultsSummary(rr.summary || null);
          setResultsTracking(rr.tracking || null);
          setResultsDaySummary(rr.day_summary ?? null);
          setResultsRollup(rr.rollup ?? null);
          setResultsRollupTracking(rr.rollup_tracking ?? null);
          setGradedDateLabel(rr.graded_date || "");
        }
      } catch (e) {
        const msg =
          e instanceof TypeError && e.message === "Failed to fetch"
            ? "Cannot load props right now. Please try again."
            : e instanceof Error
              ? e.message
              : String(e);
        setApiError(msg);
        if (view === "props") {
          setPropsRows([]);
        } else {
          setResultsRows([]);
          setResultsSummary(null);
          setResultsTracking(null);
          setResultsDaySummary(null);
          setResultsRollup(null);
          setResultsRollupTracking(null);
          setGradedDateLabel("");
        }
      } finally {
        setLoading(false);
      }
    }
    run();
  }, [appliedFilters, view]);

  const propTypes = useMemo(() => {
    const s = new Set(propsRows.map((x) => x.prop_type));
    return [...s].sort();
  }, [propsRows]);

  const filteredPropRows = useMemo(() => {
    const pf = playerFilter.trim().toLowerCase();
    if (!pf) return propsRows;
    return propsRows.filter((r) => (r.player_name || "").toLowerCase().includes(pf));
  }, [propsRows, playerFilter]);

  const topProps = useMemo(() => filteredPropRows.slice(0, 30), [filteredPropRows]);
  const shareProps = useMemo(() => topProps.slice(0, 3), [topProps]);

  const shareDate = useMemo(() => {
    const d = appliedFilters.slateDate || isoDateInEastern();
    return d;
  }, [appliedFilters.slateDate]);

  function applyFilters() {
    setAppliedFilters({
      league: selectedLeague,
      propType,
      side,
      slateDate: propsSlateDate,
      resultsDate,
      resultsRollupDays,
    });
  }

  async function buildShareCanvas() {
    const card = document.getElementById("propsShareCard");
    if (!card) throw new Error("Share card not found.");
    const html2canvas = (await import("html2canvas")).default;
    return html2canvas(card, { backgroundColor: "#ffffff", scale: 2, useCORS: true });
  }

  async function saveShareImage() {
    if (shareProps.length === 0) {
      setShareStatus("No props available to export.");
      return;
    }
    setShareStatus("Generating image...");
    try {
      const canvas = await buildShareCanvas();
      canvas.toBlob((blob) => {
        if (!blob) {
          setShareStatus("Could not generate image.");
          return;
        }
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = objectUrl;
        a.download = `player-props-${appliedFilters.league.toLowerCase()}-${shareDate}.png`;
        a.click();
        URL.revokeObjectURL(objectUrl);
        setShareStatus("Image saved.");
      }, "image/png");
    } catch (e) {
      setShareStatus("Could not generate image.");
    }
  }

  async function shareImage() {
    if (shareProps.length === 0) {
      setShareStatus("No props available to share.");
      return;
    }
    setShareStatus("Preparing share image...");
    try {
      const canvas = await buildShareCanvas();
      canvas.toBlob(async (blob) => {
        if (!blob) {
          setShareStatus("Could not generate image.");
          return;
        }
        const file = new File([blob], "player-props-share.png", { type: "image/png" });
        if (navigator.canShare && navigator.canShare({ files: [file] })) {
          try {
            await navigator.share({ files: [file], title: "Top Props" });
            setShareStatus("Shared.");
            return;
          } catch (_) {}
        }
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = objectUrl;
        a.download = "player-props-share.png";
        a.click();
        URL.revokeObjectURL(objectUrl);
        setShareStatus("Image saved for sharing.");
      }, "image/png");
    } catch (e) {
      setShareStatus("Could not generate image.");
    }
  }

  async function openShareImageInBrowser() {
    if (shareProps.length === 0) {
      setShareStatus("No props available to preview.");
      return;
    }
    const w = window.open("", "_blank", "noopener,noreferrer");
    if (!w) {
      setShareStatus("Popup blocked. Allow popups for this site, or use Save image.");
      return;
    }
    setShareStatus("Opening image...");
    try {
      const canvas = await buildShareCanvas();
      const url = canvas.toDataURL("image/png");
      w.location.href = url;
      setShareStatus("Opened in a new tab (you can save from the browser).");
    } catch {
      try { w.close(); } catch {}
      setShareStatus("Could not generate image.");
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <h1>Player Props</h1>
        {view === "props" ? (
          <p className="hero-lede">
            Picks for the slate date you choose (US/Eastern). Use <strong>Results</strong> to grade against box scores (NBA).
          </p>
        ) : (
          <p className="hero-lede">
            Graded vs ESPN box scores (NBA). Game date uses the US/Eastern calendar. Choose a window (e.g. last 5 days) to see each model’s record and daily breakdown; the prop list below is always for the
            selected end date.
          </p>
        )}
        {apiError ? (
          <div className="api-error" role="alert">
            <strong>API error.</strong> {apiError}
          </div>
        ) : null}
      </header>

      <div className="props-tabs" role="tablist" aria-label="Props views">
        <button
          type="button"
          role="tab"
          aria-selected={view === "props"}
          className={`props-tab${view === "props" ? " active" : ""}`}
          onClick={() => setView("props")}
        >
          Picks
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === "results"}
          className={`props-tab${view === "results" ? " active" : ""}`}
          onClick={() => setView("results")}
        >
          Results
        </button>
      </div>

      <section className="filters">
        {!hideSportSelect ? (
          <label>
            Sport
            <select value={selectedLeague} onChange={(e) => setSelectedLeague(e.target.value)}>
              <option value="">Select a sport</option>
              {LEAGUES.map((lg) => (
                <option key={lg} value={lg}>
                  {lg}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <label>
          Player contains
          <input
            type="text"
            value={playerFilter}
            onChange={(e) => setPlayerFilter(e.target.value)}
            placeholder="e.g. Brunson"
            autoComplete="off"
          />
        </label>
        {view === "props" ? (
          <label>
            Slate date (ET)
            <input
              type="date"
              value={propsSlateDate}
              onChange={(e) => setPropsSlateDate(e.target.value)}
            />
          </label>
        ) : (
          <>
            <label>
              Game date (ET)
              <input type="date" value={resultsDate} onChange={(e) => setResultsDate(e.target.value)} />
              <span className="field-hint">Clear to auto-pick the latest slate we find</span>
            </label>
            <label>
              Results window
              <select
                value={resultsRollupDays}
                onChange={(e) => setResultsRollupDays(parseInt(e.target.value, 10) || 1)}
              >
                <option value={1}>This date only</option>
                <option value={3}>Last 3 days</option>
                <option value={5}>Last 5 days</option>
                <option value={7}>Last 7 days</option>
                <option value={10}>Last 10 days</option>
                <option value={14}>Last 14 days</option>
              </select>
              <span className="field-hint">Rolls backward from the game date</span>
            </label>
          </>
        )}
        <label>
          Prop Type
          <select value={propType} onChange={(e) => setPropType(e.target.value)}>
            <option value="">All</option>
            {propTypes.map((t) => (
              <option key={t} value={t}>
                {formatPropType(t)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Side
          <select value={side} onChange={(e) => setSide(e.target.value)}>
            <option value="">All</option>
            <option value="OVER">Over</option>
            <option value="UNDER">Under</option>
          </select>
        </label>
        <button type="button" className="run-btn" onClick={applyFilters} disabled={loading || !selectedLeague}>
          {loading ? "Loading..." : "Run"}
        </button>
      </section>

      <section className="panel">
        <h2>
          {appliedFilters.league}{" "}
          {view === "props" ? `Top Props (${topProps.length})` : `Results (${resultsRows.length})`}
        </h2>
        {view === "results" && resultsSummary ? (
          <>
            {resultsRollup && resultsRollup.days > 1 ? (
              <div className="results-rollup-note" role="status">
                <strong>{resultsRollup.days}-day</strong> roll-up ending{" "}
                <strong>{resultsRollup.anchor_date}</strong>. Summary and model tables include every graded prop in that span. The prop list
                below is only the slate for the end date ({gradedDateLabel || resultsRollup.anchor_date}).
                {resultsRollupTracking ? (
                  <>
                    {" "}
                    Window total graded: <strong>{resultsRollupTracking.graded_props ?? 0}</strong>.
                  </>
                ) : null}
              </div>
            ) : null}
            {resultsDaySummary && resultsRollup && resultsRollup.days > 1 ? (
              <p className="results-summary">
                <span style={{ fontWeight: 800 }}>Main pick on end date:</span>{" "}
                {resultsDaySummary.overall?.wins ?? 0}-{resultsDaySummary.overall?.losses ?? 0}
                {" · "}
                {Object.entries(resultsDaySummary.by_prop_type || {})
                  .map(([k, v]) => `${formatPropType(k)}: ${v.wins}-${v.losses}`)
                  .join(" · ") || "—"}
              </p>
            ) : null}
            <p className="results-summary">
              {gradedDateLabel ? (
                <>
                  <span style={{ fontWeight: 800 }}>Box score date:</span> {gradedDateLabel}
                  {" · "}
                </>
              ) : null}
              {resultsRollup && resultsRollup.days > 1 ? (
                <span style={{ fontWeight: 800 }}>Window overall</span>
              ) : (
                <span style={{ fontWeight: 800 }}>Overall</span>
              )}
              : {resultsSummary.overall?.wins ?? 0}-{resultsSummary.overall?.losses ?? 0}
              {" · "}
              {Object.entries(resultsSummary.by_prop_type || {})
                .map(([k, v]) => `${formatPropType(k)}: ${v.wins}-${v.losses}`)
                .join(" · ") || "—"}
              {resultsTracking ? (
                <>
                  {" · "}
                  <span style={{ fontWeight: 700 }}>End-date slate</span>: generated {resultsTracking.generated_props ?? 0}, graded{" "}
                  {resultsTracking.graded_props ?? 0}, candidates {resultsTracking.candidate_players ?? 0}, excluded{" "}
                  {resultsTracking.excluded_players ?? 0}, out-of-scope {resultsTracking.scope_excluded_players ?? 0}
                </>
              ) : null}
            </p>
            {resultsRollupTracking && resultsRollup && resultsRollup.days > 1 ? (
              <p className="results-summary">
                <span style={{ fontWeight: 800 }}>Window totals</span>: graded {resultsRollupTracking.graded_props ?? 0}, candidates{" "}
                {resultsRollupTracking.candidate_players ?? 0}, excluded {resultsRollupTracking.excluded_players ?? 0}
              </p>
            ) : null}

            <h3 className="results-subhead">
              Model record{resultsRollup && resultsRollup.days > 1 ? " (window)" : ""}
            </h3>
            <div className="table-wrap results-secondary-table">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Record</th>
                    <th>Win %</th>
                    <th>Sample</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_COLUMNS.map(([label, apiKey]) => {
                    const m = resultsSummary.by_model?.[apiKey];
                    const w = m?.wins ?? 0;
                    const l = m?.losses ?? 0;
                    const t = w + l;
                    return (
                      <tr key={apiKey}>
                        <td>{m?.label || label}</td>
                        <td>{t ? `${w}-${l}` : "—"}</td>
                        <td>{m?.win_pct != null ? `${m.win_pct}%` : "—"}</td>
                        <td>{t}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {resultsRollup?.by_date?.length ? (
              <>
                <h3 className="results-subhead">Daily breakdown</h3>
                <div className="table-wrap results-secondary-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Games</th>
                        <th>Graded</th>
                        <th>Overall</th>
                        {MODEL_COLUMNS.map(([label]) => (
                          <th key={label} className="model-col">
                            {label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {resultsRollup.by_date.map((row) => (
                        <tr key={row.date}>
                          <td>{row.date}</td>
                          <td>{row.games ?? "—"}</td>
                          <td>{row.graded_props ?? 0}</td>
                          <td>
                            {row.summary?.overall?.wins ?? 0}-{row.summary?.overall?.losses ?? 0}
                          </td>
                          {MODEL_COLUMNS.map(([, apiKey]) => {
                            const m = row.summary?.by_model?.[apiKey];
                            const w = m?.wins ?? 0;
                            const l = m?.losses ?? 0;
                            const t = w + l;
                            return (
                              <td key={`${row.date}-${apiKey}`} className="model-col">
                                {t ? `${w}-${l}` : "—"}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}
          </>
        ) : null}
        {loading ? (
          <p>Loading...</p>
        ) : (
          <div className="table-wrap">
            <table>
              {view === "props" ? (
                <>
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Team</th>
                      <th>Prop</th>
                      <th>Line</th>
                      <th>Projection</th>
                      <th>Pick</th>
                      {MODEL_COLUMNS.map(([label]) => (
                        <th key={label} className="model-col" title="Model confidence on this side">
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {topProps.map((r) => (
                      <tr key={`${r.player_id}-${r.prop_type}-${r.line}`}>
                        <td>{r.player_name}</td>
                        <td>{r.team}</td>
                        <td>{formatPropType(r.prop_type)}</td>
                        <td>{r.line ?? "-"}</td>
                        <td>{r.projection}</td>
                        <td>{r.picked_side}</td>
                        {MODEL_COLUMNS.map(([, apiKey]) => (
                          <td key={`${r.player_id}-${r.prop_type}-${apiKey}`} className="model-col">
                            {r.model_confidence && r.model_confidence[apiKey] != null
                              ? `${r.model_confidence[apiKey]}%`
                              : "—"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </>
              ) : (
                <>
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Team</th>
                      <th>Prop</th>
                      <th>Pick</th>
                      <th>Line</th>
                      <th>Actual</th>
                      <th>Result</th>
                      {MODEL_COLUMNS.map(([label]) => (
                        <th key={label} className="model-col" title={`${label} vs line (graded)`}>
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {resultsRows.map((r) => (
                      <tr key={`${r.player_id}-${r.prop_type}-${r.line}-${r.actual}`}>
                        <td>{r.player_name}</td>
                        <td>{r.team}</td>
                        <td>{formatPropType(r.prop_type)}</td>
                        <td>{r.pick}</td>
                        <td>{r.line}</td>
                        <td>{r.actual}</td>
                        <td>{r.result}</td>
                        {MODEL_COLUMNS.map(([, apiKey]) => {
                          const mb = r.model_breakdown?.[apiKey];
                          return (
                            <td key={`${r.player_id}-${r.prop_type}-${apiKey}`} className="model-col">
                              {mb ? (mb.hit ? "Hit" : "Miss") : "—"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </>
              )}
            </table>
          </div>
        )}

        {view === "props" ? (
          <div className="share-panel-inner">
            <h3 className="share-subhead">Top 3 props — share image</h3>
            <p className="results-summary">
              Share card matches the top three rows in the table. Use Save image, Share image, or Open in browser (PNG in a new tab).
            </p>
            <div className="share-actions">
              <button type="button" className="run-btn" onClick={saveShareImage}>
                Save image
              </button>
              <button type="button" className="run-btn secondary-btn" onClick={shareImage}>
                Share image
              </button>
              <button type="button" className="run-btn secondary-btn" onClick={openShareImageInBrowser}>
                Open in browser
              </button>
            </div>
            {shareStatus ? <p className="results-summary">{shareStatus}</p> : null}
            <div id="propsShareCard" className="props-share-card">
              <div className="props-share-head">{appliedFilters.league} Top Props</div>
              {shareProps.length === 0 ? (
                <div className="props-share-empty">No props available.</div>
              ) : (
                shareProps.map((r, idx) => (
                  <div className="props-share-row" key={`${r.player_id}-${r.prop_type}-${idx}`}>
                    <div className="ps-player">{r.player_name}</div>
                    <div className="ps-prop">{formatPropType(r.prop_type)}</div>
                    <div className="ps-pick">
                      {r.picked_side} {r.line ?? "-"}
                    </div>
                  </div>
                ))
              )}
              <div className="props-share-date">{shareDate}</div>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

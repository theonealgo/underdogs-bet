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
  const [selectedLeague, setSelectedLeague] = useState(initialLeagueFromDomAndUrl);
  const [propType, setPropType] = useState("");
  const [side, setSide] = useState("");
  const [propsSlateDate, setPropsSlateDate] = useState(() => isoDateInEastern());
  const [resultsDate, setResultsDate] = useState("");
  const [gradedDateLabel, setGradedDateLabel] = useState("");
  const [appliedFilters, setAppliedFilters] = useState({
    league: "",
    propType: "",
    side: "",
    slateDate: "",
    resultsDate: "",
  });
  const [view, setView] = useState(initialViewFromUrl);
  const [resultsRows, setResultsRows] = useState([]);
  const [resultsSummary, setResultsSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [shareStatus, setShareStatus] = useState("");
  const [sharePreviewUrl, setSharePreviewUrl] = useState("");

  useEffect(() => {
    const lg = initialLeagueFromDomAndUrl();
    if (!lg) return;
    setSelectedLeague(lg);
    setAppliedFilters({
      league: lg,
      propType: "",
      side: "",
      slateDate: isoDateInEastern(),
      resultsDate: "",
    });
  }, []);

  useEffect(() => {
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("view", view);
      if (appliedFilters.league) u.searchParams.set("league", appliedFilters.league);
      else u.searchParams.delete("league");
      window.history.replaceState({}, "", u);
    } catch {
      /* ignore */
    }
  }, [view, appliedFilters.league]);

  useEffect(() => {
    if (!appliedFilters.league) {
      setPropsRows([]);
      setResultsRows([]);
      setResultsSummary(null);
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
          );
          setResultsRows(rr.items || []);
          setResultsSummary(rr.summary || null);
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

  const topProps = useMemo(() => propsRows.slice(0, 30), [propsRows]);
  const shareProps = useMemo(() => topProps.slice(0, 3), [topProps]);

  useEffect(() => {
    if (view !== "props" || shareProps.length === 0) {
      setSharePreviewUrl("");
      return;
    }
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const card = document.getElementById("propsShareCard");
        if (!card || cancelled) return;
        const html2canvas = (await import("html2canvas")).default;
        const canvas = await html2canvas(card, { backgroundColor: "#ffffff", scale: 2, useCORS: true });
        if (!cancelled) setSharePreviewUrl(canvas.toDataURL("image/png"));
      } catch {
        if (!cancelled) setSharePreviewUrl("");
      }
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [view, shareProps, appliedFilters.league]);
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
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `player-props-${appliedFilters.league.toLowerCase()}-${shareDate}.png`;
        a.click();
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
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "player-props-share.png";
        a.click();
        setShareStatus("Image saved for sharing.");
      }, "image/png");
    } catch (e) {
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
            Graded vs closing lines from ESPN box scores. Pick a game date to review past slates (NBA). Leave date empty to use the latest completed slate we find.
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
          <label>
            Game date
            <input
              type="date"
              value={resultsDate}
              onChange={(e) => setResultsDate(e.target.value)}
            />
            <span className="field-hint">Optional — blank = latest slate</span>
          </label>
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
          <p className="results-summary">
            {gradedDateLabel ? (
              <>
                <span style={{ fontWeight: 800 }}>Box score date:</span> {gradedDateLabel}
                {" · "}
              </>
            ) : null}
            Overall: {resultsSummary.overall?.wins ?? 0}-{resultsSummary.overall?.losses ?? 0}
            {" | "}
            {Object.entries(resultsSummary.by_prop_type || {})
              .map(([k, v]) => `${formatPropType(k)}: ${v.wins}-${v.losses}`)
              .join(" | ")}
          </p>
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
            <p className="results-summary">Preview updates from the table above. Save or share for social.</p>
            <div className="share-actions">
              <button type="button" className="run-btn" onClick={saveShareImage}>
                Save image
              </button>
              <button type="button" className="run-btn secondary-btn" onClick={shareImage}>
                Share image
              </button>
            </div>
            {shareStatus ? <p className="results-summary">{shareStatus}</p> : null}
            {sharePreviewUrl ? (
              <div className="props-share-preview-wrap">
                <img src={sharePreviewUrl} alt="Top three player props preview" className="props-share-preview-img" />
              </div>
            ) : null}
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

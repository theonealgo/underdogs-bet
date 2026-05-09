function resolveApiBase() {
  const fromEnv = import.meta.env.VITE_PROPS_API_BASE;
  if (fromEnv != null && String(fromEnv).trim() !== "") {
    return String(fromEnv).replace(/\/$/, "");
  }
  if (import.meta.env.DEV) {
    return "/api";
  }
  return "/player-props-api";
}

const API_BASE = resolveApiBase();
const API_TIMEOUT_MS = 20000;

async function fetchWithTimeout(url, options = {}, timeoutMs = API_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error("Request timed out. Please click Run again.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function readJsonResponse(response) {
  const text = await response.text();
  let body = {};
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text.slice(0, 240) };
    }
  }
  if (!response.ok) {
    const detail = body.detail ?? body.message;
    const msg =
      typeof detail === "string"
        ? detail
        : detail != null
          ? JSON.stringify(detail)
          : `${response.status} ${response.statusText}`;
    throw new Error(msg);
  }
  return body;
}

export function getPropsApiBase() {
  return API_BASE;
}

export async function fetchLeagues() {
  const r = await fetchWithTimeout(`${API_BASE}/leagues`);
  return readJsonResponse(r);
}

export async function fetchPlayers(league) {
  const r = await fetchWithTimeout(`${API_BASE}/players?league=${encodeURIComponent(league)}`);
  return readJsonResponse(r);
}

export async function fetchProps({ league, propType, side, date }) {
  const q = new URLSearchParams({ league });
  if (propType) q.set("prop_type", propType);
  if (side) q.set("side", side);
  if (date) q.set("date", date);
  const r = await fetchWithTimeout(`${API_BASE}/props?${q.toString()}`);
  return readJsonResponse(r);
}

export async function fetchResults(league, date, rollupDays = 1) {
  const q = new URLSearchParams({ league });
  if (date) q.set("date", date);
  const rd = Math.min(14, Math.max(1, parseInt(String(rollupDays || 1), 10) || 1));
  if (rd > 1) q.set("rollup_days", String(rd));
  const r = await fetchWithTimeout(`${API_BASE}/results?${q.toString()}`);
  return readJsonResponse(r);
}

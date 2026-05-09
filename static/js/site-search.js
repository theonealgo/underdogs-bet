/**
 * Site search: /api/search/suggest for autocomplete; optional /api/search detail panel.
 */
(function () {
  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/"/g, "&quot;");
  }

  function setupSearchForm(form) {
    if (!form || form.dataset.siteSearchBound) return;
    form.dataset.siteSearchBound = "1";
    var input = form.querySelector('input[name="query"],input[name="q"]');
    var wrap = form.closest(".nav-search-wrap");
    var acId = form.getAttribute("data-autocomplete-id");
    var resId = form.getAttribute("data-results-id");
    var ac =
      (acId && document.getElementById(acId)) ||
      (wrap && wrap.querySelector(".search-autocomplete")) ||
      null;
    var resultsEl = (resId && document.getElementById(resId)) || null;
    var fullSearch = form.getAttribute("data-full-search") === "1";
    if (!input || !ac) return;

    var debounceTimer = null;
    input.addEventListener("input", function () {
      var q = (input.value || "").trim();
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function () {
        if (q.length < 2) {
          ac.classList.remove("show");
          ac.innerHTML = "";
          return;
        }
        fetch("/api/search/suggest?q=" + encodeURIComponent(q), {
          headers: { Accept: "application/json" },
        })
          .then(function (r) {
            return r.json();
          })
          .then(function (data) {
            var items = (data && data.items) || [];
            if (!items.length) {
              ac.innerHTML =
                '<div class="search-item"><span>No matches</span></div>';
              ac.classList.add("show");
              return;
            }
            ac.innerHTML = items
              .map(function (it) {
                return (
                  '<a class="search-item" href="' +
                  esc(it.url) +
                  '"><span>' +
                  esc(it.title) +
                  "</span><small>" +
                  esc(it.subtitle || "") +
                  "</small></a>"
                );
              })
              .join("");
            ac.classList.add("show");
          })
          .catch(function () {
            ac.innerHTML =
              '<div class="search-item"><span>Search unavailable</span></div>';
            ac.classList.add("show");
          });
      }, 280);
    });

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var query = (input.value || "").trim();
      if (!query) {
        if (resultsEl) {
          resultsEl.classList.remove("show");
          resultsEl.innerHTML = "";
        }
        return;
      }
      if (fullSearch && resultsEl) {
        resultsEl.classList.add("show");
        resultsEl.innerHTML = "<p>Searching…</p>";
        fetch("/api/search?query=" + encodeURIComponent(query), {
          headers: { Accept: "application/json" },
        })
          .then(function (r) {
            return r.json();
          })
          .then(function (data) {
            function section(title, itemsHtml) {
              if (!itemsHtml) return "";
              return (
                '<h3 class="ud-search-section-title">' +
                esc(title) +
                '</h3><ul class="ud-search-hit-list">' +
                itemsHtml +
                "</ul>"
              );
            }
            function rowTeam(r) {
              var href = r.url || "/";
              var title =
                r.title ||
                [r.away_team, r.home_team].filter(Boolean).join(" @ ");
              var sub =
                r.subtitle ||
                [
                  r.game_date,
                  r.pick_line || r.predicted_winner,
                  r.sport,
                ]
                  .filter(function (x) {
                    return x != null && String(x).trim() !== "";
                  })
                  .join(" · ");
              var props =
                r.props_url
                  ? '<a class="ud-search-hit-secondary" href="' +
                    esc(r.props_url) +
                    '">Player props</a>'
                  : "";
              return (
                '<li class="ud-search-hit">' +
                '<a class="ud-search-hit-main" href="' +
                esc(href) +
                '">' +
                '<span class="ud-search-hit-title">' +
                esc(title) +
                "</span>" +
                '<span class="ud-search-hit-sub">' +
                esc(sub) +
                "</span></a>" +
                props +
                "</li>"
              );
            }
            function rowEspn(r) {
              var href = r.url || "/";
              var title =
                r.title ||
                [r.away_team, r.home_team].filter(Boolean).join(" @ ");
              var sub =
                r.subtitle ||
                [
                  r.game_date,
                  r.sport,
                  r.status,
                ]
                  .filter(Boolean)
                  .join(" · ");
              var ext =
                r.espn_url
                  ? '<a class="ud-search-hit-secondary ud-search-hit-secondary--external" href="' +
                    esc(r.espn_url) +
                    '" target="_blank" rel="noopener noreferrer">ESPN game</a>'
                  : "";
              return (
                '<li class="ud-search-hit ud-search-hit--espn">' +
                '<a class="ud-search-hit-main" href="' +
                esc(href) +
                '">' +
                '<span class="ud-search-hit-title">' +
                esc(title) +
                "</span>" +
                '<span class="ud-search-hit-sub">' +
                esc(sub) +
                "</span></a>" +
                ext +
                "</li>"
              );
            }
            function rowModel(r) {
              var href = r.url || "/results";
              var title = r.title || (r.sport ? r.sport + " results" : "Results");
              var sub =
                r.subtitle != null && String(r.subtitle).trim() !== ""
                  ? esc(r.subtitle)
                  : esc(r.sport) +
                    ": " +
                    esc(r.record) +
                    " (" +
                    esc(String(r.accuracy)) +
                    "%)" +
                    (r.filtered_games != null
                      ? " — " + esc(String(r.filtered_games)) + " games at threshold"
                      : "");
              return (
                '<li class="ud-search-hit">' +
                '<a class="ud-search-hit-main" href="' +
                esc(href) +
                '">' +
                '<span class="ud-search-hit-title">' +
                esc(title) +
                "</span>" +
                '<span class="ud-search-hit-sub">' +
                sub +
                "</span></a></li>"
              );
            }
            var modelLine = data.matched_model
              ? "<p><strong>Model:</strong> " +
                esc(data.matched_model.public_name) +
                " → " +
                esc(data.matched_model.internal_name) +
                (data.confidence_threshold
                  ? " (confidence ≥ " + data.confidence_threshold + "%)"
                  : "") +
                "</p>"
              : "";
            var modelItems = (data.model_results || [])
              .map(rowModel)
              .join("");
            var localTeamItems = (data.team_results || [])
              .map(rowTeam)
              .join("");
            var espnItems = (data.espn_results || [])
              .map(rowEspn)
              .join("");
            var routeLine = data.suggested_route
              ? '<div class="ud-search-suggest-banner"><a class="ud-search-suggest-link" href="' +
                esc(data.suggested_route) +
                '">' +
                esc(data.suggested_route_label || data.suggested_route) +
                "</a></div>"
              : "";
            var empty =
              !modelItems && !localTeamItems && !espnItems
                ? "<p>No database rows matched. Try the dropdown suggestions while typing.</p>"
                : "";
            resultsEl.innerHTML =
              '<h3 style="margin-top:0;">Search results</h3>' +
              modelLine +
              routeLine +
              section("Model performance", modelItems) +
              section("Our predictions", localTeamItems) +
              section("Scoreboard (today)", espnItems) +
              empty;
          })
          .catch(function () {
            resultsEl.innerHTML =
              "<p>Search temporarily unavailable. Please try again.</p>";
          });
        return;
      }
      fetch("/api/search/suggest?q=" + encodeURIComponent(query), {
        headers: { Accept: "application/json" },
      })
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          var items = (data && data.items) || [];
          if (items.length >= 1) {
            window.location.href = items[0].url;
            return;
          }
          window.location.href =
            "/search?query=" + encodeURIComponent(query);
        })
        .catch(function () {
          window.location.href =
            "/search?query=" + encodeURIComponent(query);
        });
    });

    document.addEventListener("click", function (event) {
      if (!wrap || wrap.contains(event.target)) return;
      ac.classList.remove("show");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-site-search-form]").forEach(setupSearchForm);
  });
})();

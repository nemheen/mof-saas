// src/App.js
import React, { useState, useEffect, useCallback } from "react";
import "./App.css";
import PredictionChart from "./PredictionChart";
import GeminiChat from "./components/GeminiChat";
import { useAuth } from "./AuthContext";

// ===============================
// Inline hooks (no extra files)
// ===============================
function useHistoryApi(api) {
  const [items, setItems] = useState([]);
  const [nextCursorTs, setNextCursorTs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const reset = useCallback(() => {
    setItems([]);
    setNextCursorTs(null);
    setErr("");
  }, []);

  const normalize = (data) => {
    const arr = Array.isArray(data) ? data : Array.isArray(data?.items) ? data.items : [];
    const cursor = Array.isArray(data) ? null : data?.next_cursor_ts ?? null;
    const norm = arr.map((it) => {
      const tsIso =
        it.ts_iso ||
        (typeof it.ts === "string" ? it.ts : null) ||
        (it.timestamp && typeof it.timestamp === "string" ? it.timestamp : null);
      return { ...it, ts_iso: tsIso };
    });
    return { arr: norm, cursor };
  };

  const fetchPage = useCallback(
    async ({ limit = 20, cursorTs = null } = {}) => {
      if (!api) return;
      setLoading(true);
      setErr("");
      try {
        const params = { limit };
        if (cursorTs) params.cursor_ts = cursorTs;
        const { data } = await api.get("/user/history", { params });
        const { arr, cursor } = normalize(data);
        setItems((prev) => (cursorTs ? [...prev, ...arr] : arr));
        setNextCursorTs(cursor);
      } catch (e) {
        console.error(e);
        setErr("Failed to load history.");
      } finally {
        setLoading(false);
      }
    },
    [api]
  );

  return { items, nextCursorTs, hasMore: !!nextCursorTs, loading, err, fetchPage, reset };
}

function useFavoritesApi(api) {
  const [items, setItems] = useState([]);
  const [nextCursor, setNextCursor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const reset = useCallback(() => {
    setItems([]);
    setNextCursor(null);
    setErr("");
  }, []);

  const fetchPage = useCallback(
    async ({ limit = 50, cursor = null } = {}) => {
      if (!api) return;
      setLoading(true);
      setErr("");
      try {
        const params = { limit };
        if (cursor) params.cursor = cursor;
        const { data } = await api.get("/user/favorites", { params });
        const newItems = Array.isArray(data?.items) ? data.items : [];
        setItems((prev) => (cursor ? [...prev, ...newItems] : newItems));
        setNextCursor(data?.next_cursor ?? null);
      } catch (e) {
        console.error(e);
        setErr("Failed to load favorites.");
      } finally {
        setLoading(false);
      }
    },
    [api]
  );

  const add = useCallback(
    async (mofId, filename) => {
      if (!mofId) return;
      try {
        await api.post(`/user/favorites/${encodeURIComponent(mofId)}`, null, {
          params: filename ? { filename } : {},
        });
        // optimistic update
        setItems((prev) => [
          { id: mofId, filename: filename || mofId, added_at: new Date().toISOString() },
          ...prev,
        ]);
      } catch (e) {
        console.error("Add favorite failed", e);
        alert(e?.response?.data?.detail || "Failed to add favorite. Are you logged in?");
      }
    },
    [api]
  );

  const remove = useCallback(
    async (mofId) => {
      try {
        await api.delete(`/user/favorites/${encodeURIComponent(mofId)}`);
        setItems((prev) => prev.filter((x) => (x.id ?? x.filename) !== mofId));
      } catch (e) {
        console.error("Remove favorite failed", e);
        alert(e?.response?.data?.detail || "Failed to remove favorite.");
      }
    },
    [api]
  );

  return { items, nextCursor, hasMore: !!nextCursor, loading, err, fetchPage, reset, add, remove };
}

// ===============================
// Small UI helpers
// ===============================

// Single, robust FavoriteButton – uses `filename` prop everywhere.
// We DON'T disable when logged out; we alert via requireAuth in the click handlers.
function FavoriteButton({ filename, favorites, onAdd, onRemove, isAuthed, requireAuth }) {
  const id = String(filename || "").trim().replace(/\//g, "_");
  const isFav = favorites.some((f) => (f.id ?? f.filename) === id);

  if (isFav) {
    return (
      <button
        className="chip danger"
        onClick={() => {
          if (!isAuthed && !requireAuth()) return;
          onRemove(id);
        }}
        title="Remove from favorites"
      >
        ★ Remove
      </button>
    );
  }

  return (
    <button
      className="chip"
      onClick={() => {
        if (!isAuthed && !requireAuth()) return;
        let target = id;
        // If no id (common for image_search), prompt the user once.
        if (!target) {
          const input = prompt("Enter a MOF id to add to favorites (e.g., ABAYIO_clean.cif):");
          if (!input) return;
          target = String(input).trim().replace(/\//g, "_");
        }
        onAdd(target, target);
      }}
      title="Add to favorites"
    >
      ☆ Add
    </button>
  );
}

// Try to guess a MOF filename from free text (image_search response)
function suggestMofFromText(txt) {
  if (!txt || typeof txt !== "string") return "";
  const mCif = txt.match(/\b([A-Za-z0-9._-]+\.cif)\b/);
  if (mCif) return mCif[1];
  const mLoud = txt.match(/\b([A-Z0-9][A-Z0-9_-]{3,})\b/);
  return mLoud ? `${mLoud[1]}.cif` : "";
}

// ===============================
// Main App
// ===============================
export default function App() {
  const { api, userId, isAuthed, login, signup, logout, requireAuth } = useAuth();

  // ----- Layout state -----
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [leftWidth, setLeftWidth] = useState(260);
  const [isDragging, setIsDragging] = useState(false);

  // ----- CIF prediction -----
  const [cifFile, setCifFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [adsorptionPlot, setAdsorptionPlot] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // ----- Image search (Gemini Vision) -----
  const [imageFile, setImageFile] = useState(null);
  const [imagePredictionResponse, setImagePredictionResponse] = useState(null);
  const [isImageLoading, setIsImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");
  const [imageFavName, setImageFavName] = useState("");

  // ----- DB Search & Recommendation -----
  const [searchKey, setSearchKey] = useState("filename");
  const [searchValue, setSearchValue] = useState("");
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState("");

  const [recommendationQuery, setRecommendationQuery] = useState("");
  const [minAsa, setMinAsa] = useState(0.0);
  const [minPld, setMinPld] = useState(0.0);
  const [recommendation, setRecommendation] = useState(null);
  const [isRecommending, setIsRecommending] = useState(false);
  const [recommendationError, setRecommendationError] = useState("");

  // ----- Selected MOF details and plot -----
  const [selectedMofDetails, setSelectedMofDetails] = useState(null);
  const [selectedMofPlot, setSelectedMofPlot] = useState(null);
  const [isMofDetailLoading, setIsMofDetailLoading] = useState(false);
  const [mofDetailError, setMofDetailError] = useState("");

  // ----- Active Navigation -----
  const [activeNav, setActiveNav] = useState("Overview");

  // Backend-driven history and favorites
  const historyApi = useHistoryApi(api);
  const favApi = useFavoritesApi(api);

  // Preload favorites when authenticated (so stars render correctly outside Favorites tab)
  useEffect(() => {
    if (isAuthed) {
      favApi.fetchPage({ limit: 50 });
    } else {
      favApi.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthed]);

  // ----- Drag handling -----
  useEffect(() => {
    const onMove = (e) => {
      if (!isDragging) return;
      const min = 200;
      const max = 420;
      const next = Math.min(max, Math.max(min, e.clientX));
      setLeftWidth(next);
    };
    const onUp = () => setIsDragging(false);

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [isDragging]);

  // ----- File handlers -----
  const handleFileChange = (e) => setCifFile(e.target.files[0]);
  const handleImageFileChange = (e) => setImageFile(e.target.files[0]);

  // ----- Predict -----
  const handleUpload = async () => {
    if (!requireAuth()) return;
    if (!cifFile) {
      setError("Please select a CIF file first.");
      return;
    }
    setIsLoading(true);
    setError("");
    setPredictions(null);
    setAdsorptionPlot(null);

    const formData = new FormData();
    formData.append("file", cifFile);

    try {
      const res = await api.post("/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPredictions(res?.data?.prediction || null);
      setAdsorptionPlot(res?.data?.adsorption_plot || null);
    } catch (err) {
      console.error(err);
      setError(err?.response?.data?.detail || "Failed to get prediction. Please check the server and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // ----- Image search -----
  const handleImageUpload = async () => {
    if (!requireAuth()) return;
    if (!imageFile) {
      setImageError("Please select an image file first.");
      return;
    }
    setIsImageLoading(true);
    setImageError("");
    setImagePredictionResponse(null);

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const res = await api.post("/image_search", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const text = res?.data?.response || "No image analysis returned.";
      setImagePredictionResponse(text);
      const guess = suggestMofFromText(text);
      setImageFavName(guess);
    } catch (err) {
      console.error(err);
      setImageError(err?.response?.data?.detail || "Failed to perform image search. Please check the server and try again.");
    } finally {
      setIsImageLoading(false);
    }
  };

  // ----- Downloads -----
  const downloadFile = (blob, filename) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // ----- MOF detail -----
  const fetchMofDetailsAndPlot = async (filename) => {
    setIsMofDetailLoading(true);
    setMofDetailError("");
    setSelectedMofDetails(null);
    setSelectedMofPlot(null);

    try {
      const detailsRes = await api.get(`/coremofs/${filename}`);
      setSelectedMofDetails(detailsRes.data);

      const plotRes = await api.get(`/coremofs/${filename}/plot`);
      setSelectedMofPlot(plotRes.data.adsorption_plot);
    } catch (err) {
      console.error(`Failed to fetch MOF details or plot for ${filename}:`, err);
      if (err.response && err.response.status === 404) {
        setMofDetailError(`MOF '${filename}' not found or adsorption data is missing.`);
      } else {
        setMofDetailError(`Failed to load details for ${filename}. Please check the server.`);
      }
    } finally {
      setIsMofDetailLoading(false);
    }
  };

  // ----- Search -----
  const handleSearch = async () => {
    if (!requireAuth()) return;
    setIsSearching(true);
    setSearchError("");
    setSearchResults(null);
    setSelectedMofDetails(null);
    setSelectedMofPlot(null);

    if (!searchValue) {
      setSearchError("Please provide a search value.");
      setIsSearching(false);
      return;
    }

    const params = {};
    params[searchKey] = searchValue;

    try {
      const res = await api.get("/coremofs/search", { params });
      setSearchResults(res.data);
      if (res.data.length === 0) {
        setSearchError("No MOFs found for the given criteria.");
      }
    } catch (err) {
      console.error("Search failed:", err);
      if (err.response?.status === 400) {
        setSearchError(err.response.data.detail);
      } else {
        setSearchError("Failed to search database. Please check your input and try again.");
      }
    } finally {
      setIsSearching(false);
    }
  };

  // ----- Recommendation -----
  const handleRecommend = async () => {
    if (!requireAuth()) return;
    if (!recommendationQuery.trim()) {
      setRecommendationError("Please enter a recommendation query.");
      return;
    }
    setIsRecommending(true);
    setRecommendationError("");
    setRecommendation(null);

    try {
      const res = await api.post("/recommend_filtered", {
        requirement: recommendationQuery,
        min_asa: parseFloat(minAsa),
        min_pld: parseFloat(minPld),
      });
      // API returns a payload with "recommendation" and more fields.
      setRecommendation(res.data?.recommendation || res.data?.llm_recommendation || "No recommendation text returned.");
    } catch (err) {
      console.error("Recommendation failed:", err);
      setRecommendationError(err?.response?.data?.detail || "Failed to get recommendation. Please try again.");
    } finally {
      setIsRecommending(false);
    }
  };

  // ----- Nav -----
  const handleNavClick = useCallback(
    (viewName) => {
      // Guard protected tabs
      if ((viewName === "Favorites" || viewName === "History") && !isAuthed) {
        if (!requireAuth(viewName === "Favorites" ? "Please log in to view Favorites." : "Please log in to view your recent activity.")) {
          return;
        }
      }
      setActiveNav(viewName);
    },
    [isAuthed, requireAuth]
  );

  // ----- Load pages when opened -----
  useEffect(() => {
    if (activeNav === "History" && isAuthed) {
      historyApi.reset();
      historyApi.fetchPage({ limit: 20 });
    }
  }, [activeNav, isAuthed]); // eslint-disable-line

  useEffect(() => {
    if (activeNav === "Favorites" && isAuthed) {
      favApi.reset();
      favApi.fetchPage({ limit: 50 });
    }
  }, [activeNav, isAuthed]); // eslint-disable-line

  const chatApi = api;
  const shortId = userId ? String(userId).slice(0, 6) : "...";

  return (
    <div className="shell">
      {/* Top Navbar */}
      <header className="topbar">
        <div className="brand">
          <button className="icon-btn" onClick={() => setLeftOpen(!leftOpen)} title="Toggle dashboard">
            ☰
          </button>
          <span>MOF Platform</span>
        </div>
        <div className="top-actions">
          {isAuthed ? (
            <button className="primary" onClick={logout}>
              Log out ({shortId})
            </button>
          ) : (
            <>
              <button
                className="primary"
                onClick={async () => {
                  const email = prompt("Email:");
                  const pw = prompt("Password:");
                  if (email && pw) await login(email, pw);
                }}
              >
                Log in
              </button>
              <button
                className="primary"
                onClick={async () => {
                  const email = prompt("Email:");
                  const pw = prompt("Password (min 8 chars):");
                  if (email && pw) await signup(email, pw);
                }}
              >
                Sign up
              </button>
            </>
          )}
        </div>
      </header>

      {/* Body layout */}
      <div className="body">
        {/* Left Dashboard (draggable) */}
        <aside className={`sidebar ${leftOpen ? "open" : "closed"}`} style={{ width: leftOpen ? leftWidth : 64 }}>
          <nav>
            <div className="section-title">Dashboard</div>
            <button className={`nav-item ${activeNav === "Overview" ? "active" : ""}`} onClick={() => handleNavClick("Overview")}>
              🏠 Overview
            </button>
            <div className="divider" />
            <button className={`nav-item ${activeNav === "Favorites" ? "active" : ""}`} onClick={() => handleNavClick("Favorites")}>
              ⭐ Favorites
            </button>
            <button className={`nav-item ${activeNav === "History" ? "active" : ""}`} onClick={() => handleNavClick("History")}>
              🕘 History
            </button>
          </nav>

          {/* drag handle */}
          {leftOpen && <div className="drag-handle" onMouseDown={() => setIsDragging(true)} />}
        </aside>

        {/* Center Content */}
        <main className="content">
          {/* Overview */}
          {activeNav === "Overview" && (
            <>
              <div className="hero">
                <h1>Carbon-Capture MOF Studio</h1>
                <p className="sub">
                  Upload a <code>.cif</code> file to predict pore/adsorption properties, visualize results, and explore material
                  recommendations via Gemini RAG.
                </p>
              </div>

              {/* CIF Card */}
              <section className="card">
                <div className="card-head">
                  <h2>📦 CIF File Prediction</h2>
                  <div className="row gap">
                    <label className="file">
                      <input type="file" accept=".cif" onChange={handleFileChange} />
                      <span>{cifFile ? cifFile.name : "Choose .cif file"}</span>
                    </label>
                    {/* Do NOT disable on auth; guard inside handler */}
                    <button className="primary" onClick={handleUpload} disabled={isLoading || !cifFile}>
                      {isLoading ? "Predicting…" : "Predict Properties"}
                    </button>
                  </div>
                </div>

                {error && <div className="alert error">{error}</div>}

                {predictions && (
                  <div className="grid two">
                    {/* Table */}
                    <div className="panel">
                      <h3>Results {cifFile ? `· ${cifFile.name}` : ""}</h3>

                      {/* ⭐ Favorite for predict/ */}
                      <div style={{ margin: "6px 0 12px" }}>
                        <FavoriteButton
                          filename={cifFile ? cifFile.name : ""}
                          favorites={favApi.items}
                          onAdd={favApi.add}
                          onRemove={favApi.remove}
                          isAuthed={isAuthed}
                          requireAuth={requireAuth}
                        />
                      </div>

                      {/* Exports */}
                      <div className="row gap" style={{ margin: "4px 0 12px" }}>
                        <button
                          className="chip"
                          onClick={async () => {
                            if (!requireAuth()) return;
                            const safe = (cifFile?.name || "mof_result").replace(/[^\w.-]/g, "_");
                            const { data } = await api.post(
                              "/export/pdf",
                              { filename: safe, prediction: predictions, adsorption_plot: adsorptionPlot },
                              { responseType: "blob" }
                            );
                            const blob = data instanceof Blob ? data : new Blob([data], { type: "application/pdf" });
                            downloadFile(blob, `${safe}.pdf`);
                          }}
                        >
                          Download PDF
                        </button>

                        <button
                          className="chip"
                          onClick={async () => {
                            if (!requireAuth()) return;
                            const safe = (cifFile?.name || "mof_result").replace(/[^\w.-]/g, "_");
                            const { data } = await api.post(
                              "/export/html",
                              { filename: safe, prediction: predictions, adsorption_plot: adsorptionPlot },
                              { responseType: "blob" }
                            );
                            const blob = data instanceof Blob ? data : new Blob([data], { type: "text/html" });
                            downloadFile(blob, `${safe}.html`);
                          }}
                        >
                          Download HTML
                        </button>

                        <button
                          className="chip"
                          onClick={async () => {
                            if (!requireAuth()) return;
                            const safe = (cifFile?.name || "mof_result").replace(/[^\w.-]/g, "_");
                            const { data } = await api.post(
                              "/export/json",
                              { filename: safe, prediction: predictions, adsorption_plot: adsorptionPlot },
                              { responseType: "blob" }
                            );
                            const blob = data instanceof Blob ? data : new Blob([data], { type: "application/json" });
                            downloadFile(blob, `${safe}.json`);
                          }}
                        >
                          Download JSON
                        </button>
                      </div>

                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Property</th>
                              <th>Predicted Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(predictions).map(([k, v]) => (
                              <tr key={k}>
                                <td>{k}</td>
                                <td>{Number(v).toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Chart */}
                    <div className="panel">
                      <h3>Visualization</h3>
                      <PredictionChart predictions={predictions} />
                      {adsorptionPlot && (
                        <>
                          <h4 style={{ marginTop: 16 }}>CO₂ Adsorption Plot</h4>
                          <img className="plot-img" src={`data:image/png;base64,${adsorptionPlot}`} alt="CO2 Adsorption Plot" />
                        </>
                      )}
                    </div>
                  </div>
                )}
              </section>

              {/* Image Search Card */}
              <section className="card">
                <div className="card-head">
                  <h2>🖼️ Image-Based MOF Search (Gemini Vision)</h2>
                  <div className="row gap">
                    <label className="file">
                      <input type="file" accept="image/*" onChange={handleImageFileChange} />
                      <span>{imageFile ? imageFile.name : "Choose an image"}</span>
                    </label>
                    {/* Don't disable on auth; guard in handler */}
                    <button className="secondary" onClick={handleImageUpload} disabled={isImageLoading || !imageFile}>
                      {isImageLoading ? "Analyzing…" : "Search by Image"}
                    </button>
                  </div>
                </div>

                {imageError && <div className="alert error">{imageError}</div>}
                {imagePredictionResponse && (
                  <div className="panel">
                    <h3>Image Analysis Result</h3>
                    <pre className="pre">{imagePredictionResponse}</pre>

                    {/* ⭐ Favorite for image_search/ (edit suggestion or click star to enter) */}
                    <div className="row gap" style={{ marginTop: 8 }}>
                      <input
                        className="flex-grow"
                        type="text"
                        placeholder="MOF filename to favorite (e.g., ABAYIO_clean.cif)"
                        value={imageFavName}
                        onChange={(e) => setImageFavName(e.target.value)}
                      />
                      <FavoriteButton
                        filename={imageFavName}
                        favorites={favApi.items}
                        onAdd={favApi.add}
                        onRemove={favApi.remove}
                        isAuthed={isAuthed}
                        requireAuth={requireAuth}
                      />
                    </div>
                    <div className="muted" style={{ marginTop: 4 }}>
                      Tip: we tried to guess a MOF id from the text — edit if needed or just click ☆ to type one.
                    </div>
                  </div>
                )}
              </section>

              {/* Consolidated DB Search Section */}
              <section className="card">
                <div className="card-head">
                  <h2>🔍 MOF Database Search</h2>
                  <p>Search by filename or by any combination of properties.</p>
                </div>
                <div className="row gap-md">
                  <select
                    value={searchKey}
                    onChange={(e) => {
                      setSearchKey(e.target.value);
                      setSearchValue("");
                    }}
                    className="select-style"
                  >
                    <option value="filename">Filename (Keyword)</option>
                    <option value="LCD">LCD</option>
                    <option value="PLD">PLD</option>
                    <option value="LFPD">LFPD</option>
                    <option value="cm3_g">Volume (cm³/g)</option>
                    <option value="ASA_m2_cm3">ASA (m²/cm³)</option>
                    <option value="ASA_m2_g">ASA (m²/g)</option>
                    <option value="AV_VF">AV (Volumetric Fraction)</option>
                    <option value="AV_cm3_g">AV (cm³/g)</option>
                    <option value="Has_OMS">Has OMS (0 or 1)</option>
                  </select>
                  <input
                    type={searchKey === "filename" ? "text" : "number"}
                    placeholder={`Enter value for ${searchKey}`}
                    value={searchValue}
                    onChange={(e) => setSearchValue(e.target.value)}
                    className="flex-grow"
                  />
                  {/* Don't disable on auth; guard in handler */}
                  <button onClick={handleSearch} disabled={isSearching} className="primary">
                    {isSearching ? "Searching..." : "Search"}
                  </button>
                </div>

                {searchError && <div className="alert error">{searchError}</div>}

                {searchResults && (
                  <div className="panel mt-4">
                    <h3>Search Results ({searchResults.length} found)</h3>
                    {searchResults.length > 0 ? (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Filename & Actions</th>
                              <th>LCD</th>
                              <th>PLD</th>
                              <th>LFPD</th>
                              <th>cm3_g</th>
                              <th>ASA_m2_cm3</th>
                              <th>ASA_m2_g</th>
                              <th>NASA_m2_cm3</th>
                              <th>NASA_m2_g</th>
                              <th>AV_VF</th>
                              <th>AV_cm3_g</th>
                              <th>NAV_cm3_g</th>
                              <th>Has_OMS</th>
                            </tr>
                          </thead>
                          <tbody>
                            {searchResults.map((mof) => (
                              <tr key={mof.filename} className="mof-result-row">
                                <td>
                                  <b>{mof.filename}</b>
                                  <div style={{ marginTop: 6 }}>
                                    <FavoriteButton
                                      filename={mof.filename}
                                      favorites={favApi.items}
                                      onAdd={favApi.add}
                                      onRemove={favApi.remove}
                                      isAuthed={isAuthed}
                                      requireAuth={requireAuth}
                                    />
                                    <button className="chip" onClick={() => fetchMofDetailsAndPlot(mof.filename)}>
                                      View
                                    </button>
                                  </div>
                                </td>
                                <td>{mof.LCD?.toFixed(4) || "N/A"}</td>
                                <td>{mof.PLD?.toFixed(4) || "N/A"}</td>
                                <td>{mof.LFPD?.toFixed(4) || "N/A"}</td>
                                <td>{mof.cm3_g?.toFixed(4) || "N/A"}</td>
                                <td>{mof.ASA_m2_cm3?.toFixed(4) || "N/A"}</td>
                                <td>{mof.ASA_m2_g?.toFixed(4) || "N/A"}</td>
                                <td>{mof.NASA_m2_cm3?.toFixed(4) || "N/A"}</td>
                                <td>{mof.NASA_m2_g?.toFixed(4) || "N/A"}</td>
                                <td>{mof.AV_VF?.toFixed(4) || "N/A"}</td>
                                <td>{mof.AV_cm3_g?.toFixed(4) || "N/A"}</td>
                                <td>{mof.NAV_cm3_g?.toFixed(4) || "N/A"}</td>
                                <td>{mof.Has_OMS !== undefined ? mof.Has_OMS : "N/A"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p>No results found for your query. Try a different search term or criteria.</p>
                    )}
                  </div>
                )}
              </section>

              {/* Selected MOF details */}
              {selectedMofDetails && (
                <section className="card">
                  <div className="card-head">
                    <h2>📊 Details for {selectedMofDetails.filename}</h2>
                    <div className="row gap">
                      <FavoriteButton
                        filename={selectedMofDetails.filename}
                        favorites={favApi.items}
                        onAdd={favApi.add}
                        onRemove={favApi.remove}
                        isAuthed={isAuthed}
                        requireAuth={requireAuth}
                      />
                    </div>
                  </div>
                  {isMofDetailLoading && <p>Loading MOF details and plot...</p>}
                  {mofDetailError && <div className="alert error">{mofDetailError}</div>}
                  {!isMofDetailLoading && !mofDetailError && (
                    <div className="grid two">
                      <div className="panel">
                        <h3>Properties</h3>
                        <div className="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Property</th>
                                <th>Value</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(selectedMofDetails).map(([key, value]) => (
                                <tr key={key}>
                                  <td>{key}</td>
                                  <td>
                                    {typeof value === "number"
                                      ? value.toFixed(4)
                                      : value !== null && value !== undefined
                                      ? value.toString()
                                      : "N/A"}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      <div className="panel">
                        <h3>CO₂ Adsorption Plot</h3>
                        {selectedMofPlot ? (
                          <img
                            className="plot-img"
                            src={`data:image/png;base64,${selectedMofPlot}`}
                            alt={`CO2 Adsorption Plot for ${selectedMofDetails.filename}`}
                          />
                        ) : (
                          <p>No adsorption plot available for this MOF.</p>
                        )}
                      </div>
                    </div>
                  )}
                </section>
              )}

              {/* Recommendation */}
              <section className="card">
                <div className="card-head">
                  <h2>🧠 Material Recommendation</h2>
                </div>
                <div className="recommendation-form">
                  <textarea
                    placeholder="Describe your requirements (e.g., 'a MOF with high selectivity for CO2 at low pressure')"
                    value={recommendationQuery}
                    onChange={(e) => setRecommendationQuery(e.target.value)}
                  />
                  <div className="row gap">
                    <input
                      type="number"
                      placeholder="Min ASA (m²/g)"
                      value={minAsa}
                      onChange={(e) => setMinAsa(e.target.value)}
                    />
                    <input
                      type="number"
                      placeholder="Min PLD (Å)"
                      value={minPld}
                      onChange={(e) => setMinPld(e.target.value)}
                    />
                  </div>
                  {/* Don't disable on auth; guard in handler */}
                  <button onClick={handleRecommend} disabled={isRecommending}>
                    {isRecommending ? "Recommending..." : "Get Recommendation"}
                  </button>
                </div>
                {recommendationError && <div className="alert error">{recommendationError}</div>}
                {recommendation && (
                  <div className="panel">
                    <h3>AI-Powered Recommendation</h3>
                    <p className="pre">{recommendation}</p>
                  </div>
                )}
              </section>
            </>
          )}

          {/* Favorites */}
          {activeNav === "Favorites" && (
            <section className="card" style={{ margin: 16 }}>
              <div className="card-head">
                <h2>⭐ Favorites</h2>
              </div>
              {favApi.loading && <p>Loading favorites...</p>}
              {favApi.err && <div className="alert error">{favApi.err}</div>}
              {!isAuthed ? (
                <p className="muted">Log in to view your favorites.</p>
              ) : favApi.items.length === 0 && !favApi.loading ? (
                <p className="muted">No favorites yet. Add some from search results.</p>
              ) : (
                <>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>MOF</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {favApi.items.map((f) => (
                          <tr key={f.id || f.filename}>
                            <td>
                              <b>{f.filename || f.id}</b>
                            </td>
                            <td>
                              <div className="row gap">
                                <button
                                  className="chip danger"
                                  onClick={() => favApi.remove(f.id || f.filename)}
                                >
                                  Remove
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {favApi.hasMore && (
                    <div className="mt-2">
                      <button className="secondary" onClick={() => favApi.fetchPage({ cursor: favApi.nextCursor })}>
                        Load more
                      </button>
                    </div>
                  )}
                </>
              )}
            </section>
          )}

          {/* History */}
          {activeNav === "History" && (
            <section className="card" style={{ margin: 16 }}>
              <div className="card-head">
                <h2>🕘 Recent Activity</h2>
                <div className="row gap">
                  <button
                    className="secondary"
                    onClick={() => {
                      historyApi.reset();
                      historyApi.fetchPage({ limit: 20 });
                    }}
                  >
                    Refresh
                  </button>
                </div>
              </div>

              {historyApi.loading && historyApi.items.length === 0 && <p>Loading history...</p>}
              {historyApi.err && <div className="alert error">{historyApi.err}</div>}

              {!isAuthed ? (
                <p className="muted">Log in to view your activity history.</p>
              ) : historyApi.items.length === 0 && !historyApi.loading ? (
                <p className="muted">No recent activity.</p>
              ) : (
                <>
                  <ul className="history-list">
                    {historyApi.items.map((it) => (
                      <li key={it.id}>
                        <strong>{it.type}</strong>
                        <span className="timestamp">
                          {it.ts_iso ? new Date(it.ts_iso).toLocaleString() : "—"}
                        </span>
                        {it.details && (
                          <span className="details">
                            {Object.entries(it.details).map(([k, v]) => (
                              <span key={k}>
                                {" "}
                                • {k}: {String(v).slice(0, 25)}
                                {String(v).length > 25 ? "…" : ""}
                              </span>
                            ))}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>

                  {historyApi.hasMore && (
                    <div className="mt-2">
                      <button
                        className="secondary"
                        onClick={() => historyApi.fetchPage({ cursorTs: historyApi.nextCursorTs })}
                      >
                        Load more
                      </button>
                    </div>
                  )}
                </>
              )}
            </section>
          )}
        </main>

        {/* Right Info Panel */}
        <aside className={`rightbar ${rightOpen ? "open" : "closed"}`}>
          <div className="rightbar-head">
            <h3>Profile & Insights</h3>
            <button className="icon-btn" onClick={() => setRightOpen(!rightOpen)} title="Collapse panel">
              {rightOpen ? "→" : "←"}
            </button>
          </div>
          {rightOpen && (
            <div className="rightbar-body">
              <div className="panel">
                <h4>Signed in {isAuthed && userId ? `(ID: ${String(userId).slice(0, 6)}...)` : ""}</h4>
              </div>

              <div className="panel">
                <h4>Quick Stats</h4>
                <ul className="stat-list">
                  <li>
                    Predictions: <b>{predictions ? 1 : 0}</b>
                  </li>
                  <li>
                    RAG Queries: <b>—</b>
                  </li>
                  <li>
                    Recommendations: <b>{recommendation ? 1 : 0}</b>
                  </li>
                </ul>
              </div>

              <div className="panel">
                <h4>Shortcuts</h4>
                <div className="chip-row">
                  <button className="chip" onClick={() => setActiveNav("Overview")}>
                    Upload .cif
                  </button>
                  <button className="chip" onClick={() => setActiveNav("Overview")}>
                    Ask RAG
                  </button>
                  <button className="chip" onClick={() => handleNavClick("Favorites")}>
                    Favorites
                  </button>
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>

      {/* Floating Gemini Chat */}
      <GeminiChat api={chatApi} />
    </div>
  );
}

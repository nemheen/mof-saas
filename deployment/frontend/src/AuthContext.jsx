// mof-dashboard/src/AuthContext.jsx
import React, { createContext, useContext, useState, useEffect, useMemo } from "react";
import axios from "axios";

// ---- (Optional) Firebase client SDK (only for Firestore/UI needs) ----
import { initializeApp } from "firebase/app";
import {
  getAuth,
  signInWithCustomToken,
  onAuthStateChanged,
} from "firebase/auth";
import { getFirestore, serverTimestamp } from "firebase/firestore";

// -------- Config & helpers --------
const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";
const ACCESS_TOKEN_KEY = "access_token";

const AuthContext = createContext(null);
export const useAuth = () => useContext(AuthContext);

// Parse the Firebase Web App config (NOT service account)
const firebaseConfigString = process.env.REACT_APP_FIREBASE_CONFIG;
let firebaseConfig = {};
try {
  if (firebaseConfigString) firebaseConfig = JSON.parse(firebaseConfigString);
} catch (e) {
  console.error("Error parsing REACT_APP_FIREBASE_CONFIG:", e);
  console.log("Config string received:", firebaseConfigString);
}

function validateFirebaseConfig(cfg) {
  const required = ["apiKey", "authDomain", "projectId", "appId"];
  const missing = required.filter((k) => !cfg?.[k] || String(cfg[k]).trim() === "");
  return { ok: missing.length === 0, missing };
}

function isAuthEndpoint(url = "") {
  // Works for relative or absolute paths stored in axios config
  return ["/auth/login", "/auth/signup", "/auth/refresh", "/auth/logout", "/auth/me"].some((p) =>
    (url || "").endsWith(p)
  );
}

// -------- Provider --------
export function AuthProvider({ children }) {
  // Persisted access token for Authorization header
  const [accessToken, setAccessToken] = useState(() => localStorage.getItem(ACCESS_TOKEN_KEY) || "");
  // User identity for UI (email/id). We derive this from /auth/me or Firebase (if you use it).
  const [userId, setUserId] = useState(null);
  // Ready flag so the app can render after we try optional Firebase init
  const [isAuthReady, setIsAuthReady] = useState(false);
  // Optional Firestore handle for other parts of your UI (not required for auth)
  const [db, setDb] = useState(null);

  const appId = firebaseConfig.appId || process.env.REACT_APP_APP_ID || "default-local-app-id";
  // Simple guard that shows a friendly message before making API calls
  const requireAuth = React.useCallback(
    (msg = "You need to log in or sign up first.") => {
    if (!accessToken) {
      alert(msg);
      return false;
}
    return true;
    },
    [accessToken]
);

  const withAuth = React.useCallback(
   (fn, msg) => (...args) => {
   if (!requireAuth(msg)) return;
  return fn?.(...args);
    },
    [requireAuth]
  );

  // ---- Optional Firebase init (no anonymous sign-in) ----
  useEffect(() => {
    const initialAuthToken = process.env.REACT_APP_INITIAL_AUTH_TOKEN || null;

    if (!firebaseConfigString) {
      // Firebase not configured — skip cleanly
      setIsAuthReady(true);
      return;
    }

    const { ok, missing } = validateFirebaseConfig(firebaseConfig);
    if (!ok) {
      console.error("Firebase config is incomplete. Missing:", missing);
      setIsAuthReady(true);
      return;
    }

    const app = initializeApp(firebaseConfig);
    const auth = getAuth(app);
    const firestore = getFirestore(app);
    setDb(firestore);

    // Listen for Firebase user *only* if you use custom token (no anonymous fallback)
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      try {
        if (user) {
          setUserId(String(user.uid));
        } else if (initialAuthToken) {
          // Optional: sign in with a provided custom token (if you have one)
          try {
            await signInWithCustomToken(auth, initialAuthToken);
          } catch (e) {
            console.error("Custom-token sign-in failed:", e);
          }
        } else {
          // No Firebase auth — that's OK; we use FastAPI JWT for backend auth.
          setUserId((prev) => prev); // no-op
        }
      } finally {
        setIsAuthReady(true);
      }
    });

    return () => unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once

  // ---- Axios instance with token + refresh & retry ----
  const api = useMemo(() => {
    const instance = axios.create({
      baseURL: API_BASE,
      withCredentials: true, // send/receive refresh cookie
    });

    // Attach Authorization header if we have an access token
    instance.interceptors.request.use((config) => {
      if (accessToken) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${accessToken}`;
      }
      return config;
    });

    // Auto-refresh on 401 (once), then retry original request
    instance.interceptors.response.use(
      (r) => r,
      async (err) => {
        const originalRequest = err.config || {};
    
        // If the backend says 401 and the user has NO token at all,
        // tell them to log in instead of attempting /auth/refresh.
        if (
          err.response?.status === 401 &&
          !isAuthEndpoint(originalRequest?.url)
        ) {
          if (!accessToken) {
            alert("You need to log in or sign up first.");
            return Promise.reject(err);
          }
        }
    
        // Token exists but request failed with 401 => try refresh once
        if (
          err.response?.status === 401 &&
          !originalRequest._retry &&
          !isAuthEndpoint(originalRequest?.url)
        ) {
          originalRequest._retry = true;
          try {
            const { data } = await axios.post(
              `${API_BASE}/auth/refresh`,
              {},
              { withCredentials: true }
            );
            setAccessToken(data.access_token);
            localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
            originalRequest.headers = {
              ...(originalRequest.headers || {}),
              Authorization: `Bearer ${data.access_token}`,
            };
            return instance(originalRequest);
          } catch (e) {
            setAccessToken("");
            localStorage.removeItem(ACCESS_TOKEN_KEY);
            alert("Your session expired. Please log in again.");
          }
        }
    
        return Promise.reject(err);
      }
    );
    

    return instance;
  }, [accessToken]);

  // ---- Derive user identity from backend (/auth/me) when we have a token ----
  useEffect(() => {
    let cancelled = false;
    async function fetchMe() {
      if (!accessToken) {
        setUserId(null);
        return;
      }
      try {
        const { data } = await api.get("/auth/me");
        // Prefer numeric id, fall back to email or sub
        const raw = data?.id ?? data?.email ?? data?.sub ?? null;
        const uid = raw == null ? null : String(raw);
        if (!cancelled) setUserId(uid);
      } catch (e) {
        // If /auth/me fails (expired token), the interceptor will try refresh on the next request.
        if (!cancelled) setUserId(null);
      }
    }
    fetchMe();
    return () => {
      cancelled = true;
    };
  }, [accessToken, api]);

  // ------- Auth actions (FastAPI) -------
  const login = async (email, password) => {
    try {
      const { data } = await api.post("/auth/login", { email, password });
      if (data?.access_token) {
        localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
        setAccessToken(data.access_token);
      }
      return true;
    } catch (err) {
      console.error("Login failed:", err?.response?.data?.detail || err.message);
      alert(err?.response?.data?.detail || "Login failed.");
      return false;
    }
  };

  const signup = async (email, password) => {
    try {
      await api.post("/auth/signup", { email, password });
      await login(email, password);
      alert("Account created successfully! You are now logged in.");
      return true;
    } catch (err) {
      console.error("Signup failed:", err?.response?.data?.detail || err.message);
      alert(err?.response?.data?.detail || "Signup failed.");
      return false;
    }
  };

  const logout = async () => {
    try {
      await api.post("/auth/logout"); // clears refresh cookie server-side
    } catch (err) {
      console.error("Backend logout failed (may already be logged out):", err);
    } finally {
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      setAccessToken("");
      setUserId(null);
      alert("Logged out successfully.");
    }
  };

  const value = useMemo(
    () => ({
      api,
      db, // optional Firestore instance if you still need it elsewhere
      userId,
      isAuthed: !!accessToken,
      isAuthReady,
      login,
      signup,
      logout,
      requireAuth,
      withAuth,
      serverTimestamp, // exported for any remaining Firestore writes
      appId,
    }),
    [api, db, userId, accessToken, isAuthReady, appId]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

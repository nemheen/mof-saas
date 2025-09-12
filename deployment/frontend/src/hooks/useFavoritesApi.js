// src/hooks/useFavoritesApi.js
import { useState, useCallback } from "react";

/**
 * Backend-driven favorites with docId cursor pagination.
 */
export default function useFavoritesApi(api) {
  const [items, setItems] = useState([]);
  const [nextCursor, setNextCursor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const reset = useCallback(() => {
    setItems([]);
    setNextCursor(null);
    setErr("");
  }, []);

  const fetchPage = useCallback(async ({ limit = 50, cursor = null } = {}) => {
    if (!api) return;
    setLoading(true);
    setErr("");
    try {
      const params = { limit };
      if (cursor) params.cursor = cursor;
      const { data } = await api.get("/user/favorites", { params });
      const newItems = Array.isArray(data.items) ? data.items : data?.items ?? [];
      setItems((prev) => (cursor ? [...prev, ...newItems] : newItems));
      setNextCursor(data.next_cursor ?? null);
    } catch (e) {
      console.error(e);
      setErr("Failed to load favorites.");
    } finally {
      setLoading(false);
    }
  }, [api]);

  const add = useCallback(async (mofId, filename) => {
    await api.post(`/user/favorites/${encodeURIComponent(mofId)}`, null, {
      params: filename ? { filename } : {},
    });
    // optimistic: prepend new favorite
    setItems((prev) => [{ id: mofId, filename: filename || mofId, added_at: new Date().toISOString() }, ...prev]);
  }, [api]);

  const remove = useCallback(async (mofId) => {
    await api.delete(`/user/favorites/${encodeURIComponent(mofId)}`);
    setItems((prev) => prev.filter((x) => x.id !== mofId));
  }, [api]);

  return { items, nextCursor, hasMore: !!nextCursor, loading, err, fetchPage, reset, add, remove };
}

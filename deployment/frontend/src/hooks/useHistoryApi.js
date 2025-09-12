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
      // Supports:
      //   1) { items: [...], next_cursor_ts: "..." }
      //   2) [...plainItems]
      const arr = Array.isArray(data) ? data : Array.isArray(data?.items) ? data.items : [];
      const cursor = Array.isArray(data) ? null : (data?.next_cursor_ts ?? null);
  
      // Be robust to different timestamp field names: ts_iso, ts, timestamp
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
  
          // Debug log (helps diagnose backend shape quickly)
          // eslint-disable-next-line no-console
          console.debug("GET /user/history ->", data);
  
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
  
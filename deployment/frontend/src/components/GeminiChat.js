import React, { useState } from 'react';
import './GeminiChat.css';

// Accept the authenticated axios instance from parent (App.js passes `api`)
export default function GeminiChat({ api }) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const toggleChat = () => setIsOpen(!isOpen);

  const handleSend = async () => {
    if (!query.trim()) return;
    if (!api) {
      setResponse("Not ready: API client missing.");
      return;
    }
    setResponse("Thinking…");
    try {
      // Use the authenticated axios instance (has Authorization + withCredentials)
      const res = await api.post("/rag_query", { query });
      setResponse(res?.data?.response ?? "No answer returned.");
    } catch (err) {
      console.error(err);
      const msg = err?.response?.data?.detail || err?.message || "Unknown error";
      setResponse(`Error: ${msg}`);
    }
  };

  return (
    <div className="chat-wrapper">
      <button className="chat-toggle" onClick={toggleChat}>💬</button>
      {isOpen && (
        <div className="chat-box">
          <div className="chat-header">Ask Gemini</div>
          <textarea
            className="chat-input"
            placeholder="Ask about MOFs, synthesis, etc..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button className="chat-send" onClick={handleSend}>Send</button>
          <div className="chat-response">{response}</div>
        </div>
      )}
    </div>
  );
}

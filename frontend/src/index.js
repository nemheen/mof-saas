// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import './index.css';
// import App from './App';
// import reportWebVitals from './reportWebVitals';

// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );

// // If you want to start measuring performance in your app, pass a function
// // to log results (for example: reportWebVitals(console.log))
// // or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();

// src/index.js (or src/main.jsx)
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Your global CSS, if any
import App from './App'; // Import your main App component
import { AuthProvider } from './AuthContext'; // Import your AuthProvider

// Create a root to render your React app into the DOM
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render your App component, wrapped with AuthProvider for context
root.render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>
);

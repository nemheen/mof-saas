// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

const firebaseConfig = {
    apiKey: "AIzaSyC9FgwP-7nzbfFzntOuRVD4rBO54p7GolU",
    authDomain: "mof-firebase-b6b3b.firebaseapp.com",
    projectId: "mof-firebase-b6b3b",
    storageBucket: "mof-firebase-b6b3b.firebasestorage.app",
    messagingSenderId: "976119423731",
    appId: "1:976119423731:web:e0807f2b54a320534e4d86",
    measurementId: "G-L4X0MN6WD5" 
  };
  
  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);



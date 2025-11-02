// Cargá tu config real de Firebase (Project settings -> General -> Your apps)
const firebaseConfig = {
  apiKey: "AIzaSyCHc6uy6uc1Jr6bzHQYGUZi2uZvTX0S9fE",
  authDomain: "apuntesya-d7d72.firebaseapp.com",
  projectId: "apuntesya-d7d72",
  storageBucket: "apuntesya-d7d72.firebasestorage.app",
  messagingSenderId: "332327927567",
  appId: "1:332327927567:web:22ecbb47817c2b7c71487a",
  measurementId: "G-9MBP39X788"
};

// Inicializa Firebase
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, signInWithPopup, GoogleAuthProvider, signOut, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

async function backendSessionLogin(idToken) {
  const res = await fetch("/auth/session_login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_token: idToken })
  });
  if (!res.ok) throw new Error("No se pudo crear la sesión en el servidor");
  return res.json();
}

window.googleSignIn = async function googleSignIn() {
  try {
    const result = await signInWithPopup(auth, provider);
    const idToken = await result.user.getIdToken();
    await backendSessionLogin(idToken);
    window.location.href = "/";
  } catch (e) {
    console.error(e);
    alert("Error al iniciar sesión con Google.");
  }
};

window.googleLogout = async function googleLogout() {
  try {
    await signOut(auth);
    window.location.href = "/logout";
  } catch (e) {
    console.error(e);
    window.location.href = "/logout";
  }
};

// (opcional) Mostrar el botón de logout si hay sesión de Firebase
onAuthStateChanged(auth, (user) => {
  const btn = document.getElementById("googleLogoutBtn");
  if (!btn) return;
  btn.style.display = user ? "inline-block" : "none";
});

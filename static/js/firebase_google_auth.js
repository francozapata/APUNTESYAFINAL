// Cargá tu config real de Firebase
const firebaseConfig = window.FIREBASE_WEB_CONFIG;

// Imports
import { initializeApp, getApps, getApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import {
  getAuth,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult,
  GoogleAuthProvider,
  signOut,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Inicializar app sin duplicarla
const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
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

// 🔵 PROCESAR REDIRECT (si venís de incógnito o popup bloqueado)
getRedirectResult(auth)
  .then(async (result) => {
    if (result?.user) {
      const idToken = await result.user.getIdToken();
      await backendSessionLogin(idToken);
      window.location.href = "/";
    }
  })
  .catch((e) => {
    console.warn("Redirect result:", e?.code, e?.message);
  });


// 🔵 LOGIN GOOGLE (Popup + Fallback Redirect)
window.googleSignIn = async function googleSignIn() {
  try {
    const result = await signInWithPopup(auth, provider);
    const idToken = await result.user.getIdToken();
    await backendSessionLogin(idToken);
    window.location.href = "/";
  } catch (e) {
    const code = e?.code || "";

    console.warn("Popup falló:", code);

    // Fallback típico para incógnito
    if (
      code === "auth/popup-blocked" ||
      code === "auth/popup-closed-by-user" ||
      code === "auth/web-storage-unsupported" ||
      code === "auth/operation-not-supported-in-this-environment"
    ) {
      await signInWithRedirect(auth, provider);
      return;
    }

    console.error(e);
    alert("Error al iniciar sesión con Google.");
  }
};


// 🔵 LOGOUT
window.googleLogout = async function googleLogout() {
  try {
    await signOut(auth);
    window.location.href = "/logout";
  } catch (e) {
    console.error(e);
    window.location.href = "/logout";
  }
};


// 🔵 Mostrar botón logout si hay sesión
onAuthStateChanged(auth, (user) => {
  const btn = document.getElementById("googleLogoutBtn");
  if (!btn) return;
  btn.style.display = user ? "inline-block" : "none";
});
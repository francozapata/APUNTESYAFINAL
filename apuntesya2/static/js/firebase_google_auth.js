// ==== Firebase Google Auth (v11) ====

// Config inyectada por el backend
const firebaseConfig = window.FIREBASE_WEB_CONFIG;
const missing = Array.isArray(window.FIREBASE_WEB_MISSING) ? window.FIREBASE_WEB_MISSING : [];

// SDK imports
import { initializeApp, getApps, getApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import {
    getAuth,
    GoogleAuthProvider,
    signInWithPopup,
    signInWithRedirect,
    getRedirectResult,
    onAuthStateChanged,
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Firebase init (evitar doble init)
let app = null;
let auth = null;
let provider = null;

if (!firebaseConfig || typeof firebaseConfig !== "object" || missing.length) {
    console.warn("Firebase Web SDK config incompleta. Missing:", missing);
} else {
    app = getApps().length ? getApp() : initializeApp(firebaseConfig);
    auth = getAuth(app);
    provider = new GoogleAuthProvider();
}

// Backend session: crea cookie de sesión y devuelve next
async function backendSessionLogin(idToken) {
    const res = await fetch("/auth/session_login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ id_token: idToken }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
        throw new Error(data?.error || `HTTP ${res.status}`);
    }

    window.location.href = data.next || "/";
}

// Procesar resultado al volver de redirect (UNA sola vez)
async function handleRedirectResultOnce() {
    if (!auth) return;
    try {
        const result = await getRedirectResult(auth);
        if (result?.user) {
            const idToken = await result.user.getIdToken(true);
            await backendSessionLogin(idToken);
        }
    } catch (e) {
        // Si no hay redirect pendiente, normalmente tira null o errores no críticos.
        console.warn("getRedirectResult:", e?.code || "", e?.message || e);
    }
}
handleRedirectResultOnce();

// Login con Google: primero popup, si falla -> redirect (ideal para incógnito)
async function doGoogleSignIn() {
    if (!auth || !provider) {
        alert(
            "Login con Google no está configurado todavía.\n" +
            "Faltan variables de Firebase en el servidor (Render)."
        );
        return;
    }

    try {
        // Intento 1: popup
        const result = await signInWithPopup(auth, provider);
        const idToken = await result.user.getIdToken(true);
        await backendSessionLogin(idToken);
    } catch (e) {
        const code = e?.code || "";
        console.warn("signInWithPopup falló:", code, e?.message || e);

        // Fallback típico (incógnito / bloqueos de popup / storage)
        if (
            code === "auth/popup-blocked" ||
            code === "auth/popup-closed-by-user" ||
            code === "auth/web-storage-unsupported" ||
            code === "auth/operation-not-supported-in-this-environment"
        ) {
            await signInWithRedirect(auth, provider);
            return;
        }

        alert("Error al iniciar sesión con Google.\n" + (code || e?.message || ""));
    }
}

// Hook UI
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("btnGoogle");
    if (btn) btn.addEventListener("click", doGoogleSignIn);
});

// (opcional) mostrar/ocultar logout si existe en alguna vista
if (auth) {
    onAuthStateChanged(auth, (user) => {
        const logoutBtn = document.getElementById("googleLogoutBtn");
        if (logoutBtn) logoutBtn.style.display = user ? "inline-block" : "none";
    });
}

export { };
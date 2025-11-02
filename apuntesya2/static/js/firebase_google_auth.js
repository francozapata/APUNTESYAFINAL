// ==== Firebase Google Auth (v11) ====
// PONÉ TU CONFIG REAL:
const firebaseConfig = {
    apiKey: "AIzaSyCHc6uy6uc1Jr6bzHQYGUZi2uZvTX0S9fE",
    authDomain: "apuntesya-d7d72.firebaseapp.com",
    projectId: "apuntesya-d7d72",
    storageBucket: "apuntesya-d7d72.firebasestorage.app",
    messagingSenderId: "332327927567",
    appId: "1:332327927567:web:22ecbb47817c2b7c71487a",
    measurementId: "G-9MBP39X788"
};

// SDK imports
import {
    initializeApp, getApps, getApp
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import {
    getAuth,
    GoogleAuthProvider,
    signInWithPopup,
    signInWithRedirect,
    getRedirectResult,
    signOut,
    onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Evitar doble inicialización
const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Back-end session (envía el ID token a Flask)
async function backendSessionLogin(idToken) {
    const res = await fetch("/auth/session_login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: idToken })
    });
    if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error("Servidor rechazó el token: " + txt);
    }
    return res.json();
}

// Login con popup (fallback a redirect si se bloquea/cierra)
async function doGoogleSignIn() {
    try {
        const result = await signInWithPopup(auth, provider);
        const idToken = await result.user.getIdToken();
        await backendSessionLogin(idToken);
        window.location.href = "/";
    } catch (e) {
        console.error("Popup error:", e);
        if (e?.code === "auth/popup-blocked" || e?.code === "auth/popup-closed-by-user") {
            try {
                await signInWithRedirect(auth, provider);
                return;
            } catch (e2) {
                console.error("Redirect error:", e2);
            }
        }
        alert("Error al iniciar sesión con Google.\n" + (e?.code || e?.message || ""));
    }
}

// Procesar resultado al volver de redirect
(async () => {
    try {
        const result = await getRedirectResult(auth);
        if (result?.user) {
            const idToken = await result.user.getIdToken();
            await backendSessionLogin(idToken);
            window.location.href = "/";
        }
    } catch (e) {
        console.warn("Redirect result error:", e);
    }
})();

// Hook UI
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("btnGoogle");
    if (btn) btn.addEventListener("click", doGoogleSignIn);
});

// (opcional) Mostrar/ocultar logout si lo agregás en alguna vista
onAuthStateChanged(auth, (user) => {
    const logoutBtn = document.getElementById("googleLogoutBtn");
    if (logoutBtn) logoutBtn.style.display = user ? "inline-block" : "none";
});

export { };

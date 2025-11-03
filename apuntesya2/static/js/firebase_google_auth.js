// ==== Firebase Google Auth (v11) ====

// Tus claves del SDK Web (están bien estas):
const firebaseConfig = {
    apiKey: "AIzaSyCHc6uy6uc1Jr6bzHQYGUZi2uZvTX0S9fE",
    authDomain: "apuntesya-d7d72.firebaseapp.com",
    projectId: "apuntesya-d7d72",
    storageBucket: "apuntesya-d7d72.firebasestorage.app",
    messagingSenderId: "332327927567",
    appId: "1:332327927567:web:22ecbb47817c2b7c71487a",
    measurementId: "G-9MBP39X788",
};

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

// Evitar doble init
const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Llama al backend y que el backend decida el next
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
    // redirige según indique el backend
    window.location.href = data.next || "/";
}

// Login con popup (y fallback a redirect)
async function doGoogleSignIn() {
    try {
        const result = await signInWithPopup(auth, provider);
        const idToken = await result.user.getIdToken(/* forceRefresh */ true);
        await backendSessionLogin(idToken);
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
            const idToken = await result.user.getIdToken(/* forceRefresh */ true);
            await backendSessionLogin(idToken);
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

// (opcional) mostrar/ocultar logout si lo agregás en alguna vista
onAuthStateChanged(auth, (user) => {
    const logoutBtn = document.getElementById("googleLogoutBtn");
    if (logoutBtn) logoutBtn.style.display = user ? "inline-block" : "none";
});

export { };

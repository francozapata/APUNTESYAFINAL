// ==== Firebase Google Auth (v11) ====

// Tus claves del SDK Web (están bien estas):
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

// Si falta config, no intentamos inicializar Firebase (evita errores en consola)
let app = null;
let auth = null;
let provider = null;

if (!firebaseConfig || typeof firebaseConfig !== "object" || missing.length) {
    console.warn("Firebase Web SDK config incompleta. Missing:", missing);
} else {
    // Evitar doble init
    app = getApps().length ? getApp() : initializeApp(firebaseConfig);
    auth = getAuth(app);
    provider = new GoogleAuthProvider();
}

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
    if (!auth || !provider) {
        alert(
            "Login con Google no está configurado todavía.\n" +
            "Faltan variables de Firebase en el servidor (Render)."
        );
        return;
    }
    try {
        const result = await signInWithRedirect(auth, provider);
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
        if (!auth) return;
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
if (auth) {
    onAuthStateChanged(auth, (user) => {
        const logoutBtn = document.getElementById("googleLogoutBtn");
        if (logoutBtn) logoutBtn.style.display = user ? "inline-block" : "none";
    });
}

export { };


getRedirectResult(auth)
    .then(async (result) => {
        if (!result) return;

        const idToken = await result.user.getIdToken();
        // tu POST actual a /auth/session_login con idToken
        await fetch("/auth/session_login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id_token: idToken }),
        }).then(async (r) => {
            if (!r.ok) throw new Error(await r.text());
            window.location.href = "/";
        });
    })
    .catch((err) => {
        console.error("Redirect login error:", err);
        alert("Error al iniciar sesión con Google.\n" + (err?.message || err));
    });
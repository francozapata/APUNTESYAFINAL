// Firebase Google Auth
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, signInWithPopup, GoogleAuthProvider } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";


// 1) Pegá tu configuración real de Firebase acá:
const firebaseConfig = {
    apiKey: "TU_API_KEY",
    authDomain: "TU_PROJECT_ID.firebaseapp.com",
    projectId: "TU_PROJECT_ID",
    appId: "TU_APP_ID"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Si volvemos de redirect, terminamos el login
getRedirectResult(auth).then(async (result) => {
    if (result && result.user) {
        const idToken = await result.user.getIdToken();
        await backendSessionLogin(idToken);
        window.location.href = "/";
    }
}).catch((e) => console.error("Redirect error:", e));

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

window.googleSignIn = async function () {
    try {
        const result = await signInWithPopup(auth, provider);
        const user = result.user;
        const token = await user.getIdToken();
        console.log("✅ Login correcto:", user.email);

        // Enviar token a tu backend Flask
        const res = await fetch("/auth/session_login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id_token: token }),
        });
        if (!res.ok) throw new Error("Servidor rechazó el token");
        window.location.href = "/";
    } catch (e) {
        console.error("⚠️ Error Firebase:", e);
        alert("Error al iniciar sesión con Google.\n" + e.code);
    }
};

window.googleLogout = async function () {
    try {
        await signOut(auth);
        window.location.href = "/logout";
    } catch (e) {
        console.error(e);
        window.location.href = "/logout";
    }
};

onAuthStateChanged(auth, (user) => {
    const btn = document.getElementById("googleLogoutBtn");
    if (!btn) return;
    btn.style.display = user ? "inline-block" : "none";
});
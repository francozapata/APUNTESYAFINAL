// 1) Pegá tu configuración real de Firebase acá:
const firebaseConfig = {
    apiKey: "AIzaSyCHc6uy6uc1Jr6bzHQYGUZi2uZvTX0S9fE",
    authDomain: "apuntesya-d7d72.firebaseapp.com",
    projectId: "apuntesya-d7d72",
    storageBucket: "apuntesya-d7d72.firebasestorage.app",
    messagingSenderId: "332327927567",
    appId: "1:332327927567:web:22ecbb47817c2b7c71487a",
    measurementId: "G-9MBP39X788"
};

};
// 2) SDK Firebase
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import {
    getAuth, signInWithPopup, GoogleAuthProvider, signInWithRedirect,
    getRedirectResult, signOut, onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

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
        const idToken = await result.user.getIdToken();
        await backendSessionLogin(idToken);
        window.location.href = "/";
    } catch (e) {
        console.error("Popup error:", e);
        // Fallback si el popup se bloquea o se cierra
        if (e && (e.code === "auth/popup-blocked" || e.code === "auth/popup-closed-by-user")) {
            try {
                await signInWithRedirect(auth, provider);
                return;
            } catch (e2) {
                console.error("Redirect error:", e2);
            }
        }
        alert("Error al iniciar sesión con Google.\n" + (e && e.code ? e.code : ""));
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
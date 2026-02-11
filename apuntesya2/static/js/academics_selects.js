// static/js/academics_selects.js
(function () {

    function $(id) { return document.getElementById(id); }
    function setVisible(el, v) { if (el) el.style.display = v ? '' : 'none'; }
    function enable(el, v) { if (el) el.disabled = !v; }

    async function jfetch(url, opts) {
        const r = await fetch(url, opts || {});
        let data = {};
        try { data = await r.json(); } catch (_) { }
        if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
        return data;
    }

    async function load(sel, url) {
        if (!sel) return;
        const list = await jfetch(url);
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(x => {
            const o = document.createElement('option');
            o.value = String(x.id);
            o.textContent = x.name;
            sel.insertBefore(o, other);
        });
    }

    async function create(url, body) {
        return jfetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
    }

    window.initAcademicsSelects = async function (opts = {}) {
        const prefix = opts.prefix || 'acad';
        const enableCreate = !!opts.enableCreate;

        const uniSel = $(prefix + '-uni');
        const facSel = $(prefix + '-fac');
        const carSel = $(prefix + '-car');

        const uniOther = $(prefix + '-uni-other');
        const facOther = $(prefix + '-fac-other');
        const carOther = $(prefix + '-car-other');

        const hUni = $(prefix + '-hidden-university');
        const hFac = $(prefix + '-hidden-faculty');
        const hCar = $(prefix + '-hidden-career');

        let uniId = null;
        let facId = null;

        if (uniSel) await load(uniSel, '/api/academics/universities');

        uniSel?.addEventListener('change', async () => {
            if (uniSel.value === '__other__') {
                setVisible(uniOther, true);
                enable(facSel, false);
                return;
            }
            setVisible(uniOther, false);
            uniId = parseInt(uniSel.value || '0') || null;
            if (facSel && uniId) {
                facSel.innerHTML = '<option disabled selected>Elegí tu Facultad</option><option value="__other__">Otra…</option>';
                enable(facSel, true);
                await load(facSel, `/api/academics/faculties?university_id=${uniId}`);
            }
        });

        facSel?.addEventListener('change', async () => {
            if (facSel.value === '__other__') {
                setVisible(facOther, true);
                enable(carSel, false);
                return;
            }
            setVisible(facOther, false);
            facId = parseInt(facSel.value || '0') || null;
            if (carSel && facId) {
                carSel.innerHTML = '<option disabled selected>Elegí tu Carrera</option><option value="__other__">Otra…</option>';
                enable(carSel, true);
                await load(carSel, `/api/academics/careers?faculty_id=${facId}`);
            }
        });

        carSel?.addEventListener('change', () => {
            setVisible(carOther, carSel.value === '__other__');
        });

        // 🔑 FIX CLAVE: NUNCA BLOQUEAR SUBMIT
        async function resolveHidden() {

            // UNIVERSIDAD
            let u = '';
            if (uniSel?.value === '__other__' && uniOther?.value.trim()) {
                const r = await create('/api/academics/universities', { name: uniOther.value.trim() });
                u = r.name;
                uniId = r.id;
            } else if (uniSel?.value) {
                u = uniSel.options[uniSel.selectedIndex]?.text || '';
            }
            if (hUni) hUni.value = u;

            // FACULTAD
            let f = '';
            if (facSel?.value === '__other__' && facOther?.value.trim() && uniId) {
                const r = await create('/api/academics/faculties', { name: facOther.value.trim(), university_id: uniId });
                f = r.name;
                facId = r.id;
            } else if (facSel?.value) {
                f = facSel.options[facSel.selectedIndex]?.text || '';
            }
            if (hFac) hFac.value = f;

            // CARRERA
            let c = '';
            if (carSel?.value === '__other__' && carOther?.value.trim() && facId) {
                const r = await create('/api/academics/careers', { name: carOther.value.trim(), faculty_id: facId });
                c = r.name;
            } else if (carSel?.value) {
                c = carSel.options[carSel.selectedIndex]?.text || '';
            }
            if (hCar) hCar.value = c;
        }

        return { resolveHidden };
    };
})();

// static/js/academics_selects.js
(function () {
    function $(id) { return document.getElementById(id); }
    function vis(el, on) { el.style.display = on ? '' : 'none'; }
    function enable(el, on) { el.disabled = !on; }
    function opt(sel, ph, includeOther = true) {
        sel.innerHTML = '';
        const a = document.createElement('option');
        a.value = ''; a.disabled = true; a.selected = true; a.textContent = ph;
        sel.appendChild(a);
        if (includeOther) {
            const b = document.createElement('option');
            b.value = '__other__'; b.textContent = 'Otra…';
            sel.appendChild(b);
        }
    }
    async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw 0; return r.json(); }
    async function postJSON(url, body) {
        const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.error || 'Error');
        return d;
    }

    // Fire-and-forget suggestions (do not block user flow)
    async function suggest(kind, name, extra) {
        try {
            if (!name) return;
            await postJSON('/api/academics/suggestions', Object.assign({ kind, name }, extra || {}));
        } catch (_) {}
    }

    /**
     * initAcademicsSelects
     * opts = {
     *   prefix: 'cp' | 'up' | 'sr',
     *   formId?: 'formId' (si está => completa hidden y maneja submit),
     *   initial?: {university, faculty, career},  // para precarga libre
     *   enableCreate?: boolean (default true). Si false, NO permite "Otra…" (modo búsqueda)
     *   onChange?: function({universityText, facultyText, careerText}) // útil en búsqueda
     * }
     */
    async function initAcademicsSelects(opts) {
        const {
            prefix, formId = null, initial = null, enableCreate = true, onChange = null
        } = opts;

        const uni = $(`${prefix}-uni`);
        const fac = $(`${prefix}-fac`);
        const car = $(`${prefix}-car`);
        const uniO = $(`${prefix}-uni-other`);
        const facO = $(`${prefix}-fac-other`);
        const carO = $(`${prefix}-car-other`);
        const hUni = $(`${prefix}-hidden-university`);
        const hFac = $(`${prefix}-hidden-faculty`);
        const hCar = $(`${prefix}-hidden-career`);

        let uniId = null, facId = null;

        // Si no se permite crear, ocultar inputs libres y quitar "Otra…"
        if (!enableCreate) {
            vis(uniO, false); vis(facO, false); vis(carO, false);
            // Quitar opción "__other__" si existe
            ['uni', 'fac', 'car'].forEach(k => {
                const sel = $(`${prefix}-${k}`);
                const other = sel.querySelector('option[value="__other__"]');
                if (other) other.remove();
            });
        }

        // Cargar universidades
        try {
            const list = await getJSON('/api/academics/universities');
            const other = uni.querySelector('option[value="__other__"]');
            (list || []).forEach(u => {
                const o = document.createElement('option');
                o.value = String(u.id); o.textContent = u.name;
                other ? uni.insertBefore(o, other) : uni.appendChild(o);
            });
        } catch (_) { }

        uni.addEventListener('change', async () => {
            const v = uni.value;
            if (enableCreate && v === '__other__') {
                uniId = null; vis(uniO, true); vis(facO, true); vis(carO, true);
                enable(fac, false); enable(car, false);
            } else {
                vis(uniO, false); if (enableCreate) { vis(facO, false); vis(carO, false); }
                uniId = v ? parseInt(v, 10) : null;
                opt(fac, 'Elegí tu Facultad', enableCreate);
                opt(car, 'Elegí tu Carrera', enableCreate);
                // En modo creación, si no hay universidad seleccionada aún,
                // igual permitimos escribir Facultad/Carrera ("Otra…") para que
                // la búsqueda no sea restrictiva.
                if (!v && enableCreate) {
                    enable(fac, true);
                    enable(car, true);
                } else {
                    enable(fac, !!v);
                    enable(car, false);
                }

                if (v) {
                    try {
                        const list = await getJSON('/api/academics/faculties?university_id=' + encodeURIComponent(v));
                        const other = fac.querySelector('option[value="__other__"]');
                        (list || []).forEach(f => {
                            const o = document.createElement('option');
                            o.value = String(f.id); o.textContent = f.name;
                            other ? fac.insertBefore(o, other) : fac.appendChild(o);
                        });
                    } catch (_) { }
                }
            }
            fireChange();
        });

        fac.addEventListener('change', async () => {
            const v = fac.value;
            if (enableCreate && v === '__other__') {
                facId = null; vis(facO, true); vis(carO, true); enable(car, false);
            } else {
                vis(facO, false); if (enableCreate) { vis(carO, false); }
                facId = v ? parseInt(v, 10) : null;
                opt(car, 'Elegí tu Carrera', enableCreate);
                enable(car, !!v);
                if (v) {
                    try {
                        const list = await getJSON('/api/academics/careers?faculty_id=' + encodeURIComponent(v));
                        const other = car.querySelector('option[value="__other__"]');
                        (list || []).forEach(ca => {
                            const o = document.createElement('option');
                            o.value = String(ca.id); o.textContent = ca.name;
                            other ? car.insertBefore(o, other) : car.appendChild(o);
                        });
                    } catch (_) { }
                }
            }
            fireChange();
        });

        car.addEventListener('change', () => {
            if (enableCreate) vis(carO, car.value === '__other__');
            fireChange();
        });

        function getTexts() {
            const uniText = uni.value === '__other__' ? (uniO.value || '').trim()
                : (uni.value ? uni.options[uni.selectedIndex].text : '');
            const facText = fac.value === '__other__' ? (facO.value || '').trim()
                : (fac.value ? fac.options[fac.selectedIndex].text : '');
            const carText = car.value === '__other__' ? (carO.value || '').trim()
                : (car.value ? car.options[car.selectedIndex].text : '');
            return { universityText: uniText, facultyText: facText, careerText: carText };
        }
        function fireChange() {
            const detail = getTexts();

            // UX: if selected faculty is actually an institution, label it accordingly.
            try {
                const facLabel = fac && fac.parentElement ? fac.parentElement.querySelector('label') : null;
                const facText = detail.facultyText || '';
                const isInst = /Colegio Nacional de Monserrat|Escuela Superior de Comercio Manuel Belgrano/i.test(facText);
                if (facLabel) facLabel.textContent = isInst ? 'Institución' : 'Facultad';
            } catch (_) {}

            if (typeof onChange === 'function') onChange(detail);
            document.dispatchEvent(new CustomEvent('ay:academics-change', { detail: { prefix, ...detail } }));
        }

        // Submit (si hay formId): completa hidden y registra sugerencias si hace falta
        const form = formId ? $(formId.replace('#', '')) : null;
        if (form) {
            form.addEventListener('submit', async (ev) => {
                ev.preventDefault();
                try {
                    const uniText = uni.value === '__other__' ? (uniO.value || '').trim() : (uni.value ? uni.options[uni.selectedIndex].text : '');
                    const facText = fac.value === '__other__' ? (facO.value || '').trim() : (fac.value ? fac.options[fac.selectedIndex].text : '');
                    const carText = car.value === '__other__' ? (carO.value || '').trim() : (car.value ? car.options[car.selectedIndex].text : '');

                    if (!uniText) throw new Error('Seleccioná o escribí tu Universidad.');
                    if (!facText) throw new Error('Seleccioná o escribí tu Facultad/Institución.');
                    if (!carText) throw new Error('Seleccioná o escribí tu Carrera.');

                    // Registrar sugerencias si el usuario escribió "Otra…"
                    if (enableCreate && uni.value === '__other__') {
                        await suggest('university', uniText, {});
                    }
                    if (enableCreate && fac.value === '__other__') {
                        await suggest('faculty', facText, { university_id: uniId, university_name: uniText });
                    }
                    if (enableCreate && car.value === '__other__') {
                        await suggest('career', carText, { university_id: uniId, faculty_id: facId, university_name: uniText, faculty_name: facText });
                    }

                    if (hUni) hUni.value = uniText;
                    if (hFac) hFac.value = facText;
                    if (hCar) hCar.value = carText;

                    form.submit();
                } catch (e) { alert(e.message || 'No se pudo guardar. Intentá de nuevo.'); }
            });
        }

        // Precarga libre (opcional)
        if (initial && initial.university && enableCreate) {
            uni.value = '__other__'; vis(uniO, true); vis(facO, true); vis(carO, true);
            uniO.value = initial.university || ''; facO.value = initial.faculty || ''; carO.value = initial.career || '';
            enable(fac, false); enable(car, false);
        }
    }

    window.initAcademicsSelects = initAcademicsSelects;
})();

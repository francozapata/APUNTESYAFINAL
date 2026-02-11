// static/js/academics_selects.js
// Selects unificados Universidad / Facultad / Carrera con soporte "Otra…"
// - enableCreate=true: permite crear nuevas opciones vía API (POST) y muestra inputs "Otra…"
// - enableCreate=false: modo búsqueda, sin "Otra…" ni creación.
//
// Endpoints esperados:
//   GET  /api/academics/universities             -> [{id,name}]
//   GET  /api/academics/faculties?university_id  -> [{id,name}]
//   GET  /api/academics/careers?faculty_id       -> [{id,name}]
//   POST /api/academics/universities  {name}
//   POST /api/academics/faculties     {name, university_id}
//   POST /api/academics/careers       {name, faculty_id}
//
// API pública:
//   window.initAcademicsSelects({
//     prefix: 'acad',
//     enableCreate: true/false,
//     keywordInputId: 'keyword',   // opcional (búsqueda)
//     submitBtnId: 'search-btn',   // opcional (búsqueda)
//     onSearch: (params) => {},    // opcional (búsqueda)
//     onChange: ({level,value,universityText,facultyText,careerText}) => {}
//   })
//
(function () {
    function $(id) { return document.getElementById(id); }
    function setVisible(el, v) { if (el) el.style.display = v ? '' : 'none'; }
    function enable(el, v) { if (el) el.disabled = !v; }

    function clearSelect(sel, placeholder, includeOther) {
        if (!sel) return;
        sel.innerHTML = '';

        const ph = document.createElement('option');
        ph.value = '';
        ph.disabled = true;
        ph.selected = true;
        ph.textContent = placeholder;
        sel.appendChild(ph);

        if (includeOther) {
            const other = document.createElement('option');
            other.value = '__other__';
            other.textContent = 'Otra…';
            sel.appendChild(other);
        }
    }

    function getSelectedText(sel) {
        if (!sel || !sel.value) return '';
        const opt = sel.options[sel.selectedIndex];
        return opt ? (opt.text || '') : '';
    }

    function qs(params) {
        const u = new URLSearchParams();
        Object.entries(params).forEach(([k, v]) => {
            if (v !== undefined && v !== null && String(v).trim() !== '') u.set(k, v);
        });
        return u.toString();
    }

    async function jfetch(url, opts) {
        const r = await fetch(url, opts || {});
        let data = {};
        try { data = await r.json(); } catch (_) { data = {}; }
        if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
        return data;
    }

    async function loadUniversities(sel) {
        if (!sel) return;
        const list = await jfetch('/api/academics/universities');
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(u => {
            const o = document.createElement('option');
            o.value = String(u.id);
            o.textContent = u.name;
            if (other) sel.insertBefore(o, other);
            else sel.appendChild(o);
        });
    }

    async function loadFaculties(sel, universityId) {
        if (!sel) return;
        const list = await jfetch('/api/academics/faculties?university_id=' + encodeURIComponent(universityId));
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(f => {
            const o = document.createElement('option');
            o.value = String(f.id);
            o.textContent = f.name;
            if (other) sel.insertBefore(o, other);
            else sel.appendChild(o);
        });
    }

    async function loadCareers(sel, facultyId) {
        if (!sel) return;
        const list = await jfetch('/api/academics/careers?faculty_id=' + encodeURIComponent(facultyId));
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(c => {
            const o = document.createElement('option');
            o.value = String(c.id);
            o.textContent = c.name;
            if (other) sel.insertBefore(o, other);
            else sel.appendChild(o);
        });
    }

    async function createUniversity(name) {
        return jfetch('/api/academics/universities', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
    }

    async function createFaculty(name, university_id) {
        return jfetch('/api/academics/faculties', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, university_id })
        });
    }

    async function createCareer(name, faculty_id) {
        return jfetch('/api/academics/careers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, faculty_id })
        });
    }

    window.initAcademicsSelects = async function initAcademicsSelects(opts) {
        opts = opts || {};
        const prefix = opts.prefix || 'acad';
        const enableCreate = !!opts.enableCreate;
        const onChange = typeof opts.onChange === 'function' ? opts.onChange : () => { };

        // selects
        const uniSel = $(prefix + '-uni');
        const facSel = $(prefix + '-fac');
        const carSel = $(prefix + '-car');

        // inputs "Otra…"
        const uniOther = $(prefix + '-uni-other');
        const facOther = $(prefix + '-fac-other');
        const carOther = $(prefix + '-car-other');

        // hidden (para enviar texto final al backend)
        const hUni = $(prefix + '-hidden-university');
        const hFac = $(prefix + '-hidden-faculty');
        const hCar = $(prefix + '-hidden-career');

        // búsqueda opcional
        const keywordInput = $(opts.keywordInputId || 'keyword');
        const submitBtn = $(opts.submitBtnId || 'search-btn');
        const onSearch = typeof opts.onSearch === 'function' ? opts.onSearch : null;

        let chosenUniId = null;
        let chosenFacId = null;

        // Si no se permite crear: eliminamos inputs y la opción "Otra…"
        if (!enableCreate) {
            if (uniOther) uniOther.remove();
            if (facOther) facOther.remove();
            if (carOther) carOther.remove();
        }

        // ✅ INICIALIZACIÓN CORRECTA (SOLUCIONA tu bug)
        // Siempre render placeholder; en create-mode agregamos "Otra…"
        clearSelect(uniSel, 'Elegí tu Universidad', enableCreate);
        clearSelect(facSel, 'Elegí tu Facultad', enableCreate);
        clearSelect(carSel, 'Elegí tu Carrera', enableCreate);
        enable(facSel, false);
        enable(carSel, false);

        // Ocultar inputs "Otra…" inicialmente
        setVisible(uniOther, false);
        setVisible(facOther, false);
        setVisible(carOther, false);

        // Cargar universidades
        if (uniSel) await loadUniversities(uniSel);

        function getCurrentTexts() {
            const universityText = (uniSel && uniSel.value === '__other__')
                ? ((uniOther && uniOther.value) || '').trim()
                : (uniSel ? getSelectedText(uniSel) : '');

            const facultyText = (facSel && facSel.value === '__other__')
                ? ((facOther && facOther.value) || '').trim()
                : (facSel ? getSelectedText(facSel) : '');

            const careerText = (carSel && carSel.value === '__other__')
                ? ((carOther && carOther.value) || '').trim()
                : (carSel ? getSelectedText(carSel) : '');

            return { universityText, facultyText, careerText };
        }

        function syncHidden() {
            const { universityText, facultyText, careerText } = getCurrentTexts();
            if (hUni) hUni.value = universityText || '';
            if (hFac) hFac.value = facultyText || '';
            if (hCar) hCar.value = careerText || '';
        }

        function emitChange(level, value) {
            const { universityText, facultyText, careerText } = getCurrentTexts();
            onChange({ level, value, universityText, facultyText, careerText });
            syncHidden();
        }

        // handlers
        if (uniSel) {
            uniSel.addEventListener('change', async () => {
                const v = uniSel.value;
                emitChange('university', v);

                if (v === '__other__') {
                    setVisible(uniOther, true);
                    // si es "Otra…" dejamos facultad/carrera editables por "Otra…" también,
                    // pero no cargamos listas (no hay uniId todavía)
                    setVisible(facOther, true);
                    setVisible(carOther, true);
                    enable(facSel, false);
                    enable(carSel, false);
                    chosenUniId = null;
                    chosenFacId = null;
                    return;
                }

                setVisible(uniOther, false);
                setVisible(facOther, false);
                setVisible(carOther, false);

                chosenUniId = parseInt(v || '0', 10) || null;

                // cargar facultades
                clearSelect(facSel, 'Elegí tu Facultad', enableCreate);
                clearSelect(carSel, 'Elegí tu Carrera', enableCreate);
                enable(facSel, !!chosenUniId);
                enable(carSel, false);

                if (chosenUniId) {
                    await loadFaculties(facSel, v);
                }
            });
        }

        if (facSel) {
            facSel.addEventListener('change', async () => {
                const v = facSel.value;
                emitChange('faculty', v);

                if (v === '__other__') {
                    setVisible(facOther, true);
                    setVisible(carOther, true);
                    enable(carSel, false);
                    chosenFacId = null;
                    return;
                }

                setVisible(facOther, false);
                setVisible(carOther, false);

                chosenFacId = parseInt(v || '0', 10) || null;

                // cargar carreras
                clearSelect(carSel, 'Elegí tu Carrera', enableCreate);
                enable(carSel, !!chosenFacId);

                if (chosenFacId) {
                    await loadCareers(carSel, v);
                }
            });
        }

        if (carSel) {
            carSel.addEventListener('change', () => {
                const v = carSel.value;
                emitChange('career', v);
                setVisible(carOther, v === '__other__' && enableCreate);
            });
        }

        // ✅ RESOLVE (NUNCA BLOQUEA; crea si hay input y se puede)
        async function resolveHidden() {
            // UNIVERSIDAD
            let uName = '';
            if (uniSel && uniSel.value === '__other__') {
                const name = (uniOther && uniOther.value.trim()) || '';
                if (name) {
                    const u = await createUniversity(name);
                    chosenUniId = u.id;
                    uName = u.name;
                }
            } else if (uniSel && uniSel.value) {
                uName = getSelectedText(uniSel);
                chosenUniId = parseInt(uniSel.value, 10) || null;
            }
            if (hUni) hUni.value = uName;

            // FACULTAD
            let fName = '';
            if (facSel && facSel.value === '__other__') {
                const name = (facOther && facOther.value.trim()) || '';
                if (name && chosenUniId) {
                    const f = await createFaculty(name, chosenUniId);
                    chosenFacId = f.id;
                    fName = f.name;
                }
            } else if (facSel && facSel.value) {
                fName = getSelectedText(facSel);
                chosenFacId = parseInt(facSel.value, 10) || null;
            }
            if (hFac) hFac.value = fName;

            // CARRERA
            let cName = '';
            if (carSel && carSel.value === '__other__') {
                const name = (carOther && carOther.value.trim()) || '';
                if (name && chosenFacId) {
                    const c = await createCareer(name, chosenFacId);
                    cName = c.name;
                }
            } else if (carSel && carSel.value) {
                cName = getSelectedText(carSel);
            }
            if (hCar) hCar.value = cName;
        }

        // -------- MODO BÚSQUEDA ----------
        function collectSearchParams() {
            const q = (keywordInput && keywordInput.value || '').trim();
            const university = uniSel ? getSelectedText(uniSel) : '';
            const faculty = facSel ? getSelectedText(facSel) : '';
            const career = carSel ? getSelectedText(carSel) : '';
            return { q, university, faculty, career };
        }

        async function triggerSearch() {
            const params = collectSearchParams();
            if (onSearch) return onSearch(params);
            const query = qs(params);
            window.location.assign('/search' + (query ? ('?' + query) : ''));
        }

        if (!enableCreate) {
            if (keywordInput) {
                keywordInput.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Enter') { ev.preventDefault(); triggerSearch(); }
                });
            }
            if (submitBtn) {
                submitBtn.addEventListener('click', (ev) => {
                    ev.preventDefault(); triggerSearch();
                });
            }
        }

        // inicial sync hidden
        syncHidden();

        return {
            resolveHidden,
            getSearchParams: collectSearchParams,
            search: triggerSearch
        };
    };
})();

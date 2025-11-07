// static/js/academics_selects.js
// Componente único de selects (Universidad/Facultad/Carrera) con "Otra…" y creación,
// y soporte de MODO BÚSQUEDA con palabra clave + filtros combinables.
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
//     prefix: 'acad',           // ids: acad-uni, acad-fac, acad-car (+ *-other y *-hidden-.. si enableCreate)
//     enableCreate: false,      // true en perfil / subir; false en buscador
//
//     // --- OPCIONALES para modo búsqueda ---
//     keywordInputId: 'keyword',     // id del input de texto libre (si existe)
//     submitBtnId: 'search-btn',     // id de botón para disparar búsqueda (si existe)
//     onSearch: (params) => {        // params = { q, university, faculty, career }
//        // si no lo definís, navega a /search con querystring.
//     },
//     onChange: ({level, value}) => {}  // callback genérico ante cambios
//   })
//
(function () {
    function $(id) { return document.getElementById(id); }
    function setVisible(el, v) { if (el) el.style.display = v ? '' : 'none'; }
    function enable(el, v) { if (el) el.disabled = !v; }
    function clearSelect(sel, ph) {
        if (!sel) return;
        sel.innerHTML = '';
        const phOpt = document.createElement('option');
        phOpt.value = ''; phOpt.disabled = true; phOpt.selected = true; phOpt.textContent = ph;
        sel.appendChild(phOpt);
        const other = document.createElement('option');
        other.value = '__other__'; other.textContent = 'Otra…';
        sel.appendChild(other);
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
            sel.insertBefore(o, other);
        });
    }
    async function loadFaculties(sel, universityId) {
        if (!sel) return;
        clearSelect(sel, 'Elegí tu Facultad');
        const list = await jfetch('/api/academics/faculties?university_id=' + encodeURIComponent(universityId));
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(f => {
            const o = document.createElement('option');
            o.value = String(f.id);
            o.textContent = f.name;
            sel.insertBefore(o, other);
        });
    }
    async function loadCareers(sel, facultyId) {
        if (!sel) return;
        clearSelect(sel, 'Elegí tu Carrera');
        const list = await jfetch('/api/academics/careers?faculty_id=' + encodeURIComponent(facultyId));
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(c => {
            const o = document.createElement('option');
            o.value = String(c.id);
            o.textContent = c.name;
            sel.insertBefore(o, other);
        });
    }
    async function createUniversity(name) {
        return jfetch('/api/academics/universities', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
    }
    async function createFaculty(name, university_id) {
        return jfetch('/api/academics/faculties', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, university_id })
        });
    }
    async function createCareer(name, faculty_id) {
        return jfetch('/api/academics/careers', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, faculty_id })
        });
    }

    // API pública
    window.initAcademicsSelects = async function initAcademicsSelects(opts) {
        opts = opts || {};
        const prefix = opts.prefix || 'acad';
        const enableCreate = !!opts.enableCreate; // true en perfil/subir; false en buscador
        const onChange = typeof opts.onChange === 'function' ? opts.onChange : () => { };

        // elementos base
        const uniSel = $(prefix + '-uni');
        const facSel = $(prefix + '-fac');
        const carSel = $(prefix + '-car');

        const uniOther = $(prefix + '-uni-other');
        const facOther = $(prefix + '-fac-other');
        const carOther = $(prefix + '-car-other');

        const hUni = $(prefix + '-hidden-university');
        const hFac = $(prefix + '-hidden-faculty');
        const hCar = $(prefix + '-hidden-career');

        // elementos de búsqueda (opcionales)
        const keywordInput = $(opts.keywordInputId || 'keyword');
        const submitBtn = $(opts.submitBtnId || 'search-btn');
        const onSearch = typeof opts.onSearch === 'function' ? opts.onSearch : null;

        let chosenUniId = null;
        let chosenFacId = null;

        // Si no se permite crear, oculto inputs "otra…" y quito opción "Otra…"
        if (!enableCreate) {
            if (uniOther) uniOther.remove();
            if (facOther) facOther.remove();
            if (carOther) carOther.remove();
            [uniSel, facSel, carSel].forEach(sel => {
                if (!sel) return;
                const optOther = sel.querySelector('option[value="__other__"]');
                if (optOther) optOther.remove();
            });
        }

        // Primer render
        if (uniSel) await loadUniversities(uniSel);

        // Handlers selects
        if (uniSel) {
            uniSel.addEventListener('change', async () => {
                const v = uniSel.value;
                onChange({ level: 'university', value: v });

                if (v === '__other__') {
                    if (!enableCreate) { uniSel.value = ''; return; }
                    setVisible(uniOther, true);
                    setVisible(facOther, true);
                    setVisible(carOther, true);
                    enable(facSel, false); enable(carSel, false);
                    chosenUniId = null;
                    return;
                }
                setVisible(uniOther, false); setVisible(facOther, false); setVisible(carOther, false);
                chosenUniId = parseInt(v || '0', 10) || null;
                if (facSel) { enable(facSel, !!v); await loadFaculties(facSel, v); }
                if (carSel) { enable(carSel, false); clearSelect(carSel, 'Elegí tu Carrera'); }
            });
        }

        if (facSel) {
            facSel.addEventListener('change', async () => {
                const v = facSel.value;
                onChange({ level: 'faculty', value: v });

                if (v === '__other__') {
                    if (!enableCreate) { facSel.value = ''; return; }
                    setVisible(facOther, true);
                    setVisible(carOther, true);
                    enable(carSel, false);
                    chosenFacId = null;
                    return;
                }
                setVisible(facOther, false); setVisible(carOther, false);
                chosenFacId = parseInt(v || '0', 10) || null;
                if (carSel) { enable(carSel, !!v); await loadCareers(carSel, v); }
            });
        }

        if (carSel) {
            carSel.addEventListener('change', () => {
                const v = carSel.value;
                onChange({ level: 'career', value: v });
                setVisible(carOther, v === '__other__' && enableCreate);
            });
        }

        // -- Resolver ocultos (para formularios de perfil/subir) --
        async function resolveHidden() {
            // UNIVERSIDAD
            let uName = '';
            if (uniSel && uniSel.value === '__other__') {
                if (!enableCreate) throw new Error('No se permite crear Universidad aquí.');
                const name = (uniOther && uniOther.value.trim()) || '';
                if (!name) throw new Error('Escribí tu Universidad.');
                const u = await createUniversity(name);
                chosenUniId = u.id;
                uName = u.name;
            } else if (uniSel && uniSel.value) {
                uName = getSelectedText(uniSel);
                chosenUniId = parseInt(uniSel.value, 10);
            } else {
                throw new Error('Seleccioná tu Universidad.');
            }
            if (hUni) hUni.value = uName;

            // FACULTAD
            let fName = '';
            if (!facSel || facSel.value === '__other__' || !facSel.value) {
                if (!enableCreate) throw new Error('Seleccioná tu Facultad.');
                const name = (facOther && facOther.value.trim()) || '';
                if (!name) throw new Error('Escribí tu Facultad.');
                const f = await createFaculty(name, chosenUniId);
                chosenFacId = f.id; fName = f.name;
            } else {
                fName = getSelectedText(facSel);
                chosenFacId = parseInt(facSel.value, 10);
            }
            if (hFac) hFac.value = fName;

            // CARRERA
            let cName = '';
            if (!carSel || carSel.value === '__other__' || !carSel.value) {
                if (!enableCreate) throw new Error('Seleccioná tu Carrera.');
                const name = (carOther && carOther.value.trim()) || '';
                if (!name) throw new Error('Escribí tu Carrera.');
                const c = await createCareer(name, chosenFacId);
                cName = c.name;
            } else {
                cName = getSelectedText(carSel);
            }
            if (hCar) hCar.value = cName;
        }

        // -------- MODO BÚSQUEDA (opcional) ----------
        function collectSearchParams() {
            const q = (keywordInput && keywordInput.value || '').trim();

            // Para búsqueda, mandamos NOMBRES (texto del option) en lugar de IDs
            const university = uniSel ? getSelectedText(uniSel) : '';
            const faculty = facSel ? getSelectedText(facSel) : '';
            const career = carSel ? getSelectedText(carSel) : '';

            return { q, university, faculty, career };
        }

        async function triggerSearch() {
            const params = collectSearchParams();
            if (onSearch) {
                // callback del integrador (por ej. fetch y render en vivo)
                return onSearch(params);
            }
            // fallback: navegar a /search con querystring
            const query = qs(params);
            const url = '/search' + (query ? ('?' + query) : '');
            window.location.assign(url);
        }

        if (keywordInput) {
            keywordInput.addEventListener('keydown', (ev) => {
                if (ev.key === 'Enter') {
                    ev.preventDefault();
                    triggerSearch();
                }
            });
        }
        if (submitBtn) {
            submitBtn.addEventListener('click', (ev) => {
                ev.preventDefault();
                triggerSearch();
            });
        }

        // También disparar búsqueda si cambian filtros (solo si no usamos enableCreate)
        if (!enableCreate) {
            [uniSel, facSel, carSel].forEach(sel => {
                if (!sel) return;
                sel.addEventListener('change', () => {
                    // si querés búsqueda “en vivo” al elegir filtros, descomentá:
                    // triggerSearch();
                });
            });
        }

        // Exponer helpers al integrador
        return {
            resolveHidden,          // para formularios
            getSearchParams: collectSearchParams,
            search: triggerSearch
        };
    };
})();

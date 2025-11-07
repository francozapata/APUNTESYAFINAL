// static/js/academics_selects.js
// Componente único de selects (Universidad/Facultad/Carrera) con "Otra…" y creación.
// Endpoints esperados:
//   GET  /api/academics/universities            -> [{id,name}]
//   GET  /api/academics/faculties?university_id -> [{id,name}]
//   GET  /api/academics/careers?faculty_id      -> [{id,name}]
//   POST /api/academics/universities  {name}
//   POST /api/academics/faculties     {name, university_id}
//   POST /api/academics/careers       {name, faculty_id}

(function () {
    function $(id) { return document.getElementById(id); }
    function setVisible(el, v) { if (el) el.style.display = v ? '' : 'none'; }
    function enable(el, v) { if (el) el.disabled = !v; }
    function clearSelect(sel, ph) {
        sel.innerHTML = '';
        const a = document.createElement('option');
        a.value = ''; a.disabled = true; a.selected = true; a.textContent = ph;
        sel.appendChild(a);
        const b = document.createElement('option');
        b.value = '__other__'; b.textContent = 'Otra…';
        sel.appendChild(b);
    }
    async function jfetch(url, opts) {
        const r = await fetch(url, opts || {});
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
        return data;
    }
    async function loadUniversities(sel) {
        const list = await jfetch('/api/academics/universities');
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(u => {
            const o = document.createElement('option'); o.value = String(u.id); o.textContent = u.name;
            sel.insertBefore(o, other);
        });
    }
    async function loadFaculties(sel, universityId) {
        clearSelect(sel, 'Elegí tu Facultad');
        const list = await jfetch('/api/academics/faculties?university_id=' + encodeURIComponent(universityId));
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(f => {
            const o = document.createElement('option'); o.value = String(f.id); o.textContent = f.name;
            sel.insertBefore(o, other);
        });
    }
    async function loadCareers(sel, facultyId) {
        clearSelect(sel, 'Elegí tu Carrera');
        const list = await jfetch('/api/academics/careers?faculty_id=' + encodeURIComponent(facultyId));
        const other = sel.querySelector('option[value="__other__"]');
        (list || []).forEach(c => {
            const o = document.createElement('option'); o.value = String(c.id); o.textContent = c.name;
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
        const prefix = opts.prefix || 'acad';
        const enableCreate = !!opts.enableCreate;     // true en perfil/subir; false en buscador
        const onChange = typeof opts.onChange === 'function' ? opts.onChange : () => { };

        const uniSel = $(prefix + '-uni');
        const facSel = $(prefix + '-fac');
        const carSel = $(prefix + '-car');
        const uniOther = $(prefix + '-uni-other');
        const facOther = $(prefix + '-fac-other');
        const carOther = $(prefix + '-car-other');

        const hUni = $(prefix + '-hidden-university');
        const hFac = $(prefix + '-hidden-faculty');
        const hCar = $(prefix + '-hidden-career');

        let chosenUniId = null;
        let chosenFacId = null;

        if (!enableCreate) {
            // quitar inputs "otra…" y las opciones "Otra…"
            if (uniOther) uniOther.remove();
            if (facOther) facOther.remove();
            if (carOther) carOther.remove();
            [uniSel, facSel, carSel].forEach(sel => {
                if (!sel) return;
                const optOther = sel.querySelector('option[value="__other__"]');
                if (optOther) optOther.remove();
            });
        }

        await loadUniversities(uniSel);

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
            chosenUniId = parseInt(v, 10);
            enable(facSel, true); enable(carSel, false);
            await loadFaculties(facSel, v);
            clearSelect(carSel, 'Elegí tu Carrera');
        });

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
            chosenFacId = parseInt(v, 10);
            enable(carSel, true);
            await loadCareers(carSel, v);
        });

        carSel.addEventListener('change', () => {
            const v = carSel.value;
            onChange({ level: 'career', value: v });
            setVisible(carOther, v === '__other__' && enableCreate);
        });

        async function resolveHidden() {
            // Universidad
            let uName = '';
            if (uniSel.value === '__other__') {
                const name = (uniOther && uniOther.value.trim()) || '';
                if (!name) throw new Error('Escribí tu Universidad.');
                const u = await createUniversity(name);
                chosenUniId = u.id;
                uName = u.name;
            } else if (uniSel.value) {
                uName = uniSel.options[uniSel.selectedIndex]?.text || '';
            } else {
                throw new Error('Seleccioná tu Universidad.');
            }
            if (hUni) hUni.value = uName;

            // Facultad
            let fName = '';
            if (facSel.value === '__other__') {
                const name = (facOther && facOther.value.trim()) || '';
                if (!name) throw new Error('Escribí tu Facultad.');
                const f = await createFaculty(name, chosenUniId);
                chosenFacId = f.id;
                fName = f.name;
            } else if (facSel.value) {
                fName = facSel.options[facSel.selectedIndex]?.text || '';
            } else {
                throw new Error('Seleccioná tu Facultad.');
            }
            if (hFac) hFac.value = fName;

            // Carrera
            let cName = '';
            if (carSel.value === '__other__') {
                const name = (carOther && carOther.value.trim()) || '';
                if (!name) throw new Error('Escribí tu Carrera.');
                const c = await createCareer(name, chosenFacId);
                cName = c.name;
            } else if (carSel.value) {
                cName = carSel.options[carSel.selectedIndex]?.text || '';
            } else {
                throw new Error('Seleccioná tu Carrera.');
            }
            if (hCar) hCar.value = cName;
        }

        return { resolveHidden };
    };
})();

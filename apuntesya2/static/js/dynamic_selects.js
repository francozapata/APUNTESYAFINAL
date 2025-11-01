
// Minimal, no-aesthetic dynamic selects with "Otra..." that learn via API
(function() {
  function el(tag, attrs={}, children=[]) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k,v]) => {
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else e.setAttribute(k, v);
    });
    children.forEach(c => e.appendChild(c));
    return e;
  }
  async function fetchJSON(url, opts) {
    const res = await fetch(url, Object.assign({headers: {'Content-Type':'application/json'}}, opts || {}));
    if (!res.ok) throw new Error('HTTP '+res.status);
    return await res.json();
  }
  function makeSelect(placeholder) {
    const s = el('select', {class:'input'});
    s.appendChild(el('option', {value:'', text: placeholder || 'Seleccionar...'}));
    return s;
  }
  async function ensureOption(select, value, label) {
    let opt = Array.from(select.options).find(o => o.value == String(value));
    if (!opt) {
      opt = el('option', {value: String(value), text: label});
      select.appendChild(opt);
    }
    return opt;
  }
  async function initGroup(container) {
    const uniInput = container.querySelector('input[name="university"]');
    const facInput = container.querySelector('input[name="faculty"]');
    const carInput = container.querySelector('input[name="career"]');
    if (!(uniInput && facInput && carInput)) return;

    // Hide text inputs but keep them as hidden to submit string values server expects
    [uniInput, facInput, carInput].forEach(i => { i.type = 'hidden'; });

    const uniSelect = makeSelect('Universidad');
    const facSelect = makeSelect('Facultad');
    const carSelect = makeSelect('Carrera');

    // Insert selects right after labels (keeping layout)
    uniInput.parentElement.insertBefore(uniSelect, uniInput);
    facInput.parentElement.insertBefore(facSelect, facInput);
    carInput.parentElement.insertBefore(carSelect, carInput);

    function syncHidden(select, hiddenInput) {
      const val = select.value;
      if (!val || val === '__OTHER__') { hiddenInput.value = ''; return; }
      const opt = select.selectedOptions[0];
      const txt = opt ? (opt.dataset.label || opt.textContent || '').trim() : '';
      // '__TEMP__' is a user-typed value not yet saved in DB; still submit its label string
      hiddenInput.value = val === '__TEMP__' ? txt : txt;
    }

    // Add fixed "Otra..." option
    function addOtherOption(select) {
      const sep = el('option', {value:'', text:'──────────'});
      sep.disabled = true;
      select.appendChild(sep);
      const other = el('option', {value:'__OTHER__', text:'Otra…'});
      select.appendChild(other);
    }

    // Load universities
    async function loadUniversities(prefillName) {
      uniSelect.innerHTML = '';
      uniSelect.appendChild(el('option', {value:'', text:'Universidad'}));
      const data = await fetchJSON('/api/academics/universities');
      data.forEach(u => {
        const o = el('option', {value:String(u.id), text:u.name});
        o.dataset.label = u.name;
        uniSelect.appendChild(o);
      });
      addOtherOption(uniSelect);
      if (prefillName) {
        const match = Array.from(uniSelect.options).find(o => o.textContent.trim().toLowerCase() === prefillName.trim().toLowerCase());
        if (match) uniSelect.value = match.value;
        else {
          // Add as temporary option (without saving) so it shows selected
          const temp = el('option', {value:'__TEMP__', text: prefillName});
          temp.dataset.label = prefillName;
          uniSelect.appendChild(temp);
          uniSelect.value = '__TEMP__';
        }
      }
      syncHidden(uniSelect, uniInput);
      await loadFaculties(null, facInput.value);
    }

    async function loadFaculties(universityId, prefillName) {
      facSelect.innerHTML = '';
      facSelect.appendChild(el('option', {value:'', text:'Facultad'}));
      let url = '/api/academics/faculties';
      if (universityId) url += '?university_id='+encodeURIComponent(universityId);
      const data = await fetchJSON(url);
      data.forEach(f => {
        const o = el('option', {value:String(f.id), text:f.name});
        o.dataset.label = f.name;
        facSelect.appendChild(o);
      });
      addOtherOption(facSelect);
      if (prefillName) {
        const match = Array.from(facSelect.options).find(o => o.textContent.trim().toLowerCase() === prefillName.trim().toLowerCase());
        if (match) facSelect.value = match.value;
        else {
          const temp = el('option', {value:'__TEMP__', text: prefillName});
          temp.dataset.label = prefillName;
          facSelect.appendChild(temp);
          facSelect.value = '__TEMP__';
        }
      }
      syncHidden(facSelect, facInput);
      await loadCareers(null, carInput.value);
    }

    async function loadCareers(facultyId, prefillName) {
      carSelect.innerHTML = '';
      carSelect.appendChild(el('option', {value:'', text:'Carrera'}));
      let url = '/api/academics/careers';
      if (facultyId) url += '?faculty_id='+encodeURIComponent(facultyId);
      const data = await fetchJSON(url);
      data.forEach(c => {
        const o = el('option', {value:String(c.id), text:c.name});
        o.dataset.label = c.name;
        carSelect.appendChild(o);
      });
      addOtherOption(carSelect);
      if (prefillName) {
        const match = Array.from(carSelect.options).find(o => o.textContent.trim().toLowerCase() === prefillName.trim().toLowerCase());
        if (match) carSelect.value = match.value;
        else {
          const temp = el('option', {value:'__TEMP__', text: prefillName});
          temp.dataset.label = prefillName;
          carSelect.appendChild(temp);
          carSelect.value = '__TEMP__';
        }
      }
      syncHidden(carSelect, carInput);
    }

    uniSelect.addEventListener('change', async function() {
      if (this.value === '__OTHER__') {
        const wrap = this.parentElement;
        let mini = wrap.querySelector('.mini-add');
        if (mini) mini.remove();
        mini = document.createElement('input');
        mini.className = 'input mini-add';
        mini.placeholder = 'Nueva universidad';
        mini.style.marginTop = '6px';
        wrap.appendChild(mini);
        mini.focus();
        const commit = async () => {
          const name = (mini.value||'').trim();
          if (!name) { this.value=''; mini.remove(); syncHidden(this, uniInput); return; }
          const res = await fetch('/api/academics/universities', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})}).then(r=>r.json());
          const opt = document.createElement('option'); opt.value=String(res.id); opt.textContent=res.name; opt.dataset.label=res.name;
          this.insertBefore(opt, this.querySelector('option[value="__OTHER__"]'));
          this.value = String(res.id);
          mini.remove();
          syncHidden(this, uniInput);
          await loadFaculties(this.value, null);
        };
        mini.addEventListener('keydown', async (ev)=>{
          if (ev.key === 'Enter') { ev.preventDefault(); await commit(); }
          else if (ev.key === 'Escape') { this.value=''; mini.remove(); syncHidden(this, uniInput); }
        });
        mini.addEventListener('blur', commit);
        return;
      }
      syncHidden(this, uniInput);
      const uid = this.value && this.value !== '__TEMP__' ? this.value : null;
      await loadFaculties(uid, null);
    });

    facSelect.addEventListener('change', async function() {
      if (this.value === '__OTHER__') {
        const uid = uniSelect.value && uniSelect.value !== '__TEMP__' ? parseInt(uniSelect.value,10) : null;
        if (!uid) { alert('Primero seleccioná la Universidad.'); this.value = ''; return; }
        const wrap = this.parentElement;
        let mini = wrap.querySelector('.mini-add');
        if (mini) mini.remove();
        mini = document.createElement('input');
        mini.className = 'input mini-add';
        mini.placeholder = 'Nueva facultad';
        mini.style.marginTop = '6px';
        wrap.appendChild(mini);
        mini.focus();
        const commit = async () => {
          const name = (mini.value||'').trim();
          if (!name) { this.value=''; mini.remove(); syncHidden(this, facInput); return; }
          const res = await fetch('/api/academics/faculties', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name, university_id: uid})}).then(r=>r.json());
          const opt = document.createElement('option'); opt.value=String(res.id); opt.textContent=res.name; opt.dataset.label=res.name;
          this.insertBefore(opt, this.querySelector('option[value="__OTHER__"]'));
          this.value = String(res.id);
          mini.remove();
          syncHidden(this, facInput);
          await loadCareers(this.value, null);
        };
        mini.addEventListener('keydown', async (ev)=>{
          if (ev.key === 'Enter') { ev.preventDefault(); await commit(); }
          else if (ev.key === 'Escape') { this.value=''; mini.remove(); syncHidden(this, facInput); }
        });
        mini.addEventListener('blur', commit);
        return;
      }
      syncHidden(this, facInput);
      const fid = this.value && this.value !== '__TEMP__' ? this.value : null;
      await loadCareers(fid, null);
    });

    carSelect.addEventListener('change', async function() {
      if (this.value === '__OTHER__') {
        const fid = facSelect.value && facSelect.value !== '__TEMP__' ? parseInt(facSelect.value,10) : null;
        if (!fid) { alert('Primero seleccioná la Facultad.'); this.value = ''; return; }
        const wrap = this.parentElement;
        let mini = wrap.querySelector('.mini-add');
        if (mini) mini.remove();
        mini = document.createElement('input');
        mini.className = 'input mini-add';
        mini.placeholder = 'Nueva carrera';
        mini.style.marginTop = '6px';
        wrap.appendChild(mini);
        mini.focus();
        const commit = async () => {
          const name = (mini.value||'').trim();
          if (!name) { this.value=''; mini.remove(); syncHidden(this, carInput); return; }
          const res = await fetch('/api/academics/careers', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name, faculty_id: fid})}).then(r=>r.json());
          const opt = document.createElement('option'); opt.value=String(res.id); opt.textContent=res.name; opt.dataset.label=res.name;
          this.insertBefore(opt, this.querySelector('option[value="__OTHER__"]'));
          this.value = String(res.id);
          mini.remove();
          syncHidden(this, carInput);
        };
        mini.addEventListener('keydown', async (ev)=>{
          if (ev.key === 'Enter') { ev.preventDefault(); await commit(); }
          else if (ev.key === 'Escape') { this.value=''; mini.remove(); syncHidden(this, carInput); }
        });
        mini.addEventListener('blur', commit);
        return;
      }
      syncHidden(this, carInput);
    });

    // Initial load with possible prefilled hidden values
    await loadUniversities(uniInput.value);
  }

  document.addEventListener('DOMContentLoaded', function() {
    // For each form that has the 3 inputs, initialize a group
    document.querySelectorAll('form').forEach(form => {
      const hasAll = form.querySelector('input[name="university"]') && form.querySelector('input[name="faculty"]') && form.querySelector('input[name="career"]');
      if (!hasAll) return;
      // avoid running twice
      if (form.dataset.dynamicSelectsInit) return;
      form.dataset.dynamicSelectsInit = '1';
      initGroup(form);
    });
  });
})();

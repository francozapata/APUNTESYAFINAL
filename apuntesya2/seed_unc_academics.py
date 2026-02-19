import re
from typing import Dict, List, Tuple

from sqlalchemy import select

from .models import University, Faculty, Career


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    # Normalize inclusive forms like "Contador/a" -> "Contador"
    s = s.replace("/a", "")
    s = s.replace("/o", "")
    s = s.replace("Técnico/", "Técnico ")
    # Fix common double spaces after replacements
    s = re.sub(r"\s+", " ", s).strip()
    # Title-style: keep common prepositions lower
    lower_words = {"en", "de", "del", "la", "las", "los", "y", "a"}
    parts = [p.capitalize() if p.lower() not in lower_words else p.lower() for p in s.split(" ")]
    # Preserve acronyms
    out = " ".join(parts)
    out = out.replace("Unc", "UNC")
    return out


def get_unc_dataset() -> Tuple[str, Dict[str, List[str]]]:
    uni = "Universidad Nacional de Córdoba (UNC)"

    faculties: Dict[str, List[str]] = {
        "Facultad de Arquitectura, Urbanismo y Diseño": [
            "Arquitectura",
            "Diseño Industrial",
            "Licenciatura en Diseño del Paisaje",
        ],
        "Facultad de Artes": [
            "Licenciatura en Artes Visuales",
            "Licenciatura en Cine y Artes Audiovisuales",
            "Licenciatura en Composición Musical",
            "Licenciatura en Dirección Coral",
            "Licenciatura en Educación Musical",
            "Licenciatura en Teatro",
            "Profesorado en Educación Musical",
            "Profesorado en Educación Plástica y Visual",
            "Profesorado de Teatro",
            "Tecnicatura en Artes Escenotécnicas",
            "Técnico Productor en Medios Audiovisuales",
        ],
        "Facultad de Ciencias Agropecuarias": [
            "Ingeniería Agronómica",
            "Ingeniería Zootecnista",
            "Licenciatura en Agroalimentos",
            "Licenciatura en Diseño del Paisaje",
            "Tecnicatura en Jardinería y Floricultura",
            "Tecnicatura Universitaria en Agroalimentos",
        ],
        "Facultad de Ciencias de la Comunicación": [
            "Licenciatura en Comunicación Social",
            "Profesorado Universitario en Comunicación Social",
            "Tecnicatura Universitaria en Gestión de la Comunicación Turística",
            "Tecnicatura Universitaria en Periodismo Deportivo",
            "Tecnicatura en Comunicación Social",
        ],
        "Facultad de Ciencias Económicas": [
            "Contador Público",
            "Licenciatura en Administración",
            "Licenciatura en Economía",
        ],
        "Facultad de Ciencias Exactas, Físicas y Naturales": [
            "Biología",
            "Constructor",
            "Geología",
            "Ingeniería Aeroespacial",
            "Ingeniería Ambiental",
            "Ingeniería Biomédica",
            "Ingeniería Civil",
            "Ingeniería Electromecánica",
            "Ingeniería Electrónica",
            "Ingeniería en Agrimensura",
            "Ingeniería en Computación",
            "Ingeniería Industrial",
            "Ingeniería Mecánica",
            "Ingeniería Química",
            "Licenciatura en Hidrometeorología",
            "Profesorado en Ciencias Biológicas",
            "Tecnicatura en Mecánica Electricista",
            "Tecnicatura Universitaria en Análisis Químico Industrial",
            "Tecnicatura Universitaria en Sistemas Digitales",
        ],
        "Facultad de Ciencias Médicas": [
            "Licenciatura en Enfermería",
            "Licenciatura en Fonoaudiología",
            "Licenciatura en Kinesiología y Fisioterapia",
            "Licenciatura en Nutrición",
            "Licenciatura en Producción de Bioimágenes",
            "Medicina",
            "Tecnicatura en Enfermería",
            "Tecnicatura en Laboratorio Clínico e Histopatología",
        ],
        "Facultad de Ciencias Químicas": [
            "Bioquímica",
            "Farmacia",
            "Licenciatura en Biotecnología",
            "Licenciatura en Química",
        ],
        "Facultad de Ciencias Sociales": [
            "Licenciatura en Ciencia Política",
            "Licenciatura en Sociología",
            "Licenciatura en Trabajo Social",
        ],
        "Facultad de Derecho": [
            "Abogacía",
            "Notariado",
            "Profesorado en Ciencias Jurídicas",
            "Tecnicatura Superior Universitaria en Asistencia en Investigación Penal",
        ],
        "Facultad de Filosofía y Humanidades": [
            "Bibliotecólogo",
            "Licenciatura en Ciencias de la Educación",
            "Licenciatura en Antropología",
            "Licenciatura en Archivología",
            "Licenciatura en Bibliotecología y Documentación",
            "Licenciatura en Filosofía",
            "Licenciatura en Geografía",
            "Licenciatura en Historia",
            "Licenciatura en Letras Clásicas",
            "Licenciatura en Letras Modernas",
            "Profesorado de Ciencias de la Educación",
            "Profesorado en Filosofía",
            "Profesorado Universitario en Geografía",
            "Profesorado en Historia",
            "Profesorado en Letras Clásicas",
            "Profesorado en Letras Modernas",
            "Tecnicatura en Corrección Literaria",
            "Técnico Profesional Archivero",
        ],
        "Facultad de Lenguas": [
            "Licenciatura en Español Lengua Materna y Lengua Extranjera",
            "Licenciatura en Lengua y Literatura Inglesa, Francesa, Italiana o Alemana",
            "Profesorado de Español Lengua Materna y Lengua Extranjera",
            "Profesorado de Lengua Inglesa, Francesa, Italiana o Alemana",
            "Profesorado de Portugués",
            "Traductorado Público Nacional (Inglés, Francés, Italiano o Alemán)",
        ],
        "Facultad de Matemática, Astronomía, Física y Computación": [
            "Analista en Computación",
            "Licenciatura en Astronomía",
            "Licenciatura en Ciencias de la Computación",
            "Licenciatura en Hidrometeorología",
            "Licenciatura en Física",
            "Licenciatura en Matemática",
            "Licenciatura en Matemática Aplicada",
            "Profesorado en Física",
            "Profesorado de Matemática",
            "Tecnicatura Universitaria en Astronomía",
            "Tecnicatura Universitaria en Matemática Aplicada",
        ],
        "Facultad de Odontología": ["Odontología"],
        "Facultad de Psicología": [
            "Licenciatura en Psicología",
            "Profesorado de Psicología",
            "Tecnicatura en Acompañamiento Terapéutico",
        ],
        # Institutions (stored as special faculties, UX will label them as "Institución")
        "Colegio Nacional de Monserrat": [
            "Comunicación Visual",
            "Martillero y Corredor Público",
            "Tecnicatura Superior en Bromatología",
        ],
        "Escuela Superior de Comercio Manuel Belgrano": [
            "Analista Universitario de Sistemas de Informática",
            "Tecnicatura Superior Universitaria en Administración de Cooperativa y Mutuales",
            "Tecnicatura Superior Universitaria en Comercialización",
            "Tecnicatura Superior Universitaria en Gestión Financiera",
            "Tecnicatura Superior Universitaria en Recursos Humanos",
        ],
    }

    # Normalize
    uni = _norm(uni)
    out: Dict[str, List[str]] = {}
    for fac, cars in faculties.items():
        out[_norm(fac)] = [_norm(c) for c in cars]
    return uni, out


def seed_unc(session) -> dict:
    """Idempotent seed: inserts UNC + faculties + careers if missing."""
    uni_name, data = get_unc_dataset()

    u = session.execute(select(University).where(University.name == uni_name)).scalar_one_or_none()
    if not u:
        u = University(name=uni_name)
        session.add(u)
        session.flush()

    created_fac = 0
    created_car = 0

    # Map existing faculties by (university_id, name)
    existing_fac = {
        (f.university_id, f.name): f
        for f in session.execute(select(Faculty).where(Faculty.university_id == u.id)).scalars().all()
    }

    for fac_name, careers in data.items():
        f = existing_fac.get((u.id, fac_name))
        if not f:
            f = Faculty(name=fac_name, university_id=u.id)
            session.add(f)
            session.flush()
            existing_fac[(u.id, fac_name)] = f
            created_fac += 1

        # Existing careers
        existing_car = {
            c.name
            for c in session.execute(select(Career).where(Career.faculty_id == f.id)).scalars().all()
        }
        for car_name in careers:
            if car_name in existing_car:
                continue
            session.add(Career(name=car_name, faculty_id=f.id))
            created_car += 1

    return {
        "university": uni_name,
        "created_faculties": created_fac,
        "created_careers": created_car,
    }

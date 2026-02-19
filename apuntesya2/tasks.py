from wsgi import app  # o desde donde exportes tu app Flask
from your_db_module import Session, Note  # ajustá imports
# importa generate_note_preview, etc.

def generate_preview_job(note_id: int):
    """
    Background job: generate preview for note_id and persist.
    """
    with app.app_context():
        with Session() as s:
            note = s.get(Note, int(note_id))
            if not note:
                return

            try:
                if hasattr(note, "preview_status"):
                    note.preview_status = "running"
                if hasattr(note, "preview_error"):
                    note.preview_error = None
                s.commit()

                pages, imgs = generate_note_preview(note)
                if imgs:
                    note.preview_pages = {"pages": pages}
                    note.preview_images = {"images": imgs}
                    if hasattr(note, "preview_status"):
                        note.preview_status = "done"
                    s.commit()
                else:
                    if hasattr(note, "preview_status"):
                        note.preview_status = "failed"
                    if hasattr(note, "preview_error"):
                        note.preview_error = "No se pudieron generar imágenes."
                    s.commit()

            except Exception as e:
                if hasattr(note, "preview_status"):
                    note.preview_status = "failed"
                if hasattr(note, "preview_error"):
                    note.preview_error = str(e)[:500]
                s.commit()
                raise

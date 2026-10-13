from pathlib import Path
 
from fastapi.responses import FileResponse
 
from src.app.main import app
 
PAGINA = Path(__file__).parent / "static" / "index.html"
 
# Quita la redirección original de "/" hacia /docs y sirve la página en su lugar.
app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
 
 
@app.get("/", include_in_schema=False)
def inicio() -> FileResponse:
    return FileResponse(PAGINA)
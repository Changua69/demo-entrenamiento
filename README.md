# Predicción de popularidad de canciones

Taller en dos sprints para una disquera que debe decidir, **antes del lanzamiento**, en qué canciones invertir promoción.

- **Sprint 1, regresión:** estimar la popularidad (0 a 100) de una canción nueva y recomendar si se promociona.
- **Sprint 2, clasificación binaria:** decidir la promoción según lo que cuesta equivocarse.
- **Link de Render:**
https://demo-reg-clas-canciones1-bg3d.onrender.com/

Cada sprint cierra con un incremento en producción: un servicio FastAPI publicado en Render que lee un único archivo, `models/modelo.pkl`, con el modelo entrenado y su ficha.

> Universidad Central, Ingeniería de Sistemas, 2026-2. Autor: **[Nombre del estudiante]**

---

## Resumen de resultados

Evaluación sobre las 415 canciones de 2020-2025, que no intervienen en ninguna decisión de entrenamiento.

| | Sprint 1: regresión | Sprint 2: clasificación |
|---|---|---|
| Objetivo del modelo | `popularidad` (0 a 100) | `es_hit` (1 si la popularidad llega a 70) |
| Variables excluidas | `reproducciones_sem1`, `es_hit` | `reproducciones_sem1`, `popularidad` |
| Modelo de referencia | Regresión lineal con duración² | Regresión logística |
| **Modelo publicado** | **Ridge con duración²** | **Gradient boosting con umbral por costos** |
| Resultado del modelo publicado | MAE 5,76; RMSE 7,16; R² 0,718; sesgo +0,27 | Costo 114; precisión 0,636; recall 0,792 |
| Línea base | MAE 10,44 (media por género) | Costo 318 (nunca promocionar); 309 (siempre promocionar) |
| Resultado de referencia | MAE 5,75 | Costo 99 |
| Regla de decisión | Promocionar si la popularidad esperada supera 65 | Promocionar si la probabilidad de hit supera 0,22 |
| Código de entrenamiento | `src/entrenar_regresion.py` | `src/entrenar_clasificacion.py` |

En clasificación, la **regresión logística** (el modelo de referencia) obtuvo un costo menor que el gradient boosting: 99 frente a 114. Ambos se documentan en el registro comparativo de más abajo.

---

## Datos

- `data/canciones.csv`: 2.000 canciones **sintéticas** lanzadas entre 1995 y 2025.
- **Entrenamiento:** 1.585 canciones de 1995 a 2019.
- **Prueba:** 415 canciones de 2020 a 2025, de las cuales 106 (25,5 %) fueron hit.
- Columnas de entrada disponibles antes del lanzamiento: `bailabilidad`, `energia`, `valencia`, `acustica`, `tempo`, `duracion_min`, `volumen_db`, `anio`, `colaboracion` y `genero` (pop, urbano, rock, electrónica, indie).

---

## Metodología

### 1. Momento de la predicción y control de fuga

La predicción se necesita antes del lanzamiento, así que solo son legítimas las variables que existen en ese momento. Cada columna se revisó contra esa regla:

| Columna | ¿Existe antes de lanzar? | Decisión | Razón |
|---|---|---|---|
| Atributos de audio, `anio`, `colaboracion`, `genero` | Sí | Entra | Se conocen al decidir |
| `reproducciones_sem1` | No | Fuera | Ocurre una semana **después** del lanzamiento (fuga: posterior al evento) |
| `es_hit` | No | Fuera en regresión | Se calcula con el objetivo (popularidad ≥ 70) |
| `popularidad` | No | Fuera en clasificación | Define `es_hit`; usarla equivale a conocer la respuesta |

Las dos columnas con fuga tienen una correlación muy alta con el objetivo (0,99 y 0,67 en entrenamiento). Una correlación casi perfecta no es un hallazgo: es una pregunta sobre cuándo y cómo se calculó la columna.

### 2. Partición temporal

Se entrena con el pasado (1995-2019) y se prueba con el futuro (2020-2025) **una sola vez**. Una partición aleatoria mezclaría épocas y podría dar un error más optimista que el real en uso.

Las decisiones internas (el `alpha` de Ridge y el umbral por costos) se toman dentro del periodo de entrenamiento con `TimeSeriesSplit` (ventana creciente), de modo que 2020-2025 no interviene en ninguna elección.

### 3. Pipeline

Todo el preprocesamiento vive dentro del pipeline, que recibe las 10 columnas crudas:

```
columnas crudas → derivar (duración²) → codificar y escalar → estimador → predicción
```

- `duracion_c2 = (duracion_min − 3,5)²`: término cuadrático centrado en 3,5 minutos, definido en `src/variables.py`.
- `OneHotEncoder` para `genero` y `StandardScaler` para las numéricas, ajustados solo con entrenamiento.
- Cambiar de algoritmo es cambiar el último paso del pipeline; el resto se conserva.

### 4. El artefacto

`models/modelo.pkl` contiene un diccionario con el pipeline entrenado, la **ficha** del modelo (qué predice, con qué datos, qué error tiene y qué decisión sostiene) y las columnas de entrada. El servicio no cambia cuando cambia el modelo, porque las decisiones viajan en la ficha.

---

## Sprint 1: regresión

**Pregunta:** ¿qué popularidad alcanzará la canción?

**Modelo publicado:** Ridge con duración², con `RidgeCV` y `alpha` elegido por validación cruzada temporal (5 pliegues, métrica MAE). Resultado: `alpha = 3,556`.

### Resultado (2020-2025)

| Métrica | Valor |
|---|---|
| MAE | 5,76 |
| RMSE | 7,16 |
| R² | 0,718 |
| Sesgo (real − esperado) | +0,27 |
| MAE de la línea base (media por género) | 10,44 |

El error baja de 10,44 a 5,76 puntos, una mejora de cerca del 45 % frente a la línea base. El R² es la fracción de la varianza explicada, no un porcentaje de aciertos.

### Desempeño por género

Un sesgo positivo indica que el modelo **subestima**; uno negativo, que **sobreestima**.

| Género | MAE | Sesgo | Lectura |
|---|---|---|---|
| Urbano | 6,04 | +0,48 | Mayor error |
| Pop | 5,95 | +1,35 | Subestima más de un punto |
| Indie | 5,70 | −0,93 | Sobreestima |
| Electrónica | 5,64 | −0,11 | Sin sesgo relevante |
| Rock | 5,15 | −0,73 | Menor error |

El sesgo global (+0,27) es pequeño y esconde diferencias por segmento. Con un corte de 65, la subestimación en pop puede dejar sin promoción a canciones que sí llegarían a 70, y la sobreestimación en indie puede promocionar canciones que no despegan.

### Coeficientes principales (variables estandarizadas)

| Variable | Coeficiente |
|---|---|
| `bailabilidad` | +7,21 |
| `energia` | +6,01 |
| género urbano | +5,30 |
| género indie | −5,24 |
| género rock | −3,01 |

Describen asociaciones en datos sintéticos, no causas.

### Regla de decisión

**Promocionar si la popularidad esperada supera 65.** El corte lo fija quien responde por el presupuesto, no el modelo.

---

## Sprint 2: clasificación binaria

**Pregunta:** ¿la canción será un hit (popularidad ≥ 70)?

**Modelo publicado:** `HistGradientBoostingClassifier` (profundidad 3, tasa de aprendizaje 0,05, 200 iteraciones, `min_samples_leaf=20`, `l2_regularization=1,0`, sin parada temprana) con umbral elegido por costos.

### Costos de los errores

| Error | Costo |
|---|---|
| Falso negativo (dejar sin promoción una canción que sí fue hit) | 3 |
| Falso positivo (promocionar una canción que no fue hit) | 1 |

Con probabilidades bien calibradas, el umbral óptimo teórico es 1 / (1 + 3) = 0,25. El umbral se elige minimizando el costo con probabilidades fuera de muestra dentro de 1995-2019 (`TimeSeriesSplit`, 4 pliegues).

### Comparación de clasificadores (2020-2025)

| | Logística (referencia) | Gradient boosting (publicado) | Nunca promocionar | Siempre promocionar |
|---|---|---|---|---|
| **Costo** | **99** | 114 | 318 | 309 |
| Umbral | 0,27 | 0,22 | n/d | n/d |
| Precisión | 0,688 | 0,636 | n/d | n/d |
| Recall | 0,811 | 0,792 | n/d | n/d |
| Verdaderos positivos | 86 | 84 | 0 | 106 |
| Falsos positivos | 39 | 48 | 0 | 309 |
| Falsos negativos | 20 | 22 | 106 | 0 |
| Verdaderos negativos | 270 | 261 | 309 | 0 |
| Canciones promocionadas (de 415) | 125 | 132 | 0 | 415 |

- El costo se verifica a mano: logística 3 × 20 + 39 = 99; boosting 3 × 22 + 48 = 114; "nunca" 3 × 106 = 318.
- Como el 25,5 % de las canciones de prueba fueron hit, "siempre promocionar" (309) casi empata con "nunca" (318): con costos 3 a 1 se igualan en 25 % de hits. Por eso la línea base exigente es 309, no solo 318.
- Ambos modelos superan las dos líneas base. El boosting reduce el costo en un 64 % frente a "nunca promocionar"; la logística, en un 69 %.
- La logística gana en costo, precisión y recall. Los árboles no extrapolan la tendencia del año (toda canción posterior a 2019 se trata como 2019), mientras que la logística sí.

### Regla de decisión

**Promocionar si la probabilidad de hit supera 0,22** (umbral del modelo publicado). Los costos de cada error los fija quien responde por el presupuesto.

---

## Registro comparativo de regresión

Todas las corridas se evalúan sobre las mismas 415 canciones de 2020-2025 y se conservan, incluidas las descartadas.

| Corrida | MAE | Origen | Lectura |
|---|---|---|---|
| Media global del entrenamiento | 11,06 | Material del curso | Error de no tener modelo |
| Media por género | 10,44 | Medido | Línea base |
| Lineal con duración² | 5,75 | Material del curso | Modelo de referencia |
| **Ridge con duración²** | **5,76** | Medido | **Publicado** |
| Lasso con duración² | 5,79 | Medido | `alpha` = 0,057; descartó `tempo` y el indicador de género electrónica |
| Gradient boosting | 6,63 | Material del curso | Subestima; no extrapola el año |
| Random Forest | 6,95 | Material del curso | Más complejo no es mejor |

Ridge y Lasso quedan prácticamente iguales a la regresión lineal: con 1.585 canciones y 10 variables, la regularización aporta poco. Que Lasso haya llevado a cero el indicador de electrónica no significa que el género no importe: las indicadoras de género suman 1, así que su efecto pasa al intercepto.

---

## Cómo reproducirlo

Requiere Python 3.11, 3.12 o 3.13 (no 3.14 ni 3.10 o anteriores) y las versiones de `requirements.txt` (scikit-learn 1.6.1).

```bash
git clone <URL-de-este-repositorio>
cd <carpeta-del-repositorio>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Entrenar (siempre con `python -m`, desde la raíz, para que el servicio pueda cargar el pickle):

```bash
python -m src.entrenar_regresion       # Sprint 1: escribe models/modelo.pkl
python -m src.entrenar_clasificacion   # Sprint 2: escribe models/modelo.pkl
```

Cada entrenamiento **sobrescribe** `models/modelo.pkl`. Para guardar en otra ruta sin pisar el entregable (cmd de Windows):

```bat
set RUTA_MODELO=models/modelo_prueba.pkl
python -m src.entrenar_clasificacion
set RUTA_MODELO=
```

Comprobar:

```bash
pytest -q                            # 9 pruebas
uvicorn src.app.main:app --reload    # http://127.0.0.1:8000/docs
python -m src.probar_servicio        # con el servicio arriba, en otra terminal
```

Inspeccionar la ficha del modelo guardado:

```python
from src.artefacto import cargar

f = cargar()["ficha"]
print(f["nombre"], f["tarea"], f["desempeno"], f["decision"])
```

---

## API

| Método | Ruta | Respuesta |
|---|---|---|
| `GET` | `/docs` | Documentación interactiva |
| `GET` | `/salud` | Estado del servicio y modelo cargado |
| `GET` | `/modelo` | Ficha del modelo |
| `POST` | `/predecir` | Predicción y decisión para una canción |
| `POST` | `/predecir_lote` | Lo mismo para hasta 1.000 canciones |

Ejemplo de entrada para `/predecir`:

```json
{"bailabilidad": 0.72, "energia": 0.65, "valencia": 0.55, "acustica": 0.12, "tempo": 118,
 "duracion_min": 3.4, "volumen_db": -6.5, "anio": 2026, "colaboracion": 1, "genero": "urbano"}
```

Una entrada con un campo que no existe antes del lanzamiento, como `reproducciones_sem1`, recibe `422`. Ese rechazo es intencional y es parte de la prueba de humo.

---

## Despliegue y verificación

| Sprint | Modelo publicado | URL | Verificación |
|---|---|---|---|
| 1 | Ridge con duración² | `https://demo-reg-clas-canciones-1y5r.onrender.com` | `probar_servicio`: "servicio en orden" |
| 2 | Gradient boosting con umbral por costos | `https://demo-reg-clas-canciones1-bg3d.onrender.com` | `probar_servicio`: "servicio en orden" (2026-10-18) |

La prueba de humo comprueba cuatro cosas: el servicio responde, la ficha está completa, la predicción de ejemplo es coherente y se rechazan las variables posteriores al lanzamiento.

```bash
python -m src.probar_servicio https://su-servicio.onrender.com
```

En el plan gratuito de Render el servicio se suspende tras 15 minutos sin uso y la primera petición puede tardar cerca de un minuto.

---

## Limitaciones

- **Datos sintéticos.** Los resultados describen asociaciones en un conjunto simulado; no son evidencia causal ni se pueden extrapolar a un catálogo real.
- **Sin información del artista.** Las columnas de entrada no incluyen artista ni historial, que en la práctica suele ser lo más predictivo.
- **Extrapolación del año.** `anio` entra como variable numérica y el servicio recibe años posteriores a 2019. Los modelos lineales extienden la tendencia; los árboles, no.
- **Probabilidades sin calibrar.** El umbral por costos depende de las probabilidades del modelo; con un modelo mal calibrado, el umbral óptimo se desplaza.
- **Sesgo por segmento.** El modelo de regresión subestima pop y sobreestima indie y rock; la decisión de promoción debe leerse con ese sesgo en mente.
- **Los costos y el corte los fija quien responde por el presupuesto.** El modelo es un insumo; la decisión es del negocio.

---

## Estructura del repositorio

```
├── data/canciones.csv              # 2.000 canciones sintéticas, 1995-2025
├── models/modelo.pkl               # Modelo entrenado y su ficha
├── src/
│   ├── variables.py                # Columnas de entrada, variables derivadas y partición temporal
│   ├── artefacto.py                # guardar() y cargar() del modelo
│   ├── entrenar_regresion.py       # Sprint 1
│   ├── entrenar_clasificacion.py   # Sprint 2
│   ├── probar_servicio.py          # Prueba de humo contra el servicio
│   └── app/                        # Servicio FastAPI
├── tests/test_servicio.py          # Pruebas
├── requirements.txt                # Dependencias con versiones fijas
└── render.yaml                     # Configuración para Render
```

---

## Uso de un asistente de IA

Parte del código de entrenamiento (variantes Ridge y Lasso, gradient boosting) y de la documentación se elaboró con apoyo de un asistente de IA. Los resultados de este documento provienen de las ejecuciones locales descritas arriba, no de las respuestas del asistente. Las cifras marcadas como "Material del curso" en el registro comparativo no se midieron en este repositorio.

---

## Referencias

- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4).
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9).
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3.ª ed.). OTexts.
- Scikit-learn Developers. *User Guide*: Pipeline, Common pitfalls (data leakage), Metrics and scoring.

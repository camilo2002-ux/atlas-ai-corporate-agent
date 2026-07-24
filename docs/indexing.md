# Indexación vectorial de Atlas

## Objetivo

Convertir los chunks producidos por el pipeline documental en embeddings y almacenarlos en una colección persistente para realizar búsquedas semánticas con filtros de metadatos.

## Componentes

- **FastEmbed:** genera embeddings locales mediante ONNX.
- **Modelo predeterminado:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Chroma:** almacena texto, embeddings y metadatos en una colección persistente.
- **Manifiesto:** registra proveedor, modelo, dimensión y origen de cada construcción del índice.

## Regla de consistencia

Los documentos y las consultas deben utilizar exactamente el mismo proveedor y modelo. `search_index.py` valida el manifiesto antes de consultar la colección.

## Flujo

```text
chunks.jsonl
    ↓
validación por línea
    ↓
embeddings en lotes
    ↓
normalización de vectores
    ↓
normalización de metadatos
    ↓
upsert en Chroma
    ↓
index-manifest.json
```

## Índice de demostración

```bash
python scripts/index_chunks.py \
  --input tests/fixtures/demo-chunks.jsonl \
  --reset
```

La primera ejecución descarga el modelo local y lo conserva en `.cache/fastembed`.

## Búsqueda

```bash
python scripts/search_index.py \
  "¿Cuántos días de vacaciones recibe un colaborador nuevo?" \
  --top-k 3
```

Filtro por categoría:

```bash
python scripts/search_index.py \
  "¿Cuál es el límite de alojamiento?" \
  --category Finanzas
```

## Pipeline real

Después de cargar los documentos oficiales:

```bash
python scripts/process_documents.py
python scripts/index_chunks.py --reset
```

No se debe cambiar el modelo sin reconstruir completamente el índice mediante `--reset`.

## Datos generados

La base persistente de Chroma, el manifiesto y los chunks procesados se generan localmente y no se versionan en Git. Los documentos ficticios originales sí podrán versionarse más adelante.

## Proveedor de pruebas

`--provider hashing` existe para pruebas rápidas sin descargar un modelo. No proporciona una búsqueda semántica de calidad y no debe utilizarse en la versión final.

# Procesamiento y extracción de contenido

Atlas transforma archivos corporativos en fragmentos limpios y trazables antes de generar embeddings.

## Formatos soportados

| Formato | Biblioteca | Unidad lógica y ubicación |
|---|---|---|
| PDF | pypdf | Página |
| DOCX | python-docx | Sección y tablas |
| XLSX | openpyxl | Hoja y fila |
| PPTX | python-pptx | Diapositiva y notas del orador |
| Markdown | Python y expresiones regulares | Sección |
| CSV | csv de Python | Fila |
| JSON | json de Python | Ruta JSON |
| HTML | Beautiful Soup | Sección y tablas |

## Flujo

1. Validación del formato y existencia del archivo.
2. Extracción mediante el adaptador correspondiente.
3. Limpieza de espacios, caracteres invisibles y numeración repetitiva.
4. División por unidad lógica y ventanas de hasta 1200 caracteres.
5. Superposición de 180 caracteres para conservar contexto.
6. Incorporación de metadatos del archivo y del inventario documental.
7. Creación de un identificador estable para cada chunk.
8. Escritura de los chunks en JSONL y de un reporte de ejecución en JSON.

## Metadatos

Cada chunk puede incluir:

- `source_file`
- `file_type`
- `file_size_bytes`
- `last_modified_utc`
- `sha256`
- `document_id`
- `category`
- `owner`
- `version`
- `status`
- página, sección, hoja, fila, diapositiva o ruta JSON
- `chunk_index`
- `unit_index`
- `part_index`

## PDF escaneado y OCR

La primera versión extrae texto de PDFs digitales. Cuando una página contiene poco o ningún texto, el pipeline añade una advertencia indicando que probablemente requiere OCR. El OCR se incorporará como adaptador opcional para evitar dependencias de sistema innecesarias en el MVP.

## Ejecución

```bash
python scripts/process_documents.py
```

Archivos producidos:

- `data/processed/chunks.jsonl`
- `data/processed/processing-report.json`

## Pruebas

```bash
pytest
```

Las pruebas generan temporalmente archivos de los ocho formatos y verifican que todos produzcan chunks con metadatos de ubicación.

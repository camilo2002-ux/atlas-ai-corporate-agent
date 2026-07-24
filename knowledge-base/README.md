# Base de conocimiento de NovaCommerce

Esta carpeta contiene los documentos internos ficticios utilizados por Atlas, un agente corporativo de inteligencia artificial basado en arquitectura RAG.

> Todos los nombres, políticas, cifras, empleados, procedimientos y demás datos de esta base de conocimiento son ficticios y fueron creados exclusivamente con fines educativos.

## Empresa ficticia

**NovaCommerce** es una empresa ficticia dedicada al comercio electrónico y a la distribución de productos mediante centros logísticos regionales.

El agente Atlas permitirá que los colaboradores consulten información de Recursos Humanos, Finanzas, Operaciones, Tecnología, Estrategia y Legal desde una interfaz conversacional.

## Fuentes documentales

Durante la primera versión del proyecto, los documentos serán generados específicamente para el desafío y cargados manualmente en el repositorio.

La fuente inicial será:

* Documentos ficticios creados para el proyecto.
* Archivos almacenados dentro del repositorio.
* Carga manual desde la interfaz del agente.

En una versión posterior, los documentos podrán almacenarse y actualizarse mediante Oracle Cloud Infrastructure Object Storage.

## Categorías

| Categoría        | Contenido                                    | Responsable ficticio      |
| ---------------- | -------------------------------------------- | ------------------------- |
| Recursos Humanos | Beneficios, onboarding y políticas laborales | Líder de Recursos Humanos |
| Finanzas         | Gastos, presupuestos y límites               | Líder Financiero          |
| Operaciones      | Centros de distribución y procedimientos     | Líder de Operaciones      |
| Tecnología       | Incidentes técnicos, sistemas y APIs         | Líder de Tecnología       |
| Estrategia       | Roadmaps, objetivos y prioridades            | Dirección Estratégica     |
| Legal            | Privacidad, cumplimiento y condiciones       | Líder Jurídico            |

## Formatos planificados

La base de conocimiento tendrá inicialmente ocho documentos:

1. PDF: política de beneficios laborales.
2. DOCX: manual de onboarding.
3. XLSX: política y límites de gastos.
4. PPTX: roadmap estratégico 2026–2027.
5. Markdown: procedimiento de incidentes técnicos.
6. CSV: catálogo de centros de distribución.
7. JSON: catálogo interno de APIs.
8. HTML: política de privacidad.

## Reglas de curaduría

Solo se incorporarán documentos que cumplan las siguientes condiciones:

* El documento tiene una finalidad clara.
* La categoría y el responsable están definidos.
* El documento posee un identificador único.
* Se conoce su versión.
* No es un borrador.
* No está duplicado.
* No contiene contraseñas, tokens ni claves privadas.
* No contiene información personal real.
* Su contenido es relevante para preguntas de colaboradores.
* Su información no contradice otros documentos vigentes sin una explicación.

## Convención de nombres

Los archivos utilizarán:

* Letras minúsculas.
* Palabras separadas mediante guiones.
* Nombres descriptivos.
* Extensión correspondiente al formato.

Ejemplo:

`politica-beneficios-laborales.pdf`

## Control de versiones

Cada documento tendrá una versión identificada mediante el formato:

`vMAYOR.MENOR`

Ejemplos:

* `v1.0`: primera versión oficial.
* `v1.1`: corrección o actualización menor.
* `v2.0`: cambio importante en el contenido.

Cuando exista más de una versión, solamente la versión vigente deberá ser utilizada por el agente.

## Propiedad de los documentos

Cada documento tendrá un responsable ficticio encargado de:

* Aprobar su incorporación.
* Verificar que la información sea correcta.
* Autorizar las actualizaciones.
* Indicar cuándo debe retirarse o sustituirse.
* Resolver posibles contradicciones.

## Acceso

El agente estará disponible para todos los colaboradores ficticios de NovaCommerce.

Todos los documentos incluidos en esta demostración serán de lectura general. No se utilizarán documentos confidenciales, credenciales ni información personal real.

## Método de ingesta

La ingesta se implementará en dos fases.

### Fase inicial

1. Generación de documentos ficticios.
2. Almacenamiento en el repositorio.
3. Carga manual mediante la aplicación.
4. Extracción y normalización del contenido.
5. División del contenido en fragmentos.
6. Generación de embeddings.
7. Indexación en la base vectorial.

### Fase futura

1. Almacenamiento en OCI Object Storage.
2. Detección de documentos nuevos o modificados.
3. Procesamiento incremental.
4. Actualización automática del índice vectorial.

## Criterios de aceptación

Un documento podrá ingresar a la base cuando:

* Aparezca en el inventario documental.
* Tenga categoría y responsable.
* Posea versión y estado vigente.
* Pueda ser leído por el extractor correspondiente.
* No contenga datos sensibles reales.
* Incluya información suficiente para construir preguntas de prueba.

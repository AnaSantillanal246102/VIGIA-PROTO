# Fuentes de video para el prototipo

**Verificar antes de usar: condiciones de uso, clave de API y que el servicio siga activo.**
Los catalogos cambian. Ultima revision de esta lista: 17-oct-2026.

| Fuente | Tipo | Clave | Notas |
|---|---|---|---|
| Caltrans CCTV (California) | JPG por URL; algunas con HLS | No (segun su documentacion) | Campos `currentImageURL` y `streamingVideoURL`. Mejor candidata para empezar. `python -m pipeline.discover` |
| Departamentos de transito de EE. UU. (NYC DOT, WSDOT, FL511, etc.) | JPG; algunos HLS | Varias piden clave gratuita (p. ej. 511 NY) | Cada estado tiene sus propias condiciones |
| Ontario 511 / Alberta 511 (Canada) | JPG | Clave gratuita | API de desarrolladores |
| Windy Webcams API v3 | JPG con URL que caduca (10-15 min en plan gratuito) | Si (gratuita) | Solo imagenes pequenas en el plan gratuito, se actualizan cada 5-15 min y exige atribucion. **Poco util para placas**; sirve para mostrar variedad de escenas |
| TrafficLand (via FirstNet) | JPG y streaming | Acceso por portal | Verificar condiciones de acceso |
| Mexico | — | — | No encontramos catalogos publicos de camaras de transito con API abierta. Dejarlo como propuesta de alianza con el municipio |

Notas tecnicas
- Casi todas las camaras publicas dan **una imagen JPG que se actualiza cada pocos segundos**, no video continuo. `type: snapshot` cubre ese caso.
- Intervalo minimo por camara: 2 s (no saturar servidores publicos). Para el prototipo bastan 5-10 s.
- Claves de API: variable de entorno y `${NOMBRE}` en la configuracion. Nunca en git.
- Las placas de EE. UU. no se parecen a las mexicanas: sirven para probar la **mejora de imagen**, no para medir la lectura de placas mexicanas.
- No publiquen en el video ni la presentacion imagenes donde se lean placas o se vean rostros reales; difuminenlos.

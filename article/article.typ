#import "@preview/versatile-apa:7.2.0": (
  abstract-page, appendix, appendix-outline, title-page,
  versatile-apa as apa-style,
)
#import "@preview/orchid:0.1.0": (
  generate-link as orcid-link, logo-icon as orcid-logo,
)

#set text(lang: "es")

#set document(
  title: [Segmentación de solicitantes: perfiles interpretables con clustering],
)

#show: apa-style.with(
  font-size: 12pt,
  running-head: "",
)

#title-page(
  authors: (
    (
      name: [Harrizon Soler],
      affiliations: (
        "ID-1",
      ),
    ),
  ),
  affiliations: (
    "ID-1": [Universidad Santo Tomás],
  ),

  course: [Machine Learning],
  instructor: [Carlos Andres Sepulveda Leon],
  due-date: datetime(day: 15, month: 11, year: 2026).display(),
)


El laboratorio es accesible en #link("https://activity-3-smote-harrysoler.surge.sh/")[https://activity-3-smote-harrysoler.surge.sh/]. Los archivos requeridos se encuentran en #link("https://github.com/harrysoler/smote_with_randomforest")[GitHub] ya que no funcionaba OneDrive en el momento.

== Explicación global del modelo

El modelo entrenado es un clasificador de bosque aleatorio (Random Forest) cuya tarea es predecir, a partir de las características de un crédito y de su solicitante, si el préstamo terminará en incumplimiento (`Status` = 1) o se pagará normalmente (`Status` = 0). El conjunto de datos contiene cerca de 148.000 registros de créditos, de los cuales se reservó el 20 % como conjunto de prueba (29.734 registros) mediante una partición estratificada con semilla fija, de modo que ambos subconjuntos conservan la misma proporción de incumplimientos (aproximadamente 75 % de clase 0 y 25 % de clase 1, un desbalance moderado de 3:1).

Antes del clasificador, un `ColumnTransformer` prepara las variables: las numéricas (monto, tasa de interés, plazo, valor de la propiedad, ingreso, puntaje crediticio, LTV y relación deuda-ingreso) se imputan con la mediana y se estandarizan; las categóricas (aprobación anticipada, propósito, solvencia, tipo de ocupación, entre otras) se imputan con la moda y se codifican con one-hot. Todo este preprocesamiento se ajusta únicamente con los datos de entrenamiento, lo que evita fuga de información hacia el conjunto de prueba.

Sobre esa base se compararon dos variantes:

1. *Sin tratamiento del desbalance:* el bosque se entrena con los datos tal como vienen.
2. *Con SMOTE:* dentro del mismo pipeline, y solo durante el entrenamiento, se generan ejemplos sintéticos de la clase minoritaria (incumplimientos) hasta equilibrar las dos clases.

La lectura es la siguiente: SMOTE aumenta el recall de la clase de interés del 85 % al 92 %, es decir, el modelo pasa de detectar 85 de cada 100 incumplimientos reales a detectar 92. A cambio, la precisión baja del 88 % al 81 %: de cada 100 alertas que emite, más resultan ser falsas alarmas. El F1 y el accuracy prácticamente no cambian, lo que confirma que SMOTE no hace al modelo "mejor" en abstracto, sino que desplaza su punto de equilibrio hacia la detección. Para un problema de riesgo crediticio, donde omitir un incumplimiento real suele costar más que revisar una falsa alarma, este intercambio es generalmente favorable. Cabe notar que el efecto es moderado precisamente porque el desbalance original (3:1) es leve; en problemas con desbalances de 10:1 o mayores, el tratamiento suele ser imprescindible.

== Dos casos individuales

Los siguientes perfiles son ilustrativos: se construyeron a partir de rangos plausibles del dataset para explicar el comportamiento del modelo, y no corresponden a registros reales específicos.

=== Caso A: un crédito de bajo riesgo, correctamente aprobado

Camila solicita un préstamo de 180.000 dólares para comprar su vivienda principal. Tiene un puntaje crediticio de 780, destina apenas el 22 % de sus ingresos al pago de deudas, aporta una cuota inicial que deja la relación préstamo-valor (LTV) en 65 %, y el crédito es de tasa fija sin periodos de interés únicamente ni amortización negativa.

Para el modelo, cada una de estas señales empuja en la misma dirección: un historial de pago sólido, una carga financiera cómoda y un colateral con margen amplio. El bosque aleatorio, que resume la decisión de cientos de árboles entrenados con casos históricos, clasifica a Camila en la clase 0 con alta confianza. En términos prácticos, es el tipo de expediente que el modelo acierta casi siempre: del reporte se deduce que los buenos pagadores se reconocen con un recall del 96 % (sin SMOTE), de modo que clientes como Camila no son frenados por falsas alertas.

=== Caso B: un incumplimiento que el modelo con SMOTE sí captura

Jorge solicita 320.000 dólares para un local comercial. Su puntaje crediticio es 590, su relación deuda-ingreso supera el 43 %, el LTV es del 95 % (apenas aporta cuota inicial) y el producto incluye amortización negativa, es decir, en los primeros años su deuda crece en lugar de reducirse.

Ninguna de estas señales, por sí sola, condena el crédito; es su combinación la que el bosque asocia con los incumplimientos observados en el pasado. Aquí aparece la diferencia entre las dos variantes: un modelo sin tratamiento del desbalance tiende a ser conservador con la clase minoritaria y puede dejar pasar perfiles como el de Jorge si las señales no son extremas. El modelo con SMOTE, entrenado con muchos más ejemplos de incumplimiento (reales y sintéticos), ha "visto" con más frecuencia esta combinación de condiciones y emite la alerta. El costo de esa sensibilidad es que algunos solicitantes intermedios, que habrían pagado sin problema, también quedan marcados: es la contrapartida de subir el recall del 85 % al 92 %. Para la entidad, la decisión de negocio es si prefiere revisar de más o arriesgar de más; el modelo con SMOTE elige revisar de más.

== Riesgos y cuidados

=== Los datos faltantes como fuente potencial de sesgo

En este dataset varias columnas relevantes (tasa de interés, ingreso, valor de la propiedad) tienen valores ausentes, y el pipeline los rellena con estadísticos simples: mediana para las numéricas y moda para las categóricas. El riesgo es que la ausencia no sea aleatoria. Si, por ejemplo, los solicitantes que no reportan su ingreso son precisamente quienes tienen ingresos informales o inestables, el dato faltante no es un hueco neutro: está correlacionado con el riesgo que se quiere predecir. Imputarlo con la mediana borra esa señal y, peor aún, la reemplaza por el valor de un solicitante "promedio" que no se parece al real. Las revisiones recientes sobre el tema coinciden en que la validez de cualquier imputación depende del mecanismo que generó la ausencia (MCAR, MAR o MNAR), y que los métodos simples solo son defendibles bajo supuestos fuertes de aleatoriedad (Emmanuel et al., 2021). Métodos más robustos como la imputación múltiple por ecuaciones encadenadas (MICE) permiten modelar la incertidumbre de cada valor imputado, pero exigen supuestos igualmente explícitos (van Buuren, 2018). La literatura metodológica más reciente advierte además que, cuando los datos faltan de forma no aleatoria (MNAR), incluso técnicas sofisticadas pueden incrementar el sesgo si las variables auxiliares solo predicen la ausencia y no el valor (Curnow et al., 2024).

=== La necesidad de documentar los supuestos de imputación

De lo anterior se desprende una exigencia práctica: toda decisión de imputación debe quedar documentada como un supuesto del análisis, no como un detalle técnico. Como mínimo debería registrarse la proporción de faltantes por variable, la estrategia aplicada, la justificación del mecanismo de ausencia que se asume, y el efecto de la imputación sobre los resultados (por ejemplo, comparando métricas con y sin las filas imputadas). Esta trazabilidad es además un requisito de explicabilidad: la literatura de IA explicable señala que un modelo solo es interpretable de manera honesta cuando se puede reconstruir la cadena completa de transformaciones que aplicó a los datos (Barredo Arrieta et al., 2020; Molnar, 2022). Un bosque aleatorio ya es una caja opaca por naturaleza; si además sus entradas provienen de imputaciones indocumentadas, cualquier explicación posterior (importancia de variables, SHAP, etc.) descansa sobre una base que nadie puede auditar. Documentar los supuestos no elimina el sesgo, pero lo hace visible y discutible, que es la condición mínima para confiar en el modelo en un contexto tan sensible como el otorgamiento de crédito.

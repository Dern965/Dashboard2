# Dashboard
Dashboard de análisis financiero y predicción de series de tiempo con el modelo **Gamma**.

# 📈 EDA + Gamma (BMV) • Multi-ticker

Aplicación desarrollada en **Streamlit** para analizar precios históricos de múltiples emisoras de la **Bolsa Mexicana de Valores (BMV)** y generar predicciones mediante el modelo **Gamma**.

La app permite explorar el comportamiento de los tickers, visualizar su evolución histórica, ejecutar predicciones con un horizonte fijo de **2 semanas**, evaluar resultados con métricas de error y comparar emisoras mediante un ranking.

---

## ✨ Funcionalidades principales

- 📊 Carga automática de datos desde un archivo CSV.
- 📈 Visualización histórica de múltiples tickers.
- 🔍 Análisis exploratorio por emisora.
- ⚙️ Predicción con el modelo **Gamma**.
- 📅 Horizonte de predicción **fijo a 2 semanas**.
- 📉 Métricas de evaluación:
  - **MAE**
  - **RMSE**
  - **MAPE**
  - **SMAPE**
  - **R²**
- 🏆 Ranking de emisoras según el desempeño estimado del modelo.

---

## 🧩 Requisitos

- **Python 3.10 o superior**
- Compatible con:
  - **Windows**
  - **macOS**
  - **Linux**

---

## 📦 Dependencias

Instala las dependencias del proyecto con:

```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución

1. Coloca tu archivo de datos dentro de la carpeta `datos/`, por ejemplo:

```txt
datos/market_prices.csv
```

2. Ejecuta la aplicación con:

```bash
streamlit run app_gamma.py
```

3. Streamlit abrirá la app en el navegador en:

```txt
http://localhost:8501
```

---

## 📄 Formato del archivo CSV

El archivo debe incluir al menos las siguientes columnas:

| Columna         | Descripción                    | Ejemplo       |
|-----------------|--------------------------------|---------------|
| `date`          | Fecha del registro             | 2024-01-31    |
| `instrument_id` | Identificador del ticker       | BIMBOA_MX     |
| `adj_close`     | Precio ajustado de cierre      | 77.45         |

Además, para mejorar el funcionamiento del modelo Gamma, también se recomienda incluir:

| Columna  | Descripción               |
|----------|---------------------------|
| `high`   | Precio máximo del periodo |
| `low`    | Precio mínimo del periodo |
| `volume` | Volumen operado           |

### Ejemplo de estructura

```csv
date,instrument_id,adj_close,high,low,volume
2023-01-31,BIMBOA_MX,77.45,78.10,76.90,1200000
2023-02-01,BIMBOA_MX,78.10,78.50,77.80,1100000
2023-01-31,WALMEX_MX,62.30,62.90,61.80,950000
2023-02-01,WALMEX_MX,63.05,63.40,62.70,980000
```

---

## 🧭 Uso de la aplicación

### 1. Carga de datos
La aplicación carga automáticamente el archivo definido en el código, por lo que es importante verificar que el CSV exista en la ruta esperada.

### 2. Configuración fija del modelo
La app ya no usa selección manual de:

- frecuencia
- periodos
- horizonte de predicción

El sistema trabaja con una configuración fija para realizar predicciones a **2 semanas**, lo que estandariza el análisis para todas las emisoras.

### 3. Pestañas principales

#### 📊 Resumen multi-ticker
Muestra una vista general de varias emisoras, permitiendo comparar visualmente su evolución histórica.

#### 🔍 EDA por ticker
Permite explorar una emisora de forma individual, mostrando su serie y un resumen estadístico básico.

#### 📈 Gamma por ticker
Ejecuta el modelo Gamma para una emisora específica y genera la predicción a **2 semanas** junto con sus métricas de evaluación.

#### 🏆 Ranking Gamma
Construye una comparación entre múltiples emisoras para identificar cuáles presentan mejor desempeño estimado según el modelo.

---

## ⚙️ Métricas de evaluación

La aplicación incorpora las siguientes métricas:

- **MAE (Mean Absolute Error)**  
  Error absoluto promedio entre valores reales y predichos.

- **RMSE (Root Mean Squared Error)**  
  Penaliza con mayor peso los errores grandes.

- **MAPE (Mean Absolute Percentage Error)**  
  Mide el error promedio en términos porcentuales.

- **SMAPE (Symmetric Mean Absolute Percentage Error)**  
  Variante porcentual simétrica, útil para comparar errores relativos.

- **R² (Coeficiente de determinación)**  
  Indica qué tan bien el modelo explica la variabilidad observada.

---

## 🧠 Modelo utilizado

Esta versión del dashboard **ya no utiliza ARIMA ni SARIMA**.

El sistema ahora se basa en el modelo **Gamma**, diseñado para generar predicciones a partir de la dinámica reciente de la serie y de las variables auxiliares disponibles en el dataset.

### Cambios principales respecto a la versión anterior

- Se eliminó el modelado con **ARIMA/SARIMA**.
- Se eliminó la selección de **frecuencia** y **horizonte** desde la interfaz.
- Se fijó el horizonte de predicción a **2 semanas**.
- Se añadieron las métricas **MAPE** y **SMAPE**.

---

## 🤖 Actualización automática de datos (GitHub Actions)

El proyecto incluye un pipeline de **actualización incremental automática** que se ejecuta **todos los días hábiles a las 7:00 AM (hora Ciudad de México)** sin que necesites hacer nada manualmente.

### ¿Cómo funciona?

```
Cada día hábil 7:00 AM CDMX
        │
GitHub Actions (servidores de GitHub)
        │
   update_data.py
        │
   Lee market_prices.csv → detecta la última fecha de cada ticker
        │
   Descarga SOLO los días nuevos desde yfinance (incremental)
        │
   Combina + elimina duplicados
        │
   git commit + push automático al repositorio
```

### ¿Necesito herramientas externas o de pago?

**No.** GitHub Actions está incluido en tu cuenta de GitHub sin costo adicional.

| Herramienta | ¿Necesaria? | Costo |
|---|---|---|
| GitHub Actions | ✅ Sí | Gratis (incluido en GitHub) |
| Servidor propio / VPS | ❌ No | — |
| Servicios como Railway, Heroku, etc. | ❌ No | — |

> **Nota:** Los repositorios privados tienen 2,000 minutos gratuitos al mes. Este workflow consume ~110 minutos al mes (5 min × 22 días hábiles), bien dentro del límite.

### Activación (2 pasos)

**Paso 1 — Dar permisos de escritura al workflow:**

1. Ve a tu repositorio en GitHub
2. `Settings` → `Actions` → `General`
3. Baja hasta **"Workflow permissions"**
4. Selecciona ✅ **"Read and write permissions"**
5. Clic en **Save**

**Paso 2 — Verificar la ruta del archivo:**

El archivo `update_data.yml` debe estar exactamente en `.github/workflows/update_data.yml` dentro del repositorio.

### Probar manualmente

Para ejecutar el workflow sin esperar al horario automático:

1. Ve a la pestaña **"Actions"** de tu repositorio en GitHub
2. Selecciona **"📈 Actualización diaria de datos BMV"**
3. Clic en **"Run workflow"** → **"Run workflow"**

También puedes correrlo localmente:

```bash
python update_data.py
```

---

## 🧪 Recomendaciones de uso

- Asegúrate de que los datos estén ordenados por fecha.
- Verifica que no existan valores nulos en columnas importantes.
- Usa suficientes observaciones históricas por ticker.
- Incluye `high`, `low` y `volume` si el flujo Gamma los requiere.
- Interpreta el ranking como una herramienta de apoyo analítico, no como una recomendación financiera definitiva.

---

## 🛠️ Solución de errores comunes

| Problema | Posible causa | Solución |
|----------|----------------|----------|
| ❌ No se pudo leer el CSV | Ruta incorrecta o archivo inexistente | Verifica la ruta y el nombre del archivo |
| ⚠️ Serie vacía | El ticker no tiene suficientes datos | Revisa el contenido del CSV |
| ❗ Error al calcular métricas | Muy pocas observaciones para evaluar | Usa una serie más larga |
| 📉 Predicción poco estable | Datos insuficientes o incompletos | Agrega más historial y valida columnas auxiliares |
| 🔄 El workflow no hace commit | Sin permisos de escritura en Actions | Ver **Paso 1** de la sección de activación |
| 🔄 El workflow no aparece en Actions | Archivo en ruta incorrecta | Mover a `.github/workflows/update_data.yml` |

---

## 🧱 Estructura del proyecto

```txt
.
├── app_gamma.py                        ← Aplicación Streamlit
├── update_data.py                      ← Script de actualización diaria
├── requirements.txt
├── README.md
├── datos/
│   ├── market_prices.csv               ← Se actualiza automáticamente
│   └── update.log                      ← Log de cada ejecución
└── .github/
    └── workflows/
        └── update_data.yml             ← Workflow de GitHub Actions
```


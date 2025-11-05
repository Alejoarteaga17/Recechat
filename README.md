## Recechat

¡Hola!   
Este es nuestro proyecto **Recechat**, desarrollado para el curso **1013 - Inteligencia Artificial**.

Recechat es un chatbot inteligente diseñado para interactuar de forma natural con los usuarios y ofrecer respuestas útiles basadas en procesamiento del lenguaje natural (NLP) y técnicas de inteligencia artificial.

---

### Integrantes del equipo

- **Alejandra Ortíz**  
- **Camila Vélez**  
- **Alejandro Arteaga**

---

### Requisitos previos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- **Python 3.10**
- **git**  
- **virtualenv**

---

### ⚙️ Instalación y ejecución

Sigue los pasos a continuación para ejecutar el proyecto localmente:

#### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/recechat.git
cd recechat
```
#### 2. Crear y activar un entorno virtual (opcional)

```bash
python -m venv env

venv\Scripts\activate
```

#### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### 4. Ejecutar Proyecto (desde la carpeta raiz)
- La primera vez que ejecutes el proyecto se descagarán las diferentes dependencias, además, se generan los Embeddings y TFI-DF (puede tardar +5min dependiento la capacidad de tu compu :) )
```bash
uvicorn src.server:app --reload --port 8000
```
#### 5. Abrir Recechat en el puerto habilitado 
- Abre la ip local ```http://127.0.0.1:8000/``` y listo puedes escribir ingredientes y mirar las recetas ofrecidas




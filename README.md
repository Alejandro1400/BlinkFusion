# **PulseSTORM**

This repository hosts a Python-based Streamlit application designed for data analysis and visualization of scientific data. The app includes various dashboards and tools for working with SOAC and STORM data.

---

## **Features**
- Filament analysis pipeline for SOAC data.
- STORM molecule merging and dashboard.
- Interactive user interface for data preprocessing and visualization.
- Integration with tools for file exploration and data manipulation.

---

## **Requirements**

### **1. Install Python**
Ensure you have Python installed on your system. We recommend downloading Python 3.9+ from the [official Python website](https://www.python.org/downloads/).

### **2. Install VS Code (Preferred IDE)**
We recommend using Visual Studio Code for editing and running the application. Download VS Code from [here](https://code.visualstudio.com/).

### **3. Install Dependencies**
Install them by running:
```bash
pip install -r requirements.txt
```
> Generate a `requirements.txt` file by including the above dependencies with their versions.

---

## **Installation**

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repository-url.git
cd your-repository-folder
```

### Step 2: Set Up the Environment
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Running the App**

### **Run the Streamlit App**
Start the app using the following command:
```bash
python -m streamlit run app.py
```

### **File Structure**
- `Analysis/`: Contains scripts for SOAC and STORM data processing.
- `Data_access/`: Handles file exploration and data saving.
- `UI/`: Includes UI components for the Streamlit application.
- `Docs/`: Documentation folder.
- `Dashboard/`: Dashboard-related components.

---

## **Example Usage**
1. Launch the app using the `streamlit` command.
2. Follow the sidebar options to choose preprocessing, analysis, or visualization tasks.
3. Explore outputs like histograms, scatter plots, or detailed analyses.

---

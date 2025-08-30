# ThunderSTORM User Guide Tutorial

This tutorial walks through the *manual step* in the blinking-statistics workflow: from a stack of STORM microscopy images, we extract a list of localizations using **ThunderSTORM** in **ImageJ**. Here’s how to install the tools (with links) and run the processing, while keeping everything clear and reproducible.

---

##  Prerequisites & Installation

### 1. Download and install **ImageJ (or Fiji)**
- Go to [imagej.net](https://imagej.net) or [fiji.sc](https://fiji.sc) to fetch the latest version for your operating system.

### 2. Download **ThunderSTORM plugin**
- Download the latest ThunderSTORM `.jar` plugin from the official GitHub Pages:
  - **Download link**: available on the ThunderSTORM homepage [ThunderSTORM](https://github.com/zitmen/thunderstorm?tab=readme-ov-file)
  - You can also get installation instructions and access example data via the ThunderSTORM GitHub wiki or releases [Link](https://github.com/zitmen/thunderstorm/wiki/Tutorials)

**Installation steps**:
- Copy the downloaded `.jar` file into your ImageJ (or Fiji) `plugins/` directory.
- Restart ImageJ to activate the plugin.

---

##  Processing Workflow

Now, once everything's installed, here’s how to process each pre-processed TIFF file (ensuring matching metadata):

1. **Open the image**  
   Use **File → Open…** and select “Open as HyperStack.” For visibility, you can optionally do **Process → Enhance Contrast**.

2. **Run ThunderSTORM analysis**  
   Navigate to **Plugins → ThunderSTORM → Run Analysis**, as shown in the screenshot below:

<img width="762" height="343" alt="image" src="https://github.com/user-attachments/assets/0e1da53f-64a5-405f-b16a-f0da2d437c2a" />
     

3. **Adjust processing parameters (optional but recommended)**  
   - Default settings are a good starting point.
   - Adjust **Sub-pixel localization** based on known particle size.
   - Enable **Multi-emitter fitting analysis** for dense samples, with the caveat of increased computation time.

<img width="515" height="973" alt="image" src="https://github.com/user-attachments/assets/5de09d7c-ef7c-4e58-8a30-961d77653e76" />
     

4. **Inspect results and post-process**  
   After running, you'll see a window with localization data such as x/y coordinates, intensity, and uncertainty:

<img width="1525" height="840" alt="image" src="https://github.com/user-attachments/assets/0c60caf2-a3f3-4660-a92e-609012489f17" />
     

   Perform at least:
   - **Remove duplicates**
   - **Drift correction**

5. **Export results**
   - Use **Export → CSV file**.
   - Save it to the same folder as your TIFF, using a filename like `filename_locs.csv`.

---

##  Quick Summary Table

| Step | Action |
|------|--------|
| 1 | Install ImageJ/Fiji |
| 2 | Download ThunderSTORM plugin |
| 3 | Place plugin JAR into `plugins/`, restart ImageJ |
| 4 | Open TIFF file as HyperStack |
| 5 | Run ThunderSTORM with default settings, tweak if needed |
| 6 | Use post-processing (remove duplicates & drift correction) |
| 7 | Export CSV file named `<filename>_locs.csv` |

Repeat for each file to prepare them for your BlinkFusion processing page.

- Maintain the same naming conventions between TIFF and CSV output (`filename_locs.csv`) to streamline automation.
- Consider enhancing your tutorial with sections on **batch processing using ImageJ macros**, if you want to scale to large datasets.

Let me know if you'd like help structuring batch automation or integration with BlinkFusion next!
::contentReference[oaicite:6]{index=6}

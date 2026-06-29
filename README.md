# 🚆 AI-Based OHE Wire Fault Detection and Monitoring System

## Overview

The **AI-Based OHE Wire Fault Detection and Monitoring System** is an intelligent web application developed for **Konkan Railway Corporation Limited (KRCL)** to automate the inspection of **Overhead Equipment (OHE) wires** using thermal images.

The system replaces manual thermal image analysis with an AI-assisted workflow that automatically extracts temperature information, detects potential faults, generates inspection reports, and stores all inspection records in a centralized PostgreSQL database.

---

## Problem Statement

Manual analysis of thermal images is time-consuming, requires expert interpretation, and lacks centralized digital record management. The objective of this project was to develop an automated system capable of:

* Detecting temperature variations from thermal images.
* Identifying abnormal OHE wire conditions.
* Automatically generating inspection reports.
* Maintaining centralized inspection records.
* Providing an easy-to-use web dashboard for railway personnel.

---

## Key Features

* 🔥 Automated Thermal Image Analysis
* 🤖 AI-Assisted Fault Detection
* 🌡 OCR-Based Temperature Extraction
* 📊 Automatic ΔT (Temperature Difference) Calculation
* 📄 Instant Inspection Report Generation
* ☁ Centralized PostgreSQL Database
* 📈 Real-Time Database Dashboard
* 📥 CSV Export of Inspection Records
* 🌐 Web-Based Streamlit Dashboard

---

## System Workflow

1. User enters inspection metadata (Image Name, Date, Time, Section, OHE Mast) in the Google Sheet.
2. User uploads the corresponding thermal image through the dashboard.
3. The application maps the uploaded image with its inspection metadata.
4. OCR extracts thermal scale temperatures from the image.
5. Computer Vision converts thermal colours into temperature values.
6. The contact wire temperature and ΔT are calculated.
7. The system classifies the inspection status based on predefined maintenance thresholds.
8. An inspection report is generated automatically.
9. The report is stored in the PostgreSQL database.
10. Users can view and download historical inspection records through the Database Dashboard.

---

## Technology Stack

### Programming Language

* Python 3.13.7

### Frontend & User Interface

* Streamlit 1.56.0
* HTML5
* CSS3

### Backend & Processing

* OpenCV 4.13
* NumPy 2.3.3
* Pandas 3.0.2
* Pytesseract 0.3.13
* Pillow 11.3.0

### Cloud & Database

* PostgreSQL
* Supabase 2.31.0
* Google Sheets API

### Deployment

* Streamlit Community Cloud
* GitHub

---

## Project Outcome

The developed solution successfully automates the complete OHE thermal inspection workflow by integrating Artificial Intelligence, Computer Vision, OCR, thermal image processing, cloud database management, and an interactive web dashboard.

The system reduces manual inspection effort, improves reporting accuracy, enables centralized digital record management, and supports faster maintenance decision-making for railway operations.

---

## Developed For

**Konkan Railway Corporation Limited (KRCL)**

Project-Based Internship (5 Months)
Dashboard link: https://railway-montering-and-fault-alert-system.streamlit.app/
---

## Author

**Mridula Kandalgaonkar**

B.Tech. – Artificial Intelligence & Data Science

Thakur College of Engineering and Technology

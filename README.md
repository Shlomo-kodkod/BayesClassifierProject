# Bayes Classifier Project

A Python project for text classification using the Naive Bayes algorithm.

## Project Overview
This project provides a simple framework for classifying text data using the Naive Bayes algorithm. It includes modules for data cleaning, model training, prediction, and a basic API and client interface. The project is structured to be modular and easy to extend.

## Main Features
- Data cleaning and preprocessing
- Training and testing a Naive Bayes classifier
- Simple API for model interaction
- Command-line and menu-based client interface

## Technologies Used
- Python 3
- Standard Python libraries 

## Project Structure
- `src/api/` - API server code (for model interaction)
- `src/client/` - Client-side scripts and main entry point
- `src/menu/` - Menu and manager modules for user interaction
- `src/models/` - Data cleaning and Naive Bayes model implementation
- `src/data_loader/` - Utility functions for data and file loading

## How to Run
1. Clone the repository.
2. (Optional) Create a virtual environment and activate it.
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python src/main.py
   ```
5. To run the API server:
   ```bash
   python src/api/app.py
   ```
6. To run the client application:
   ```bash
   python src/client/app.py
   ```

## Example Usage
You can use the menu interface to load data, train the model, and to get model accuracy. The API can also be used to send classification requests programmatically.

